"""EcoSort: AI-based waste identification and general disposal guidance."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

from disposal import DISPOSAL_GUIDE, UNCERTAIN_GUIDE


# ============================================================================
# 1. PROJECT PATHS AND MODEL SETTINGS
# Paths are relative to this file so the app can run from source or its bundle.
# ============================================================================
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pt"
CLASSES_PATH = ROOT / "classes.json"
HISTORY_PATH = ROOT / "training_history.json"
EVALUATION_PATH = ROOT / "evaluation_results.json"
METADATA_PATH = ROOT / "training_metadata.json"
CONFIDENCE_THRESHOLD = 0.60
TOP_K = 3
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

st.set_page_config(page_title="EcoSort | Waste Sorting Assistant", page_icon="♻️", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1120px; padding-top: 1.6rem;}
  .eco-subtitle {color: #53615a; font-size: 1.08rem; margin-top: -.6rem;}
  .eco-note {padding: .8rem 1rem; border-radius: .65rem; background: #f1f7f3; border-left: 4px solid #2e8b57;}
  .eco-card {padding: 1.25rem 1.4rem; border-radius: .9rem; color: white; margin: .5rem 0 1rem;}
  div[data-testid="stMetric"] {background: #f7faf8; border: 1px solid #e4ece7; padding: .8rem; border-radius: .75rem;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. LOAD THE TRAINED MODEL ONCE
# Streamlit caches the model so each image scan only needs a forward pass.
# ============================================================================
@st.cache_resource(show_spinner="Loading the EcoSort model…")
def load_model(model_path: str, classes_path: str, model_stamp: float, classes_stamp: float):
    del model_stamp, classes_stamp  # Cache keys invalidate the resource when artifacts change.
    classes = json.loads(Path(classes_path).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, classes, device


# ============================================================================
# 3. PREPROCESS AN IMAGE AND GET THE TOP PREDICTIONS
# Keep preprocessing identical to validation: RGB, 224 x 224, ImageNet scaling.
# ============================================================================
def predict(image: Image.Image, model, classes: list[str], device):
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    count = min(TOP_K, len(classes))
    values, indices = torch.topk(probabilities, count)
    return [(classes[index], float(value)) for value, index in zip(values.cpu(), indices.cpu())]


# ============================================================================
# 4. SHOW THE DISPOSAL GUIDANCE
# The class label comes from the model; bin advice comes from disposal.py.
# ============================================================================
def show_guidance(guide: dict, confidence: float | None = None):
    st.markdown(
        f'<div class="eco-card" style="background:{guide["color"]}"><div style="font-size:2rem">{guide["emoji"]}</div>'
        f'<div style="font-size:1.4rem;font-weight:700">{guide["bin"]}</div></div>',
        unsafe_allow_html=True,
    )
    if confidence is not None:
        st.write(f"**Confidence:** {confidence:.1%}")
        st.progress(min(max(confidence, 0.0), 1.0))
    st.write("**How to prepare it**")
    for step in guide["steps"]:
        st.markdown(f"- {step}")
    st.info(f"Did you know? {guide['note']}")


# ============================================================================
# 5. RECORD ONE SCAN IN THE CURRENT BROWSER SESSION
# Session state keeps notes for this session without saving the photo to disk.
# ============================================================================
def record_scan(label: str, prediction: str, confidence: float, notes: str = ""):
    st.session_state.setdefault("scans", []).append({
        "item": label.strip() or "Unlabelled item",
        "true_class": "",
        "predicted_class": prediction,
        "confidence": confidence,
        "correct": "",
        "notes": notes,
    })


# ============================================================================
# 6. CAPTURE OR UPLOAD AN IMAGE, THEN APPLY THE CONFIDENCE RULE
# Under 60%, show the cautious fallback instead of recommending a recycling bin.
# ============================================================================
def render_scan(model_state):
    st.subheader("Scan one item")
    st.caption("For the best result, photograph one item in good light with the item filling most of the frame.")
    camera_tab, upload_tab = st.tabs(["📷 Camera", "⬆️ Upload"])
    with camera_tab:
        camera_file = st.camera_input("Take a photo")
    with upload_tab:
        upload_file = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])

    selected = camera_file or upload_file
    if selected is None:
        st.markdown('<div class="eco-note">Add a photo to identify its visible material. The model does not verify contamination or local recyclability.</div>', unsafe_allow_html=True)
        return
    try:
        image = Image.open(selected)
        st.image(image, caption="Photo to classify", use_container_width=True)
    except Exception as exc:
        st.error(f"This image could not be opened: {exc}")
        return
    if model_state is None:
        st.warning("The trained model files are not available yet. Follow the setup instructions in the sidebar, then restart or refresh the app.")
        return
    model, classes, device = model_state
    try:
        predictions = predict(image, model, classes, device)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return
    label, confidence = predictions[0]
    st.divider()
    st.subheader("Sorting guidance")
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(f"Not confident enough to recommend a bin (best guess: **{label.title()}**).")
        show_guidance(UNCERTAIN_GUIDE, confidence)
    else:
        guide = DISPOSAL_GUIDE.get(label, UNCERTAIN_GUIDE)
        st.caption(f"Detected material: **{label.title()}**")
        show_guidance(guide, confidence)
    with st.expander("Other possibilities"):
        for name, score in predictions[1:]:
            st.write(f"{name.title()} — {score:.1%}")

    st.markdown("**Add this photo to your real-world test log**")
    c1, c2 = st.columns([2, 1])
    with c1:
        item_name = st.text_input("Item name / test note", key="current_item_name", placeholder="e.g. rinsed drinks can")
    with c2:
        true_class = st.selectbox("Known class (optional)", ["Not recorded"] + (classes if model_state else []), key="current_true_class")
    if st.button("Save scan to session log", type="secondary"):
        st.session_state.setdefault("scans", []).append({
            "item": item_name.strip() or "Unlabelled item",
            "true_class": "" if true_class == "Not recorded" else true_class,
            "predicted_class": label,
            "confidence": confidence,
            "correct": "" if true_class == "Not recorded" else str(label == true_class),
            "notes": "Camera or uploaded photo",
        })
        st.success("Saved for this browser session.")


def render_history():
    # Summarize scans, calculate optional labelled accuracy, and export the log.
    st.subheader("Session history and real-world test log")
    records = st.session_state.get("scans", [])
    if not records:
        st.info("No scans saved yet. Scan an item and choose ‘Save scan to session log’ to build a test record.")
        return
    frame = pd.DataFrame(records)
    known = frame[frame["true_class"].astype(str).str.len() > 0]
    recyclable_classes = {"cardboard", "glass", "metal", "paper", "plastic"}
    recyclable_share = frame["predicted_class"].isin(recyclable_classes).mean()
    cols = st.columns(3)
    cols[0].metric("Items scanned", len(frame))
    cols[1].metric("Routed to a recycling category", f"{recyclable_share:.0%}")
    cols[2].metric("Labelled test photos", len(known))
    if not known.empty:
        correct = known["correct"].astype(str).str.lower().eq("true")
        accuracy = correct.mean()
        cols2 = st.columns(2)
        cols2[0].metric("Real-world accuracy (labelled scans)", f"{accuracy:.1%}")
        cols2[1].metric("Correct / labelled", f"{int(correct.sum())} / {len(known)}")
        st.caption("This is a small session sample, not a benchmark. Gather varied items and photos before drawing conclusions.")
        band = pd.cut(known["confidence"], bins=[-0.001, .60, .75, .90, 1.001], labels=["Below 60% (abstain)", "60–74%", "75–89%", "90–100%"])
        calibration = pd.DataFrame({"confidence_band": band, "correct": correct.to_numpy()}).groupby("confidence_band", observed=False).agg(predictions=("correct", "size"), correct=("correct", "sum"))
        calibration["share_correct"] = calibration.apply(lambda row: row["correct"] / row["predictions"] if row["predictions"] else 0, axis=1)
        st.write("**Confidence versus correctness**")
        st.dataframe(calibration, use_container_width=True)
    st.dataframe(frame.rename(columns={"item": "Item", "true_class": "True class", "predicted_class": "Predicted class", "confidence": "Confidence", "correct": "Correct?", "notes": "Notes"}), use_container_width=True, hide_index=True)
    csv = frame.to_csv(index=False).encode("utf-8")
    st.download_button("Download session log as CSV", csv, "ecosort_real_world_test_log.csv", "text/csv")
    if st.button("Clear session history"):
        st.session_state["scans"] = []
        st.rerun()


def render_evaluation():
    # Read saved training outputs to show curves, metrics, and the confusion matrix.
    st.subheader("Model evaluation")
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        summary_cols = st.columns(4)
        summary_cols[0].metric("Dataset images", metadata.get("image_count", "—"))
        summary_cols[1].metric("Training images", metadata.get("train_images", "—"))
        summary_cols[2].metric("Validation images", metadata.get("validation_images", "—"))
        summary_cols[3].metric("Training time", f"{metadata.get('elapsed_seconds', 0)/60:.1f} min")
        with st.expander("Training run configuration"):
            st.json(metadata)
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if history.get("train_loss"):
            history_frame = pd.DataFrame({
                "Epoch": range(1, len(history["train_loss"]) + 1),
                "Training loss": history["train_loss"],
                "Validation loss": history["validation_loss"],
                "Training accuracy": history["train_accuracy"],
                "Validation accuracy": history["validation_accuracy"],
            }).set_index("Epoch")
            st.write("**Training curves**")
            st.line_chart(history_frame[["Training loss", "Validation loss"]])
            st.line_chart(history_frame[["Training accuracy", "Validation accuracy"]])
            st.dataframe(history_frame.style.format("{:.3f}"), use_container_width=True)
    if EVALUATION_PATH.exists():
        result = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
        a, b = st.columns(2)
        a.metric("Validation / evaluation accuracy", f"{result['accuracy']:.1%}")
        b.metric("Macro F1", f"{result['macro']['f1']:.3f}")
        st.caption(result.get("evaluation_set", "Separate evaluation data; see dataset notes."))
        st.write("**Per-class precision, recall, and F1**")
        st.dataframe(pd.DataFrame(result["per_class"]).T.rename_axis("Class"), use_container_width=True)
        st.write("**Confusion matrix (rows: actual; columns: predicted)**")
        st.dataframe(pd.DataFrame(result["confusion_matrix"], index=result["class_order"], columns=result["class_order"]), use_container_width=True)
    else:
        st.info("No evaluation_results.json yet. Run evaluate.py on a labelled folder-per-class dataset to create per-class metrics and a confusion matrix.")
    st.markdown("**Interpretation reminder:** the training validation split also selects the best checkpoint, so its score can be optimistic. Keep a separate held-out test set for a stronger estimate.")


def render_dataset():
    # Let users inspect a local class-folder dataset and the guidance lookup.
    st.subheader("Dataset and class architecture")
    st.write("EcoSort is designed for TrashNet-style images: one folder per category. The six supported material labels are `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`.")
    data_dir = st.text_input("Optional local dataset folder", placeholder=r"C:\path\to\trashnet\datasets\garbage_classification")
    if data_dir:
        path = Path(data_dir).expanduser()
        if path.is_dir():
            rows = []
            for class_dir in sorted(p for p in path.iterdir() if p.is_dir()):
                count = sum(1 for p in class_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
                rows.append({"Class": class_dir.name, "Images": count})
            frame = pd.DataFrame(rows)
            if len(frame):
                frame["Share"] = frame["Images"] / max(frame["Images"].sum(), 1)
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.metric("Total images", int(frame["Images"].sum()))
        else:
            st.warning("That folder is not accessible to the app process.")
    st.write("**Class-to-guidance lookup**")
    st.dataframe(pd.DataFrame([{"Material class": key.title(), "Suggested route": value["bin"], "Preparation steps": " ".join(value["steps"])} for key, value in DISPOSAL_GUIDE.items()]), use_container_width=True, hide_index=True)


def main():
    # Build the five-page experience and load model artifacts before scanning.
    st.title("♻️ EcoSort")
    st.markdown('<p class="eco-subtitle">AI-based waste identification and disposal guidance</p>', unsafe_allow_html=True)
    st.markdown("An educational image-classification prototype aligned with **UN SDG 12: Responsible Consumption and Production**.")
    st.warning("General educational guidance only. Recycling rules vary by location; follow your municipal guide. The model recognizes appearance, not contamination, resin codes, or actual local recyclability.")

    model_state = None
    if MODEL_PATH.is_file() and CLASSES_PATH.is_file():
        try:
            model_state = load_model(str(MODEL_PATH), str(CLASSES_PATH), MODEL_PATH.stat().st_mtime, CLASSES_PATH.stat().st_mtime)
        except Exception as exc:
            st.error(f"The saved model could not be loaded: {exc}. Retrain it using train.py and make sure classes.json matches the checkpoint.")
    else:
        st.warning("The trained model files are not available, so image predictions are disabled. See the project setup instructions in the README.")

    scan_tab, history_tab, evaluation_tab, data_tab, about_tab = st.tabs(["📸 Scan", "🧾 Session history", "📊 Evaluation", "🗂️ Dataset & guidance", "ℹ️ About & limitations"])
    with scan_tab:
        render_scan(model_state)
    with history_tab:
        render_history()
    with evaluation_tab:
        render_evaluation()
    with data_tab:
        render_dataset()
    with about_tab:
        st.subheader("How it works")
        st.markdown("""
        1. A user takes or uploads a photo of one item.
        2. The image is converted to RGB, resized to 224×224, and normalized with ImageNet statistics.
        3. A MobileNetV2 transfer-learning model predicts one of six visible material categories.
        4. The app displays the top prediction and up to two alternatives.
        5. If confidence is below 60%, it abstains and gives a cautious fallback. Otherwise, a separate editable Python lookup supplies general preparation guidance.
        """)
        st.subheader("Model configuration")
        st.markdown(f"**Model:** MobileNetV2 · 224×224 input · 6 material classes  \n**Abstention threshold:** {CONFIDENCE_THRESHOLD:.0%}")
        st.subheader("Limitations and responsible use")
        st.markdown("""
        - TrashNet contains mostly single objects against plain backgrounds; everyday scenes can be harder.
        - The six classes do not include food waste, batteries, electronics, textiles, or medical waste.
        - The classifier identifies visual material, not whether an item is greasy, coated, empty, or accepted by a local program.
        - A softmax confidence is not a probability guarantee; confident errors can happen.
        - Validation results can be optimistic when the same split is used for checkpoint selection. Use a separate held-out test set and real-world photos.
        - Photos are processed in memory for the current session. Session history stays in browser session state and can be downloaded as CSV; it is not written by this app to a server-side history file.
        """)
        st.subheader("Project code files")
        st.markdown("- `app.py`: Streamlit interface, inference, session log, metrics display\n- `train.py`: TrashNet loading, augmentation, two-phase training, checkpointing\n- `evaluate.py`: per-class metrics and confusion matrix\n- `disposal.py`: editable class-to-guidance rules\n- `requirements.txt`: Python dependencies")


if __name__ == "__main__":
    main()

