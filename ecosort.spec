# ============================================================================
# 1. PACKAGE CONFIGURATION AND REQUIRED FRONTEND DATA
# Build a one-folder Windows app so its runtime files sit beside EcoSort.exe.
# ============================================================================
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata, get_package_paths

# Include Streamlit's browser interface, package metadata, and native libraries.
# Explicit collection keeps the bundle focused and avoids unrelated test/demo files.
datas = collect_data_files("streamlit") + collect_data_files("torchvision")
for package in ["streamlit", "torch", "torchvision", "pandas", "numpy", "pyarrow", "Pillow", "pydeck", "altair"]:
    datas += copy_metadata(package)
binaries = collect_dynamic_libs("torch") + collect_dynamic_libs("torchvision")
torchvision_dir = Path(get_package_paths("torchvision")[1])
binaries += [(str(torchvision_dir / "_C_stable.pyd"), "torchvision")]
hiddenimports = [
    "streamlit.web.cli",
    "streamlit.hello",
    "torch",
    "torchvision",
    "torchvision._C_stable",
    "torchvision.transforms",
    "torchvision.models",
    "torchvision.models.mobilenetv2",
    "pandas",
    "numpy",
    "PIL",
]
hiddenimports += collect_submodules("streamlit.runtime")

# ============================================================================
# 2. ADD THE ECO SORT APPLICATION AND TRAINED MODEL FILES
# The app reads these files from the same bundle directory at runtime.
# ============================================================================
datas += [
    ("app.py", "."), ("disposal.py", "."), ("model.pt", "."), ("classes.json", "."),
    ("training_history.json", "."), ("training_metadata.json", "."), ("evaluation_results.json", "."),
]

# ============================================================================
# 3. ANALYZE PYTHON IMPORTS AND BUILD THE EXECUTABLE
# The custom hook includes TorchVision's compiled Windows extension.
# ============================================================================
a = Analysis(
    ["launch_ecosort.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "IPython", "notebook", "jupyter", "pandas.tests", "numpy.f2py.tests", "streamlit.testing"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

# Create a quiet desktop executable; COLLECT stores its supporting files beside it.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EcoSort",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="EcoSort",
)

