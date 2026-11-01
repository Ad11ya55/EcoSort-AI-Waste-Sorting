"""Editable, general-purpose disposal guidance for EcoSort.

Check local municipal rules before acting on this guidance. A material prediction
cannot determine whether an item is contaminated, coated, or accepted locally.
"""

# ============================================================================
# 1. CLASS-TO-DISPOSAL KNOWLEDGE LAYER
# Keep municipal advice separate from image recognition so rules can be reviewed.
# Verify each instruction against local recycling guidance before real-world use.
# ============================================================================
DISPOSAL_GUIDE = {
    "cardboard": {
        "bin": "Recycling",
        "emoji": "📦",
        "color": "#a66a2c",
        "steps": ["Flatten the box.", "Remove tape and packing materials when practical.", "Keep it clean and dry; greasy or wet cardboard belongs in general waste."],
        "note": "Clean, dry cardboard is commonly accepted in paper recycling streams.",
    },
    "glass": {
        "bin": "Glass recycling (check local rules)",
        "emoji": "🍾",
        "color": "#16845b",
        "steps": ["Empty and rinse the container.", "Separate lids if your local program asks you to.", "Do not put mirrors, window glass, light bulbs, or ceramics in bottle banks."],
        "note": "Bottle and jar glass is handled differently from heat-resistant or window glass.",
    },
    "metal": {
        "bin": "Recycling (check local rules)",
        "emoji": "🥫",
        "color": "#53768a",
        "steps": ["Empty and rinse cans and containers.", "Follow local guidance for sharp edges and metal lids.", "Aerosol cans must be completely empty; otherwise use an appropriate hazardous-waste service."],
        "note": "Metal containers are often recyclable when empty and accepted by the local program.",
    },
    "paper": {
        "bin": "Paper recycling",
        "emoji": "📄",
        "color": "#3975b7",
        "steps": ["Keep paper clean and dry.", "Remove plastic wrapping and other non-paper parts.", "Bag shredded paper if your local program requires it."],
        "note": "Food-soiled paper and some coated papers may not be accepted locally.",
    },
    "plastic": {
        "bin": "Plastic recycling (check the resin code)",
        "emoji": "🧴",
        "color": "#8056a8",
        "steps": ["Check the resin code and your local accepted-items list.", "Empty and rinse the container.", "Plastic film and bags often need a separate store drop-off."],
        "note": "A plastic appearance alone does not show whether an item is accepted for recycling.",
    },
    "trash": {
        "bin": "General waste / landfill",
        "emoji": "🗑️",
        "color": "#6b7280",
        "steps": ["Use the local general-waste bin for ordinary non-recyclables.", "Do not place batteries, electronics, chemicals, or medicines in household trash.", "Take hazardous or special items to an approved collection point."],
        "note": "This is a broad class; check special collection options for unusual items.",
    },
}

# ============================================================================
# 2. SAFE FALLBACK FOR UNCERTAIN OR UNMAPPED PREDICTIONS
# When confidence is low, advise the user to check before recycling.
# ============================================================================
UNCERTAIN_GUIDE = {
    "bin": "Not sure — do not recycle yet",
    "emoji": "❓",
    "color": "#b42318",
    "steps": ["Try a clearer photo with one item, good light, and the item filling the frame.", "Check the item label and your local waste guide.", "When in doubt, keep it out of recycling until you can confirm the correct route."],
    "note": "A model confidence score is not a guarantee that a prediction is correct.",
}

