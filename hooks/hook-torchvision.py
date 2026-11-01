# ============================================================================
# 1. INCLUDE THE TORCHVISION WINDOWS EXTENSION IN THE APP BUNDLE
# TorchVision 0.29 names this compiled extension _C_stable; PyInstaller needs
# it listed explicitly so image transforms work in the packaged executable.
# ============================================================================
hiddenimports = ["torchvision._C_stable"]
module_collection_mode = "pyz+py"
