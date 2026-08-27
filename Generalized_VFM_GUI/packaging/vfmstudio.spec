# VFM Studio frozen build (later). Run from Generalized_VFM_GUI/:
#     pyinstaller packaging/vfmstudio.spec

from pathlib import Path

block_cipher = None
root = Path(SPECPATH).parent  # noqa: F821 — PyInstaller injects SPECPATH

a = Analysis(  # noqa: F821
    [str(root / "src" / "vfmstudio" / "__main__.py")],
    pathex=[str(root / "src")],
    binaries=[],
    datas=[
        (str(root / "src" / "vfmstudio" / "gui" / "style" / "app.qss"), "vfmstudio/gui/style"),
        (str(root / "src" / "vfmstudio" / "resources" / "icons"), "vfmstudio/resources/icons"),
    ],
    hiddenimports=["pyvistaqt", "vtk", "PySide6", "skfem"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VFMStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="VFMStudio",
)
