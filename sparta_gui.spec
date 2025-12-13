# PyInstaller spec for Sparta Remix GUI
block_cipher = None

def resolve_path(rel_path):
    import os
    from pathlib import Path
    return str((Path(SPECPATH).resolve() / rel_path).resolve())

ffmpeg_exe = None
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    ffmpeg_exe = None

def _safe_copy_metadata(dist_name):
    try:
        from PyInstaller.utils.hooks import copy_metadata
        return copy_metadata(dist_name)
    except Exception:
        return []

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[
        (resolve_path('rubberband.exe'), '.'),
        (resolve_path('sndfile.dll'), '.'),
        (resolve_path('PitchCorrector297_V1.0/PitchCorrector297/bin/Debug/PitchCorrector297.exe'), 'PitchCorrector297_V1.0/PitchCorrector297/bin/Debug'),
        (resolve_path('PitchCorrector297_V1.0/PitchCorrector297/bin/Debug/PitchCorrector297.exe.config'), 'PitchCorrector297_V1.0/PitchCorrector297/bin/Debug'),
        (resolve_path('PitchCorrector297_V1.0/PitchCorrector297/bin/Debug/AxInterop.WMPLib.dll'), 'PitchCorrector297_V1.0/PitchCorrector297/bin/Debug'),
        (resolve_path('PitchCorrector297_V1.0/PitchCorrector297/bin/Debug/Interop.WMPLib.dll'), 'PitchCorrector297_V1.0/PitchCorrector297/bin/Debug'),
        (resolve_path('PitchCorrector297_V1.0/PitchCorrector297/bin/Debug/NAudio.dll'), 'PitchCorrector297_V1.0/PitchCorrector297/bin/Debug'),
        (resolve_path('PitchCorrector297_V1.0/PitchCorrector297/bin/Debug/NAudio.xml'), 'PitchCorrector297_V1.0/PitchCorrector297/bin/Debug'),
    ],
    datas=(
        [
            (resolve_path('build/gun/GUN.wav'), 'build/gun'),
            (resolve_path('base.wav'), '.'),
            (resolve_path('icon.png'), '.'),
        ]
        + _safe_copy_metadata('imageio')
        + _safe_copy_metadata('moviepy')
        + _safe_copy_metadata('imageio-ffmpeg')
        + _safe_copy_metadata('imageio_ffmpeg')
        + ([(ffmpeg_exe, '.')] if ffmpeg_exe else [])
    ),
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='sparta-remix-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # change to True to keep console
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='sparta-remix-gui'
)
