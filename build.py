import os
from cx_Freeze import Executable, setup

PROJECT_NAME = "Matt's WoW Fish Bot"
AUTHOR = "Matthew Miglio"
DESCRIPTION = "Automated WoW Fishing Bot"
KEYWORDS = "World of Warcraft Classic, Fishing, Bot"
COPYRIGHT = "2024 Matthew Miglio"
ENTRY_POINT = "bot.py"
GUI = False
UPGRADE_CODE = "{3f9f4225-8af4-4024-97fd-9a2329638315}"
VERSION = "v0.0.0"

# Collect files for inclusion
files_to_include = []
skip_folders = ['data_export',]

# Helper function to add all files from a directory, excluding .png files
def add_files_from_dir(dir_path, target_dir=""):
    if os.path.exists(dir_path):
        for folder_name, subfolders, filenames in os.walk(dir_path):
            if folder_name in skip_folders: continue
            for filename in filenames:
                # Skip .png files
                if filename.lower().endswith(".png") or filename.lower().endswith(".txt"):
                    continue

                file_path = os.path.join(folder_name, filename)
                # Calculate relative path to preserve directory structure in the build
                relative_path = os.path.relpath(file_path, start=dir_path)
                # Include the file and maintain its structure
                files_to_include.append(
                    (file_path, os.path.join(target_dir, relative_path))
                )
    else:
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")

# Add files from model directories
add_files_from_dir("inference/bobber_models", "inference/bobber_models")
add_files_from_dir("inference/splash_models", "inference/splash_models")
add_files_from_dir("data_export", "data_export")
add_files_from_dir("logs", "logs")
add_files_from_dir("save_images", "save_images")

# Verify the files (check that the files actually exist)
for file, _ in files_to_include:
    if not os.path.isfile(file):
        raise FileNotFoundError(f"An included file is missing: {file}")

# Build executable options
build_exe_options = {
    "excludes": ["test", "setuptools"],
    "include_files": files_to_include,  # Include all the files in the build
    "include_msvcr": True,
}

bdist_msi_options = {
    "upgrade_code": UPGRADE_CODE,
    "add_to_path": False,
    "initial_target_dir": f"[ProgramFilesFolder]\\{PROJECT_NAME}",
    "summary_data": {
        "author": AUTHOR,
        "comments": DESCRIPTION,
        "keywords": KEYWORDS,
    },
}

exe = Executable(
    script=ENTRY_POINT,
    base="Win32GUI" if GUI else None,
    uac_admin=True,
    shortcut_name=f"{PROJECT_NAME} {VERSION}",
    shortcut_dir="DesktopFolder",
    target_name=f"{PROJECT_NAME}.exe",
    copyright=COPYRIGHT,
)

setup(
    name=PROJECT_NAME,
    description=DESCRIPTION,
    executables=[exe],
    options={
        "bdist_msi": bdist_msi_options,
        "build_exe": build_exe_options,
    },
)

# poetry run python build.py bdist_msi
