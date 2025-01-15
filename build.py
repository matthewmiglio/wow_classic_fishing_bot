import time
import shutil
import os
import datetime
import argparse
from cx_Freeze import Executable, setup


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="Build script for WoW Fishing Bot")
    parser.add_argument(
        "-v", "--version",
        help="Version number in the format vX.Y.Z (e.g., v0.1.22)",
        required=True
    )
    args = parser.parse_args()

    # Validate version format
    if not args.version.startswith('v') or not args.version[1:].replace('.', '').isdigit():
        raise ValueError(f"Invalid version format: {args.version}. Expected format: vX.Y.Z")

    return args.version


def get_include_files(top_dir, skip_folders, skip_file_types):
    file_paths = []

    for root, dirs, files in os.walk(top_dir):
        # Skip directories listed in skip_folders
        dirs[:] = [d for d in dirs if d not in skip_folders]

        for file in files:
            # Skip files with extensions listed in skip_file_types
            if any(file.endswith(ext) for ext in skip_file_types):
                continue

            # If not skipped, add the file path to the list
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths


def main():
    version = parse_arguments()
    PROJECT_NAME = f"matt-fishbot-{version}"
    AUTHOR = "Matthew Miglio"
    DESCRIPTION = "Automated WoW Fishing Bot"
    KEYWORDS = "World of Warcraft Classic, Fishing, Bot"
    COPYRIGHT = "2024 Matthew Miglio"
    ENTRY_POINT = "src/__main__.py"
    GUI = False
    UPGRADE_CODE = "{3f9f4225-8af4-4024-97fd-9a2329638315}"

    dist_dir = os.path.join(os.getcwd(), "dist")

    # Collect files for inclusion
    skip_file_types = [
        ".txt",
        ".png",
        ".jpg",
        ".ipynb",
        ".md",
        ".pyc",
        ".msi",
        ".gitignore",
        ".lock",
        ".toml",
    ]

    skip_folders = [
        "data_export",
        "build",
        ".git",
    ]

    skip_files = [
        "build.py",
    ]

    # Collect the files to include
    files_to_include = [
        (path, os.path.relpath(path, os.getcwd()))  # Relative paths for other files
        for path in get_include_files(
            top_dir=os.getcwd(),
            skip_folders=skip_folders,
            skip_file_types=skip_file_types,
        )
        if os.path.basename(path) not in skip_files
    ]

    build_exe_options = {
        "excludes": ["test", "setuptools"],
        "include_files": files_to_include,
        "include_msvcr": True,
        "build_exe": dist_dir,  # Place build output in the dist folder
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

    print("\nThese are the files_to_include")
    for f in files_to_include:
        print(f"\t{f}")

    print("\nThese are the build exe options")
    for k, v in build_exe_options.items():
        if k == "include_files":
            continue
        print(f"\t{k}: {v}")

    print("\nThese are the bdist_msi options")
    for k, v in bdist_msi_options.items():
        if k == "include_files":
            continue
        if k == "summary_data":
            print(f"\t{k}:")
            for k2, v2 in v.items():
                print(f"\t  {k2}: {v2}")
        else:
            print(f"\t{k}: {v}")

    exe = Executable(
        script=ENTRY_POINT,
        base="Win32GUI" if GUI else None,
        uac_admin=True,
        shortcut_name=f"{PROJECT_NAME} {version}",
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


def delete_build_folder():
    build_folder_path = os.path.join(os.getcwd(), "build")
    if not os.path.exists(build_folder_path):
        return
    print(build_folder_path)
    shutil.rmtree(build_folder_path)


if __name__ == "__main__":
    start_time = time.time()
    main()
    # delete_build_folder()
    end_time = time.time()
    time_taken_readable_hms = str(
        datetime.timedelta(seconds=int(end_time - start_time))
    )
    print(f"Built in {time_taken_readable_hms}")
