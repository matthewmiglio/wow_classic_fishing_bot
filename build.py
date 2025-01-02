import shutil
import os
import datetime
from cx_Freeze import Executable, setup


class Versioning:
    def __init__(self):
        self.version_file_path = "version.txt"
        self.default_version = 10

    def read_version(self):
        try:
            with open(self.version_file_path, "r") as f:
                version_index = int(f.read().strip())
                return version_index
        except:
            with open(self.version_file_path, "w") as f:
                f.write(str(self.default_version))
                return self.default_version

    def get_version(self):
        v = self.read_version()
        self.increment_version()
        return v

    def increment_version(self):
        version_index = self.read_version()
        with open(self.version_file_path, "w") as f:
            f.write(str(version_index + 1))
        return version_index + 1


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


def get_most_recent_onnx_files():
    bobber_models_folder = os.path.join(os.getcwd(), "inference", "bobber_models")
    splash_models_folder = os.path.join(os.getcwd(), "inference", "splash_models")

    most_recent_bobber_model_path = os.path.join(
        bobber_models_folder, os.listdir(bobber_models_folder)[-1]
    )

    most_recent_splash_model_path = os.path.join(
        splash_models_folder, os.listdir(splash_models_folder)[-1]
    )

    return [most_recent_bobber_model_path, most_recent_splash_model_path]


def main():
    versioning = Versioning()
    this_version_index = versioning.get_version()
    PROJECT_NAME = f"MattFishBot {datetime.datetime.now().strftime('%Y-%m-%d')} {this_version_index}"
    AUTHOR = "Matthew Miglio"
    DESCRIPTION = "Automated WoW Fishing Bot"
    KEYWORDS = "World of Warcraft Classic, Fishing, Bot"
    COPYRIGHT = "2024 Matthew Miglio"
    ENTRY_POINT = "bot.py"
    GUI = False
    UPGRADE_CODE = "{3f9f4225-8af4-4024-97fd-9a2329638315}"
    VERSION = f"v0.0.{this_version_index}"

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
        ".onnx",
    ]

    skip_folders = [
        "data_export",
        "build",
        ".git",
    ]

    skip_files = [
        "build.py",
    ]

    files_to_include = [
        path
        for path in get_include_files(
            top_dir=os.getcwd(),
            skip_folders=skip_folders,
            skip_file_types=skip_file_types,
        )
        if os.path.basename(path) not in skip_files
    ] + get_most_recent_onnx_files()

    build_exe_options = {
        "excludes": ["test", "setuptools"],
        "include_files": files_to_include,
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


def delete_build_folder():
    build_folder_path = os.path.join(os.getcwd(), "build")
    print(build_folder_path)
    shutil.rmtree(build_folder_path)


main()
delete_build_folder()
# poetry run python build.py bdist_msi
