"""A file for initializing the repository.

This file should be deleted after initial setup.
"""

import json
import shutil
import subprocess
from pathlib import Path


def _replace_all_occurences(
    old: str,
    new: str,
    exclude: set[Path] | None = None,
) -> None:
    if exclude is None:
        exclude = set()
    # Get files in this repository (e.g., exclude venv/).
    known_files: set[Path] = set()
    outer_dir = Path(".").parent.resolve()
    proc = subprocess.run(
        ["git", "ls-files"], encoding="utf-8", stdout=subprocess.PIPE, check=True
    )
    for line in proc.stdout.split("\n"):
        known_files.add((outer_dir / line).resolve())
    known_files -= exclude
    for file_path in known_files:
        if file_path.is_dir():
            continue
        with file_path.open("r", encoding="utf-8") as fp:
            file_contents = fp.read()
        updated_contents = file_contents.replace(old, new)
        if updated_contents != file_contents:
            with file_path.open("w", encoding="utf-8") as file:
                file.write(updated_contents)


def _main() -> None:
    # Parse the config.
    outer_dir = Path(".").parent.resolve()
    config_file = outer_dir / "config.json"
    assert config_file.exists(), "Missing config file"
    with open(config_file, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    # Validate the config.
    assert "developer" in config, "Missing developer name in config file"
    developer = config["developer"]
    github_username = config["github-username"]
    assert not " " in github_username, "Malformed GitHub username"
    package_name = config["your-package-name"]
    assert (
        " " not in package_name
    ), "Package names cannot contain spaces (you want to `import package_name`)"
    assert (
        "-" not in package_name
    ), "Package names cannot dashes (you want to `import package_name`)"
    python_version = config["python-version"]
    assert python_version.startswith("3"), "Only Python 3 is supported"
    assert python_version.startswith(
        "3."
    ), "Missing dot in Python version (example: 3.10)"
    python_subversion = python_version.split(".")[1]
    assert python_subversion.isdigit()

    # Get the repository name from this directory.
    repo_name = outer_dir.name

    # Delete the existing git files if they are from the starter repo.
    git_repo = outer_dir / ".git"
    if git_repo.exists():
        git_config_file = git_repo / "config"
        with open(git_config_file, "r", encoding="utf-8") as fp:
            git_config_contents = fp.read()
        if "git@github.com:tomsilver/python-starter.git" in git_config_contents:
            shutil.rmtree(git_repo)

    # Initialize the repo anew.
    subprocess.run(["git", "init", "-b", "main"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            f"git@github.com:{github_username}/{repo_name}.git",
        ],
        check=True,
    )

    # Replace all occurrences of default names.
    substitutions = {
        "Tom Silver": developer,
        "tomsilver": github_username,
        "python-starter": repo_name,
        "python_starter": package_name,
        "3.10": f"3.{python_subversion}",
        "310": f"3{python_subversion}",
    }
    for old, new in substitutions.items():
        _replace_all_occurences(
            old, new, exclude={outer_dir / "apply_configuration.py", config_file}
        )

    # Rename the package repo.
    subprocess.run(["mv", "src/python_starter", f"src/{package_name}"], check=True)


if __name__ == "__main__":
    _main()