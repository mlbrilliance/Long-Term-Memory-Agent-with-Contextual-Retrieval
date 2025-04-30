"""
Script to install all required dependencies for the evaluation benchmarks.
"""

import subprocess
import sys

import pkg_resources


def get_installed_packages():
    """Get a dictionary of installed packages and their versions."""
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}


def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"Installed {package}")


def ensure_packages_installed():
    """Ensure all required packages are installed."""
    required_packages = {
        "psutil": "7.0.0",
        "numpy": "1.20.0",
        "matplotlib": "3.5.0",
        "pandas": "1.3.0",
    }

    installed = get_installed_packages()

    for package, min_version in required_packages.items():
        if package not in installed:
            print(f"{package} not found. Installing...")
            install_package(package)
        else:
            print(f"{package} already installed (version {installed[package]})")


if __name__ == "__main__":
    print("Checking and installing required dependencies...")
    ensure_packages_installed()
    print("All dependencies installed successfully.")
