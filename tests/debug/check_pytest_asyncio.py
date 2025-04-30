"""
Script to check why pytest_asyncio isn't being properly recognized.
"""

import site
import sys
from pathlib import Path

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print("\nSite packages:")
for path in site.getsitepackages():
    print(f"  {path}")

# Try to locate the pytest_asyncio package
print("\nSearching for pytest_asyncio in site packages...")
for site_pkg in site.getsitepackages():
    site_path = Path(site_pkg)
    pytest_asyncio_paths = list(site_path.glob("**/pytest_asyncio*"))
    if pytest_asyncio_paths:
        print(f"\nFound in {site_pkg}:")
        for path in pytest_asyncio_paths:
            print(f"  {path}")

# Check if we can directly import pytest
print("\nAttempting to import pytest...")
try:
    import pytest

    print(f"  Success! Pytest version: {pytest.__version__}")
    print(f"  Pytest location: {pytest.__file__}")
except ImportError as e:
    print(f"  Failed: {e}")

# Check if pytest_asyncio is installed as an entry point
print("\nLooking for pytest_asyncio entry points...")
try:
    from importlib.metadata import entry_points

    for ep in entry_points(group="pytest11"):
        print(f"  {ep.name}: {ep.value}")
except Exception as e:
    print(f"  Error checking entry points: {e}")

print("\nCheck complete.")
