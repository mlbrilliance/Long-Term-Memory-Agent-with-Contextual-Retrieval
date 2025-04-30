"""
Custom test runner script for the LTM Agent project.

This script provides clearer output about which tests are passing and failing.
"""

import subprocess
import sys
from pathlib import Path


def run_test(test_path, verbose=True):
    """Run a specific test and return success status."""
    print(f"\n{'=' * 80}")
    print(f"Running test: {test_path}")
    print(f"{'=' * 80}")

    cmd = ["python", "-m", "pytest", test_path]
    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    print(f"Exit code: {result.returncode}")
    return result.returncode == 0


def main():
    """Run tests in sequence and report overall status."""
    # Directory containing tests
    tests_dir = Path("tests/ltm_agent")

    # Map of test categories to specific test files/directories
    test_groups = {
        "Core": ["core/test_config.py", "core/test_enhanced_knowledge_unit.py"],
        "Utils": ["utils/test_timezone_handling.py"],
        "Retrieval": ["retrieval/test_context_retriever.py"],
        "LLM": ["llm/test_anthropic_client.py", "llm/test_subset.py"],
        "Memory": [
            "memory/test_base_memory.py",
            "memory/test_in_memory_store.py",
            "memory/test_cache.py",
            "memory/test_batch.py",
        ],
        "Integration": ["integration/test_end_to_end.py"],
    }

    # Results tracking
    results = {}

    # Run each test group
    for group, test_paths in test_groups.items():
        print(f"\n\n{'#' * 80}")
        print(f"# Testing {group} Components")
        print(f"{'#' * 80}")

        group_results = {}
        for test_path in test_paths:
            full_path = tests_dir / test_path
            if full_path.exists():
                success = run_test(str(full_path))
                group_results[test_path] = success
            else:
                print(f"Test file not found: {full_path}")
                group_results[test_path] = False

        results[group] = group_results

    # Print summary
    print("\n\n")
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True

    for group, group_results in results.items():
        group_passed = all(group_results.values())
        all_passed = all_passed and group_passed

        status = "[PASS]" if group_passed else "[FAIL]"
        print(f"{group}: {status}")

        for test_path, passed in group_results.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  - {test_path}: {status}")

    print("\nOverall Status:", "[PASS]" if all_passed else "[FAIL]")

    # Return exit code based on test results
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
