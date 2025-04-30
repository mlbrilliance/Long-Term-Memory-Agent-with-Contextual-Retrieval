"""
Simple script to run tests and save output to a file.
"""

import subprocess

# Set up the command to run the tests
test_command = [
    "python",
    "-m",
    "pytest",
    "tests\\ltm_agent\\memory\\test_sqlite_store.py",
    "-v",
    "--no-header",
]

# Run the command and capture output
print(f"Running test command: {' '.join(test_command)}")
result = subprocess.run(test_command, capture_output=True, text=True)

# Write output to file
with open("test_output.txt", "w") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\n\nSTDERR:\n")
    f.write(result.stderr)

print("Test results written to test_output.txt")
print(f"Exit code: {result.returncode}")

# Print a preview of the output
print("\nPreview of STDOUT:")
print(result.stdout[:500])
print("..." if len(result.stdout) > 500 else "")

print("\nPreview of STDERR:")
print(result.stderr[:500])
print("..." if len(result.stderr) > 500 else "")
