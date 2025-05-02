# Collect Code Script
# This script collects all relevant production code from the project and compiles it into a single text file.

# Configuration
$outputFile = "project_code_collection.txt"
$projectRoot = $PSScriptRoot

# Define patterns for files to include
$includePatterns = @(
    "*.py"
)

# Define directories and patterns to exclude
$excludeDirectories = @(
    "tests",
    "examples",
    ".venv",
    ".uv",
    "build",
    "dist",
    "benchmark",
    "claude-task-master",
    "__pycache__"
)

$excludePatterns = @(
    "*_test.py",
    "test_*.py",
    "*_example.py",
    "example_*.py",
    "*_demo.py",
    "demo_*.py",
    "setup.py"
)

# Create or clear the output file
"# Project Code Collection" | Out-File -FilePath $outputFile
"# Generated on $(Get-Date)" | Out-File -FilePath $outputFile -Append
"# =======================================================" | Out-File -FilePath $outputFile -Append
"" | Out-File -FilePath $outputFile -Append

# Function to check if a file should be excluded
function ShouldExclude($filePath) {
    $relativePath = $filePath.Replace("$projectRoot\", "")

    # Check if file is in excluded directory
    foreach ($dir in $excludeDirectories) {
        if ($relativePath -match "^$dir\\") {
            return $true
        }
    }

    # Check if file matches excluded pattern
    $fileName = Split-Path $filePath -Leaf
    foreach ($pattern in $excludePatterns) {
        if ($fileName -like $pattern) {
            return $true
        }
    }

    return $false
}

# Find all relevant files
$files = @()
foreach ($pattern in $includePatterns) {
    $files += Get-ChildItem -Path $projectRoot -Filter $pattern -Recurse | Where-Object {
        -not (ShouldExclude($_.FullName))
    }
}

# Process each file
$fileCount = 0
foreach ($file in $files) {
    $relativePath = $file.FullName.Replace("$projectRoot\", "")

    # Add file separator and information
    "" | Out-File -FilePath $outputFile -Append
    "# =======================================================" | Out-File -FilePath $outputFile -Append
    "# FILE: $relativePath" | Out-File -FilePath $outputFile -Append
    "# =======================================================" | Out-File -FilePath $outputFile -Append
    "" | Out-File -FilePath $outputFile -Append

    # Add file content
    Get-Content $file.FullName | Out-File -FilePath $outputFile -Append

    $fileCount++
}

# Add summary at the end
"" | Out-File -FilePath $outputFile -Append
"# =======================================================" | Out-File -FilePath $outputFile -Append
"# Summary: Collected $fileCount files" | Out-File -FilePath $outputFile -Append
"# =======================================================" | Out-File -FilePath $outputFile -Append

Write-Host "Code collection complete! Collected $fileCount files into $outputFile"
