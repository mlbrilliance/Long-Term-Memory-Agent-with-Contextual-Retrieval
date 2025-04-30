@echo off
echo Cleaning up Git repository...

REM Unstage files that should be ignored
git rm --cached error_test_result.txt
git rm --cached example_vector_store.db
git rm --cached -r logs
git rm --cached -r claude-task-master

REM Remove all tracked files from Git's index (without deleting them)
git rm -r --cached .

REM Re-add all files according to the new .gitignore
git add .

echo Git cleanup complete!
echo.
echo Run 'git status' to confirm the changes
pause
