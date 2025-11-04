@echo off

rem Activate the virtual environment
call .venv\Scripts\activate

rem Run the Python script
python app.py

rem Deactivate the virtual environment
deactivate
pause