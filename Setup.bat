@echo off
echo preparing requirements...
py -m pip install --upgrade pip
python.exe -m pip install --upgrade pip
python -m pip install -r Requirements.txt
pause
