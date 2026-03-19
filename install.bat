@echo off
echo Installing Bringer...

py -m pip install --upgrade pip
py -m pip install -e .

echo Installation complete!
echo You can now run: Bringer
pause
