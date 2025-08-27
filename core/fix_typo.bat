@echo off
powershell -NoProfile -Command ^
  "(Get-Content 'd:\RevoicePortable\core\pipeline.py' -Raw) -replace '-> List\[Tuple\[float,float,str\]\]\]:' , '-> List[Tuple[float,float,str]]:' | Set-Content -Encoding UTF8 'd:\RevoicePortable\core\pipeline.py'"
echo Fixed pipeline.py
pause
