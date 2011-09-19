@echo off
set PATH=%CD%/lib;%PATH%

@if not "%QMAKEPATH%" == ""  goto params
set QMAKEPATH=%CD%\tools\qt4win
set QTDIR=%CD%\tools\qt4win
set PATH=%QMAKEPATH%\bin;%PATH%

:params
if "%1" == "VC7" goto vc7
if "%1" == "VC8" goto vc8
if "%1" == "VC9" goto vc9

:vc7
set QMAKESPEC=win32-msvc.net
goto common

:vc8
set QMAKESPEC=win32-msvc2005
goto common

:vc9
set QMAKESPEC=win32-msvc2008
goto common

:common
@if "%ERRORLEVEL%"=="0" goto end
echo ERROR %ERRORLEVEL%
pause
@goto end

:end
