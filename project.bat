@echo off
REM This batch permit to create all dsp project or vcproj project for
REM Visual C++ 2003 or 2005
REM use : project [VC7 / VC8 / vc9 / clean]
REM default visual project depends on the environment variable QMAKESPEC 

@if not "%QMAKEPATH%" == ""  goto params
set QMAKEPATH=%CD%\tools\qt4win
set QTDIR=%CD%\tools\qt4win
set PATH=%QMAKEPATH%\bin;%PATH%

:params
if "%1" == "VC7" goto vc7
if "%1" == "VC8" goto vc8
if "%1" == "VC9" goto vc9
if "%1" == "VC10" goto vc10
if "%1" == "VC9_X64" goto vc9_x64
if "%1" == "clean" goto clean

:console
@echo on
@echo Making Makefiles
qmake -recursive
@goto end

:vc7
set QMAKESPEC=win32-msvc.net
@echo on
@echo Making Visual project 7
qmake -tp vc -recursive -o Sofa QT_INSTALL_PREFIX="%QTDIR%"
goto common

:vc8
set QMAKESPEC=win32-msvc2005
@echo on
@echo Making Visual project 8
qmake -tp vc -recursive -o Sofa.sln Sofa.pro QT_INSTALL_PREFIX="%QTDIR%"
goto common

:vc9
set QMAKESPEC=win32-msvc2008
@echo on
@echo Making Visual project 9
qmake -tp vc -recursive -o Sofa.sln Sofa.pro QT_INSTALL_PREFIX="%QTDIR%"
goto common

:vc10
set QMAKESPEC=win32-msvc2010
@echo on
@echo Making Visual project 10
qmake -tp vc -recursive -o Sofa.sln Sofa.pro QT_INSTALL_PREFIX="%QTDIR%"
goto common

:vc9_x64
set QMAKESPEC=win64-msvc2008
@echo on
@echo Making Visual project 9 x64
qmake -tp vc -recursive -o Sofa.sln Sofa.pro QT_INSTALL_PREFIX="%QTDIR%"
@echo off
call x64convert.bat
@echo on
goto common

:common
@if "%ERRORLEVEL%"=="0" goto end
echo ERROR %ERRORLEVEL%
pause
@goto end

:clean
@echo cleaning all VC Projects
for /R %%i in (*.ncb, *.suo, Makefile, *.idb, *.pdb, *.plg, *.opt) do del "%%i"
cd src
for /R %%i in (*.dsp, *.vcproj, *.vcproj.*) do del "%%i"
cd ..

:end
