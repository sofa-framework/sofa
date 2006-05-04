@echo off
REM This batch permit to create all dsp project or vcproj project for
REM Visual C++ 6 or Visual dot Net 2003
REM use : project [VC6 / VC7 / clean]
REM default visual project depends on the environment variable QMAKESPEC
set QTDIR=%CD%\tools
set PATH=%QTDIR%;%PATH%
if "%1" == "VC6" goto vc6
if "%1" == "VC7" goto vc7
if "%1" == "VC8" goto vc8
if "%1" == "clean" goto clean
:vc6
set QMAKESPEC=win32-msvc
rem copy /A sofaVC6.cfg+sofaCommon.cfg sofa.cfg
type sofaVC6.cfg > sofa.cfg
type sofaCommon.cfg >> sofa.cfg
goto common
:vc7
set QMAKESPEC=win32-msvc.net
type sofaVC7.cfg > sofa.cfg
type sofaCommon.cfg >> sofa.cfg
goto common
:vc8
set QMAKESPEC=win32-msvc2005
type sofaVC8.cfg > sofa.cfg
type sofaCommon.cfg >> sofa.cfg
:common
@echo on
@if "%QMAKESPEC%"=="win32-msvc" @echo Making Visual project 6
@if "%QMAKESPEC%"=="win32-msvc.net" @echo Making Visual project 7
@if "%QMAKESPEC%"=="win32-msvc2005" @echo Making Visual project 8
for /R %%i in (*.pro) do qmake "%%i"
goto end
:clean
@echo cleaning all VC6 Project or VC7 Project
for /R %%i in (*.ncb, *.suo, Makefile, *.idb, *.pdb, *.plg, *.opt) do del "%%i"
cd src
for /R %%i in (*.dsp, *.vcproj, *.vcproj.old) do del "%%i"
cd ..\Projects
for /R %%i in (*.dsp, *.vcproj, *.vcproj.old) do del "%%i"
cd ..
:end
