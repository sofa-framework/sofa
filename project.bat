@echo off
REM This batch permit to create all dsp project or vcproj project for
REM Visual C++ 2005, 2008, 2010
REM use : project [VC8 / VC9 / VC10 / clean]
REM default visual project depends on the environment variable QMAKESPEC 

set TARGET_MACHINE=x86
@if not "%QMAKEPATH%" == ""  goto params
if exist %CD%\tools\qt4win\bin\qmake.exe goto qtlocal
if "%QTDIR%" == "" goto noqt

set PATH=%QTDIR%\bin;%PATH%
qmake --version
goto params

:noqt
echo No Qt found. Either install a pre-compiled Qt version and set QTDIR variable, or uncompress a Sofa dependency package providing Qt
@goto end

:qtlocal
set QMAKEPATH=%CD%\tools\qt4win
set QTDIR=%CD%\tools\qt4win
set PATH=%QMAKEPATH%\bin;%PATH%

:params
if "%1" == "VC8" goto vc8
if "%1" == "VC9" goto vc9
if "%1" == "VC10" goto vc10
if "%1" == "VC10_BS" goto vc10_bs
if "%1" == "VC10_BS_CMAKE" goto vc10_bs_cmake
if "%1" == "VC9_X64" goto vc9_x64
if "%1" == "VC10_X64" goto vc10_x64
if "%1" == "clean" goto clean
if "%QMAKESPEC%" == "win32-msvc2005" goto vc8
if "%QMAKESPEC%" == "win32-msvc2008" goto vc9
if "%QMAKESPEC%" == "win32-msvc2010" goto vc10

:console
@echo on
@echo Making Makefiles
qmake -recursive
@goto end

:vc8
set QMAKESPEC=win32-msvc2005
@echo on
@echo Making Visual project 8
qmake -tp vc -recursive -o Sofa.sln Sofa.pro QT_INSTALL_PREFIX="%QTDIR:\=/%"
echo Copying external dlls.
xcopy .\bin\dll_x86\*.* .\bin\ /y /q
goto common

:vc9
set QMAKESPEC=win32-msvc2008
@echo on
@echo Making Visual project 9
call "%VS90COMNTOOLS%..\..\VC\vcvarsall.bat" x86
qmake -tp vc -recursive -o Sofa.sln Sofa.pro QT_INSTALL_PREFIX="%QTDIR:\=/%"
echo Copying external dlls.
xcopy .\bin\dll_x86\*.* .\bin\ /y /q
goto common

:vc10
set QMAKESPEC=win32-msvc2010
@echo on
@echo Making Visual project 10
call "%VS100COMNTOOLS%..\..\VC\vcvarsall.bat" x86
qmake -tp vc -recursive -o Sofa.sln Sofa.pro QT_INSTALL_PREFIX="%QTDIR:\=/%"
echo Copying external dlls.
xcopy .\bin\dll_x86\*.* .\bin\ /y /q
goto common

:vc10_bs
set QMAKESPEC=win32-msvc2010
@echo on
@echo Making Visual project 10
call "%VS100COMNTOOLS%..\..\VC\vcvarsall.bat" x86
echo Copying external dlls.
xcopy .\bin\dll_x86\*.* .\bin\ /y /q
qmake -recursive Sofa.pro QT_INSTALL_PREFIX="%QTDIR:\=/%"
nmake
goto common

:vc10_bs_cmake
REM set QMAKESPEC=win32-msvc2010
@echo on
@echo Making Visual project 10 (cmake)
call "%VS100COMNTOOLS%..\..\VC\vcvarsall.bat" x86
echo Copying external dlls.
xcopy ..\bin\dll_x86\*.* ..\bin\ /y /q
call cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
call cmake ..
nmake
goto common

:vc11_x64_bs_cmake
set TARGET_MACHINE=x64
@echo on
@echo Making Visual project 11 (cmake)
call "%VS110COMNTOOLS%..\..\VC\vcvarsall.bat" amd64
echo Copying external dlls.
xcopy ..\bin\dll_x64\*.* ..\bin\ /y /q
call cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
call cmake ..
call nmake
goto common

:vc9_x64
set QMAKESPEC=win32-msvc2008
set TARGET_MACHINE=x64
@echo on
@echo Making Visual project 9 x64
call "%VS90COMNTOOLS%..\..\VC\vcvarsall.bat" amd64
qmake -tp vc -recursive -o Sofa.sln Sofa.pro QT_INSTALL_PREFIX="%QTDIR:\=/%"
echo Copying external dlls.
xcopy .\bin\dll_x64\*.* .\bin\ /y /q
goto common

:vc10_x64
set QMAKESPEC=win32-msvc2010
set TARGET_MACHINE=x64
@echo on
@echo Making Visual project 10 x64
call "%VS100COMNTOOLS%..\..\VC\vcvarsall.bat" amd64
qmake -tp vc -recursive -o Sofa.sln Sofa.pro QT_INSTALL_PREFIX="%QTDIR:\=/%"
echo Copying external dlls.
xcopy .\bin\dll_x64\*.* .\bin\ /y /q
goto common

:common
echo Copying qt dlls.
xcopy %QTDIR:/=\%\bin\*.dll .\bin\ /y /q
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
