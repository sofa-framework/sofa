@echo off
REM change taget platform from Win32 to x64 in Sofa.sln and related vcproj files.
@sed
@if errorlevel 0 goto main
@echo on
echo "sed is required to run that script"
echo "http://gnuwin32.sourceforge.net/packages/sed.htm"
goto end

:main
@echo off
@set BASEDIR=%CD%
@set SOFASLN=%BASEDIR%\Sofa.sln
@call :findandreplaceSLN
@for /F "tokens=3 delims==," %%X in (%SOFASLN%)  do set FILE=%%X& call :findandreplaceVCPROJ 
@goto end

:findandreplaceSLN
REM echo %SOFASLN%
@set TMP=%SOFASLN%.tmp 
@Copy /y %SOFASLN% %TMP%
@sed  -e 's/Win32/x64/g' <%TMP%>%SOFASLN%
@del %TMP%

:findandreplaceVCPROJ
REM set FILE=%FILE: =%
REM set FILE=%FILE:~1,-1%
REM set FILE=%BASEDIR%\%FILE%
REM echo %FILE%
@set TMP=%FILE%.tmp 
@Copy /y %FILE% %TMP%
@sed  -e 's/Name="Win32"/Name="x64"/g' -e 's/"Debug|Win32"/"Debug|x64"/g' -e 's/"Release|Win32"/"Release|x64"/g' <%TMP%>%FILE%
@del %TMP%

:end
@cd %CD%
