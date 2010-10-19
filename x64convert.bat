@echo on
REM change taget platform from Win32 to x64 in Sofa.sln and related vcproj files.
REM sed
REM if errorlevel 0 goto main
REM @echo on
REM echo "sed is required to run that script"
REM echo "http://gnuwin32.sourceforge.net/packages/sed.htm"
REM goto end
goto main
:main
@echo off
set BASEDIR=%CD%
set SOFASLN=%BASEDIR%\Sofa.sln
goto findandreplaceSLN

:findandreplaceSLN
@echo %SOFASLN%
set TMPSLN=%SOFASLN%.tmp 
Copy /y %SOFASLN% %TMPSLN%
sed  -e 's/Win32/x64/g' <%TMPSLN%>%SOFASLN%
del %TMPSLN%
goto findandreplaceVCPROJ

:findandreplaceVCPROJ
for /F "skip=1 tokens=3 delims==, eol=#" %%X in (%SOFASLN%)  do set FILE=%%X& call :replaceVCPROJ 
goto end

:replaceVCPROJ
@echo %FILE%
set TMP=%FILE%.tmp
Copy /y %FILE% %TMP%
sed  -e 's/Name="Win32"/Name="x64"/g' -e 's/"Debug|Win32"/"Debug|x64"/g' -e 's/"Release|Win32"/"Release|x64"/g' <%TMP%>%FILE%
del %TMP% 


:end
cd %CD%

