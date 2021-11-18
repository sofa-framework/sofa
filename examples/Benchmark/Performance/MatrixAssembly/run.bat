@echo off

:: This script executes all the simulation files found in this directory
:: It extracts and display simulation times, so they can be compared.

:: location of the runSofa executable
set SOFA=bin/runSofa.exe

for %%f in (*.scn) do (
    echo %%f
    
    set sofa_cmd=%SOFA% -g batch -n 1000 --computationTimeSampling 1000 %%f
    %sofa_cmd% > %%f.perf 2>&1
    
    findstr /c:"iterations done in" %%f.perf
    findstr /c:"LEVEL" %%f.perf
    findstr /c:"\.\.AnimateVisitor" %%f.perf
)