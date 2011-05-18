@echo off
REM This batch permit to create all dsp project or vcproj project for
REM Visual C++ 2003 or 2005
REM use : project [VC7 / VC8 / vc9 / clean] [id# project]
REM default visual project depends on the environment variable QMAKESPEC 

@if not "%SOFA_DIR%" == "" goto qmakepath
set SOFA_DIR=%CD%

:qmakepath
@if not "%QMAKEPATH%" == ""  goto subproject
set QMAKEPATH=%CD%\tools\qt4win
set QTDIR=%CD%\tools\qt4win
set PATH=%QMAKEPATH%\bin;%PATH%

:subproject
set mode= "%2"
if "%mode" == "" goto menu

:menu
echo *------------------------------------*
echo *  Welcome to the Sofa qmake parser  *
echo *------------------------------------*
echo What do you want to rebuild ?
echo 0 : rebuild all
echo 1 : rebuild projects
echo 2 : rebuild plugins
echo 3 : rebuild tutorials
echo 4 : rebuild all components
echo 5 :   - rebuild behaviormodel
echo 6 :   - rebuild collision
echo 7 :   - rebuild configurationsetting
echo 8 :   - rebuild constraint
echo 9 :   - rebuild contextobject
echo 10 :   - rebuild controller
echo 11:   - rebuild engine
echo 12 :  - rebuild fem
echo 13 :  - rebuild forcefield
echo 14 :  - rebuild interactionforcefield
echo 15 :  - rebuild linearsolver
echo 16 :  - rebuild loader
echo 17 :  - rebuild mapping
echo 18 :  - rebuild mass
echo 19 :  - rebuild mastersolver
echo 20 :  - rebuild misc
echo 21 :  - rebuild odesolver
echo 22 :  - rebuild topology
echo 23 :  - rebuild typedef
echo 24 :  - rebuild visualmodel
echo 25 : rebuild gui
echo 26 : rebuild simulation
echo 27 : rebuild cuda module
echo 28 : rebuild core
echo 29 : rebuild defaulttype
echo 30 : rebuild helper
set /p choix=your choice : 
if %choix%==0	set mode=Sofa&		set proj=Sofa.sln	 
if %choix%==1	set mode=projects&	set proj=Sofa.sln&	cd applications\projects
if %choix%==2	set mode=plugins&	set proj=Sofa.sln&	cd applications\plugins
if %choix%==3 	set mode=tutorials&	set proj=Sofa.sln&	cd applications\plugins
if %choix%==4 	set mode=component&	set proj=Sofa.sln&	cd modules\sofa\component
if %choix%==5 	set mode=behaviormodel& 			set proj=sofacomponentbehaviormodel.vcproj&	 cd modules\sofa\component\behaviormodel
if %choix%==6 	set mode=collision& 				set proj=sofacomponentcollision.vcproj&	 cd modules\sofa\component\collision
if %choix%==7 	set mode=configurationsetting& 			set proj=sofacomponentconfigurationsetting.vcproj&	 cd modules\sofa\component\configurationsetting
if %choix%==8 	set mode=constraint& 				set proj=sofacomponentconstraint.vcproj&	 cd modules\sofa\component\constraint
if %choix%==9 	set mode=contextobject& 			set proj=sofacomponentcontextobject.vcproj&	 cd modules\sofa\component\contextobject
if %choix%==10 	set mode=controller& 				set proj=sofacomponentcontroller.vcproj&	 cd modules\sofa\component\controller
if %choix%==11	set mode=engine& 					set proj=sofacomponentengine.vcproj&	 cd modules\sofa\component\engine
if %choix%==12	set mode=fem& 						set proj=sofacomponentfem.vcproj&	cd modules\sofa\component\fem
if %choix%==13	set mode=forcefield& 				set proj=sofacomponentforcefield.vcproj&	 cd modules\sofa\component\forcefield
if %choix%==14 	set mode=interactionforcefield& 	set proj=sofacomponentinteractionforcefield.vcproj&	 cd modules\sofa\component\interactionforcefield
if %choix%==15 	set mode=linearsolver& 				set proj=sofacomponentlinearsolver.vcproj&	 cd modules\sofa\component\linearsolver
if %choix%==16 	set mode=loader& 					set proj=sofacomponentloader.vcproj&	 cd modules\sofa\component\loader
if %choix%==17 	set mode=mapping& 					set proj=sofacomponentmapping.vcproj&	 cd modules\sofa\component\mapping
if %choix%==18 	set mode=mass& 						set proj=sofacomponentmass.vcproj&	 cd modules\sofa\component\mass
if %choix%==19 	set mode=mastersolver& 				set proj=sofacomponentmastersolver.vcproj&	 cd modules\sofa\component\mastersolver
if %choix%==20 	set mode=misc& 						set proj=sofacomponentmisc.vcproj&	 cd modules\sofa\component\misc
if %choix%==21 	set mode=odesolver& 				set proj=sofacomponentodesolver.vcproj&	 cd modules\sofa\component\odesolver
if %choix%==22 	set mode=topology& 					set proj=sofacomponenttopology.vcproj&	 cd modules\sofa\component\topology
if %choix%==23 	set mode=typedef& 					set proj=sofacomponenttypedef.vcproj&	 cd modules\sofa\component\typedef
if %choix%==24 	set mode=visualmodel& 				set proj=sofacomponentvisualmodel.vcproj&	 cd modules\sofa\component\visualmodel
if %choix%==25 	set mode=gui&			set proj=Sofa.sln&	cd applications\sofa\gui
if %choix%==26 	set mode=simulation&	set proj=Sofa.sln&	cd modules\sofa\simulation
if %choix%==27 	set mode=gpu&			set proj=Sofa.sln&	cd modules\sofa\gpu
if %choix%==28 	set mode=core&			set proj=sofacore.vcproj&	cd framework\sofa\core
if %choix%==29 	set mode=defaulttype&	set proj=sofadefaulttype.vcproj&	cd framework\sofa\defaulttype
if %choix%==30 	set mode=helper&		set proj=sofahelper.vcproj&	cd framework\sofa\helper
echo you chose %mode%
goto params


:params
if "%1" == "VC7" goto vc7
if "%1" == "VC8" goto vc8
if "%1" == "VC9" goto vc9
if "%1" == "clean" goto clean
if "%1" == "" goto menu_versions

:menu_versions
echo ------------
echo For wich Visual Studio ?
echo 0 : visual 7 (.net)
echo 1 : visual 8 (2005)
echo 2 : visual 9 (2008)
set /p choix=your choice : 
if %choix%==0 goto vc7
if %choix%==1 goto vc8
if %choix%==2 goto vc9


:console
@echo on
@echo Making Makefiles
qmake -recursive
@goto end

:vc7
set QMAKESPEC=win32-msvc.net
@echo on
@echo Making Visual project 7
qmake -tp vc -recursive -o %proj% %mode%.pro QT_INSTALL_PREFIX="%QTDIR%"
goto common

:vc8
set QMAKESPEC=win32-msvc2005
@echo on
@echo Making Visual project 8
qmake -tp vc -recursive -o %proj% %mode%.pro QT_INSTALL_PREFIX="%QTDIR%"
goto common

:vc9
set QMAKESPEC=win32-msvc2008
@echo on
@echo Making Visual project 9
qmake -tp vc -recursive -o %proj% %mode%.pro QT_INSTALL_PREFIX="%QTDIR%"
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
@cd %SOFA_DIR%
