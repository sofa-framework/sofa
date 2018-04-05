# SOFA Changelog



## On master branch (not released yet)

[Full log](https://github.com/sofa-framework/sofa/compare/v17.12...HEAD)


### Deprecated

### Breaking

### Improvements

**Kernel modules**
- [SofaLoader]
    - ADD support to load VTK polylines in legacy formated files [#576](https://github.com/sofa-framework/sofa/pull/576)

**Applications**
- [SofaPython]
    - PythonScriptDataEngine (PSDE) [#583](https://github.com/sofa-framework/sofa/pull/583)

**Tools**
- [tools]
    - FIX sofa-launcher stdout [#592](https://github.com/sofa-framework/sofa/pull/592)

### Bug Fixes

**Kernel modules**
- [all]
    - FIX warnings [#584](https://github.com/sofa-framework/sofa/pull/584)
- [SofaHelper]
    - More robust method to test end of string [#617](https://github.com/sofa-framework/sofa/pull/617)
- [SofaSimulationGraph]
    - FIX dependencies [#588](https://github.com/sofa-framework/sofa/pull/588)

**Other modules**
- [SofaMiscFem]
    - FIX dependencies [#588](https://github.com/sofa-framework/sofa/pull/588)

**Applications**
- [CImgPlugin]
    - Export CImg_CFLAGS [#595](https://github.com/sofa-framework/sofa/pull/595)
- [runSofa]
    - Fix compilation when SofaGuiQt is not activated [#599](https://github.com/sofa-framework/sofa/pull/599)
- [SofaDistanceGrid]
    - ADD .scene-tests to ignore scene [#594](https://github.com/sofa-framework/sofa/pull/594)
- [SofaPython]
    - FIX build for MacOS >10.13.0 [#614](https://github.com/sofa-framework/sofa/pull/614)

**Scenes**
- FIX collision of the fontain example [#612](https://github.com/sofa-framework/sofa/pull/612)

**Extlibs**
- [extlibs/gtest] Update gtest  & clean the CMakeLists.txt [#604](https://github.com/sofa-framework/sofa/pull/604)


### Cleaning

**Kernel modules**
- [All]
    - CMake: Remove COMPONENTSET, keep DEPRECATED [#586](https://github.com/sofa-framework/sofa/pull/586)
- [SofaHelper]
    - CLEAN commented code and double parentheses in Messaging.h [#587](https://github.com/sofa-framework/sofa/pull/587)

**Applications**
- [CImgPlugin]
    - Less scary config warnings [#607](https://github.com/sofa-framework/sofa/pull/607)
- [HeadlessRecorder]
    - Handle errors in target config [#608](https://github.com/sofa-framework/sofa/pull/608)
- [SofaGUI]
    - Move GlutGUI to projects and remove all glut references in SofaFramework [#598](https://github.com/sofa-framework/sofa/pull/598)
    - CMake: Remove useless if block in qt CMakelists.txt [#590](https://github.com/sofa-framework/sofa/pull/590)


____________________________________________________________



## [v17.12](https://github.com/sofa-framework/sofa/tree/v17.12)

[Full log](https://github.com/sofa-framework/sofa/compare/v17.06...v17.12)


### Deprecated

**Kernel modules**
- Will be removed in v17.12
    - [all]
        - SMP support [#457](https://github.com/sofa-framework/sofa/pull/457) - no more maintained
    - [SofaDefaultType]
        - LaparoscopicRigidType [#457](https://github.com/sofa-framework/sofa/pull/457) - not used/dont compiled for a really long time

- Will be removed in v18.06
    - [SofaHelper]
        - Utils::getPluginDirectory() [#518](https://github.com/sofa-framework/sofa/pull/518) - Use PluginRepository.getFirstPath() instead

**Other modules**
- Will be removed in v18.12
    - [SofaBoundaryCondition]
        - BuoyantForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
        - VaccumSphereForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
    - [SofaMisc]
        - ParallelCGLinearSolver [#457](https://github.com/sofa-framework/sofa/pull/457)
    - [SofaMiscForceField]
        - ForceMaskOff [#457](https://github.com/sofa-framework/sofa/pull/457)
        - LineBendingSprings [#457](https://github.com/sofa-framework/sofa/pull/457)
        - WashingMachineForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
		- LennardJonesForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
    - [SofaMiscMapping]
        - CatmullRomSplineMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
        - CenterPointMechanicalMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
        - CurveMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
        - ExternalInterpolationMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
        - ProjectionToLineMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
        - ProjectionToPlaneMapping
    - [SofaOpenglVisual]
        - OglCylinderModel [#457](https://github.com/sofa-framework/sofa/pull/457)
        - OglGrid [#457](https://github.com/sofa-framework/sofa/pull/457)
        - OglRenderingSRGB [#457](https://github.com/sofa-framework/sofa/pull/457)
        - OglLineAxis [#457](https://github.com/sofa-framework/sofa/pull/457)
        - OglSceneFrame
    - [SofaUserInteraction]
        - AddRecordedCameraPerformer [#457](https://github.com/sofa-framework/sofa/pull/457)
        - ArticulatedHierarchyBVHController [#457](https://github.com/sofa-framework/sofa/pull/457)
        - ArticulatedHierarchyController [#457](https://github.com/sofa-framework/sofa/pull/457)
        - DisabledContact [#457](https://github.com/sofa-framework/sofa/pull/457)
        - EdgeSetController [#457](https://github.com/sofa-framework/sofa/pull/457)
        - FixParticlePerformer [#457](https://github.com/sofa-framework/sofa/pull/457)
        - GraspingManager [#457](https://github.com/sofa-framework/sofa/pull/457)
        - InciseAlongPathPerformer [#457](https://github.com/sofa-framework/sofa/pull/457)
        - InterpolationController [#457](https://github.com/sofa-framework/sofa/pull/457)
        - MechanicalStateControllerOmni [#457](https://github.com/sofa-framework/sofa/pull/457)
        - NodeToggleController [#457](https://github.com/sofa-framework/sofa/pull/457)
        - RemovePrimitivePerformer [#457](https://github.com/sofa-framework/sofa/pull/457)
        - StartNavigationPerformer [#457](https://github.com/sofa-framework/sofa/pull/457)
        - SuturePointPerformer [#457](https://github.com/sofa-framework/sofa/pull/457)


### Breaking

**Kernel modules**
- [all]
    - issofa_visitors: Changing the way projective constraints are propagated in visitors [#216](https://github.com/sofa-framework/sofa/pull/216)
- [SofaDeformable]
    - Change how rest shape is given in RestShapeSpringsForceField [#315](https://github.com/sofa-framework/sofa/pull/315)

**Other modules**
- [SofaHelper]
    - Rewrite ArgumentParser [#513](https://github.com/sofa-framework/sofa/pull/513)
- [SofaValidation]
    - ADD Monitor test [#312](https://github.com/sofa-framework/sofa/pull/312)


### Improvements

**Kernel modules**
- [all]
    - issofa_topology: Improvement, BugFix and Cleaning on Topology [#243](https://github.com/sofa-framework/sofa/pull/243)
    - issofa_constraintsolving: improve constraints [#484](https://github.com/sofa-framework/sofa/pull/484)
    - Improve File:line info in error message (for python and xml error reporting) [#314](https://github.com/sofa-framework/sofa/pull/314)
- [SofaDeformable]
    - issofa_meshspringff [#497](https://github.com/sofa-framework/sofa/pull/497)
    - issofa_springff [#498](https://github.com/sofa-framework/sofa/pull/498)
- [SofaHelper]
    - add ability to use NoArgument  for BaseCreator and Factory [#385](https://github.com/sofa-framework/sofa/pull/385)
    - Remove legacy ImageBMP/ImagePNG and ImageQt [#424](https://github.com/sofa-framework/sofa/pull/424)
    - Improve advanced timer [#468](https://github.com/sofa-framework/sofa/pull/468)
- [SofaLoader]
    - ADD normals and vectors to Legacy vtk import [#536](https://github.com/sofa-framework/sofa/pull/536)
- [SofaSimpleFem]
    - Add check of vector size in TetrahedronFEMForceField [#341](https://github.com/sofa-framework/sofa/pull/341)

**Other modules**
- [all]
    - Fix default value rayleigh params [#350](https://github.com/sofa-framework/sofa/pull/350)
    - PSL branch prerequisites [#410](https://github.com/sofa-framework/sofa/pull/410)
    - template alias sptr for downsizing the include graph [#436](https://github.com/sofa-framework/sofa/pull/436)
    - Removing the typedef files + SOFA_DECL + SOFA_LINK [#453](https://github.com/sofa-framework/sofa/pull/453)
    - CHANGE USE_MASK option to off by default [#532](https://github.com/sofa-framework/sofa/pull/532)
- [SofaBoundaryCondition]
    - ADD flag PROJECTVELOCITY [#288](https://github.com/sofa-framework/sofa/pull/288)
    - make LinearMovementConstraint accept absolute movements [#394](https://github.com/sofa-framework/sofa/pull/394)
- [SofaCore]
    - Add some read/write free method to construct Data Read/WriteAccessor [#450](https://github.com/sofa-framework/sofa/pull/450)
- [SofaDefaulttype]
    - MapMapSparseMatrix conversion utils with Eigen Sparse [#452](https://github.com/sofa-framework/sofa/pull/452)
    - multTranspose method between MapMapSparseMatrix and BaseVector [#456](https://github.com/sofa-framework/sofa/pull/456)
- [SofaDeformable]
    - Rest shape can now be given using SingleLink [#315](https://github.com/sofa-framework/sofa/pull/315)
    - Add AngularSpringForceField [#334](https://github.com/sofa-framework/sofa/pull/334)
- [SofaEulerianFluid]
    - Pluginizing the module [#396](https://github.com/sofa-framework/sofa/pull/396)
- [SofaExporter]
    - Clean & test the exporter  [#372](https://github.com/sofa-framework/sofa/pull/372)
- [SofaGraphComponent]
    - Add SceneCheckerVisitor to detect missing RequiredPlugin [#306](https://github.com/sofa-framework/sofa/pull/306)
    - Add a mechanism (SceneChecker) to report API & SceneChange to users [#329](https://github.com/sofa-framework/sofa/pull/329)
    - Refactor the SceneChecker and add a new SceneChecker to test dumplicated names [#392](https://github.com/sofa-framework/sofa/pull/392)
- [SofaGeneralEngine]
    - Add test and minor cleaning for IndexValueMapper [#319](https://github.com/sofa-framework/sofa/pull/319)
    - Add a computeBBox to the SmoothMeshEngine [#409](https://github.com/sofa-framework/sofa/pull/409)
- [SofaGeneralObjectInteraction]
    - issofa_attachconstraint [#501](https://github.com/sofa-framework/sofa/pull/501)
- [SofaMisc]
    - Add a data repository at start-up time [#402](https://github.com/sofa-framework/sofa/pull/402)
- [SofaMiscCollision]
    - Pluginizing the module [#453](https://github.com/sofa-framework/sofa/pull/453)
- [SofaMiscFem]
    - Add hyperelasticity fem code in SOFA [#349](https://github.com/sofa-framework/sofa/pull/349)
- [SofaSimpleFem]
    - ADD computeBBox and test to HexaFEMForceField [#289](https://github.com/sofa-framework/sofa/pull/289)
- [SofaSphFluid]
    - Pluginizing the module [#453](https://github.com/sofa-framework/sofa/pull/453)
- [SofaVolumetricData]
    - Split the module in two plugins [#389](https://github.com/sofa-framework/sofa/pull/389)

**Applications**
- [CGALPlugin]
    - Add new functionality for mesh generation from image: definition of features [#294](https://github.com/sofa-framework/sofa/pull/294)
- [meshconv]
    - Improve the CMake config of meshconv requiring miniflowVR to compile [#358](https://github.com/sofa-framework/sofa/pull/358)
- [PSL]
    - Experimental : Add PSL for 17.12 [#541](https://github.com/sofa-framework/sofa/pull/541)
- [runSofa]
    - autoload plugins (2nd version) [#301](https://github.com/sofa-framework/sofa/pull/301)
    - Extend the live coding support, message API available for nodes, add an openInEditor [#337](https://github.com/sofa-framework/sofa/pull/337)
    - add verification if help is not null from displayed data [#382](https://github.com/sofa-framework/sofa/pull/382)
    - improve the html DocBrowser  [#540](https://github.com/sofa-framework/sofa/pull/540)
- [SceneChecker]
    - Add mechanism to report API & SceneChange to users [#329](https://github.com/sofa-framework/sofa/pull/329)
- [SofaDistanceGrid]
    - Pluginizing SofaVolumetricData [#389](https://github.com/sofa-framework/sofa/pull/389)
- [SofaImplicitField]
    - Pluginizing SofaVolumetricData [#389](https://github.com/sofa-framework/sofa/pull/389)
- [SofaMiscCollision]
    - Pluginizing the module [#453](https://github.com/sofa-framework/sofa/pull/453)
- [SofaPython]
    - Add unicode to string convertion and a warning message in Binding_BaseContext::pythonToSofaDataString [#313](https://github.com/sofa-framework/sofa/pull/313)
    - Add unicode to string convertion in Binding_BaseData::SetDataValuePython [#313](https://github.com/sofa-framework/sofa/pull/313)
    - Add a test [#313](https://github.com/sofa-framework/sofa/pull/313)
    - Add GIL management [#326](https://github.com/sofa-framework/sofa/pull/326)
    - Add support for Sofa.msg_ with emitter other than a string [#335](https://github.com/sofa-framework/sofa/pull/335)
    - Adding new features for AdvancedTimer [#360](https://github.com/sofa-framework/sofa/pull/360)
    - forward sys.argv to python scripts [#368](https://github.com/sofa-framework/sofa/pull/368)
    - sparse matrix aliasing scipy/eigen [#411](https://github.com/sofa-framework/sofa/pull/411)
- [SofaSphFluid]
    - Pluginizing the module [#453](https://github.com/sofa-framework/sofa/pull/453)

**Other**
- [Tools]
    - Update astyle config [#550](https://github.com/sofa-framework/sofa/pull/550)


### Bug Fixes

**Kernel modules**
- [all]
    - CMake: Fix and clean boost, when using Sofa as an external lib [#421](https://github.com/sofa-framework/sofa/pull/421)
    - Fix computeBBox functions [#527](https://github.com/sofa-framework/sofa/pull/527)
    - CMake: FIX Boost::program_options finding in install [#618](https://github.com/sofa-framework/sofa/pull/618)
- [SofaBaseLinearSolver]
    - FIX no step if condition on denominator is met at first step [#521](https://github.com/sofa-framework/sofa/pull/521)
    - FIX breaking condition in CG at first step regarding threshold [#556](https://github.com/sofa-framework/sofa/pull/556)
- [SofaBaseMechanics]
    - Make sure the mechanical object's state vectors size matches their respective argument size [#406](https://github.com/sofa-framework/sofa/pull/406)
- [SofaBaseTopology]
    - FIX wrong clean in PointSetTopologyModifier [#380](https://github.com/sofa-framework/sofa/pull/380)
- [SofaComponentCommon]
    - Always register all its components in the object factory [#488](https://github.com/sofa-framework/sofa/pull/488)
- [SofaCore]
    - FIX CreateString problem on root node [#377](https://github.com/sofa-framework/sofa/pull/377)
    - FIX don't inline exported functions [#449](https://github.com/sofa-framework/sofa/pull/449)
- [SofaDefaultType]
    - FIX Mat::transpose() and Mat::invert() [#317](https://github.com/sofa-framework/sofa/pull/317)
    - Correct CMake include_directories directive for SofaDefaultType target's [#403](https://github.com/sofa-framework/sofa/pull/403)
    - Fix compilation errors when working with transform class [#506](https://github.com/sofa-framework/sofa/pull/506)
- [SofaHelper]
    - Fix CUDA compilation with pointer of data [#320](https://github.com/sofa-framework/sofa/pull/320)
    - FIX livecoding of shaders [#415](https://github.com/sofa-framework/sofa/pull/415)
    - fixing Polynomial_LD [#442](https://github.com/sofa-framework/sofa/pull/442)
    - Replacing afficheResult with resultToString [#473](https://github.com/sofa-framework/sofa/pull/473)
    - FIX Remove override warnings [#520](https://github.com/sofa-framework/sofa/pull/520)
    - Fix memory leak while capturing screenshot [#533](https://github.com/sofa-framework/sofa/pull/533)
    - FIX Windows relative path from runSofa [#568](https://github.com/sofa-framework/sofa/pull/568)
- [SofaRigid]
    - RigidMapping: fixed setRepartition backward compatibility [#441](https://github.com/sofa-framework/sofa/pull/441)
- [SofaSimulationCommon]
    - Fix a minor regression introduced during the post-sprint [#476](https://github.com/sofa-framework/sofa/pull/476)
- [SofaSimulationCore]
    - Add stop in add_mbktomatrixVisitor [#439](https://github.com/sofa-framework/sofa/pull/439)

**Other modules**
- [all]
    - Fix warnings and strange double incrementation on iterator [#364](https://github.com/sofa-framework/sofa/pull/364)
    - installing gtest headers for separate plugin builds [#395](https://github.com/sofa-framework/sofa/pull/395)
    - Fix override warnings [#423](https://github.com/sofa-framework/sofa/pull/423)
    - FIX Sofa installation failure (tries to install non-existing files) [#470](https://github.com/sofa-framework/sofa/pull/470)
    - ADD _d suffix for debug libs [#511](https://github.com/sofa-framework/sofa/pull/511)
- [SofaBoundaryCondition]
    - Fix LinearForceField that disappears [#525](https://github.com/sofa-framework/sofa/pull/525)
    - FIX Removed incorrect warning from LinearForceField [#384](https://github.com/sofa-framework/sofa/pull/384)
- [SofaConstraint]
    - Fix error due to MacOS >= 10.11 using a relative filename [#325](https://github.com/sofa-framework/sofa/pull/325)
    - Fix issue in GenericConstraintCorrection  [#567](https://github.com/sofa-framework/sofa/pull/567)
- [SofaDeformable]
    - Fix RestShapeSpringsForceField  [#367](https://github.com/sofa-framework/sofa/pull/367)
- [SofaExporter]
    - FIX allow to extend VTKExporter in a plugin [#309](https://github.com/sofa-framework/sofa/pull/309)
- [SofaGeneralEngine]
    - Fix some XyzTransformMatrixEngine::update() function [#343](https://github.com/sofa-framework/sofa/pull/343)
- [SofaGeneralImplicitOdeSolver]
    - fix test [#478](https://github.com/sofa-framework/sofa/pull/478)
- [SofaGraphComponent]
    - Fix the test that was wrong and thus failing in SceneChecker [#405](https://github.com/sofa-framework/sofa/pull/405)
    - Fix a crashing bug in SceneCheckAPIChange. [#479](https://github.com/sofa-framework/sofa/pull/479)
    - FIX SceneChecker & RequiredPlugin [#563](https://github.com/sofa-framework/sofa/pull/563)
- [SofaGeneralObjectInteraction]
    - Remove stiffness multiplicator in SpringForceField [#290](https://github.com/sofa-framework/sofa/pull/290)
- [SofaMiscFem]
    - Fix FastTetrahedralCorotationalFF topology change [#554](https://github.com/sofa-framework/sofa/pull/554)
- [SofaOpenglVisual]
    - Fix a bug crashing sofa when the textures cannot be loaded. [#474](https://github.com/sofa-framework/sofa/pull/474)

**Extlibs**
- [CImg]
    - Refactor CImg & CImgPlugin [#562](https://github.com/sofa-framework/sofa/pull/562)
- [Eigen]
    - Warning fix with eigen when compiling with msvc [#447](https://github.com/sofa-framework/sofa/pull/447)
- [libQGLViewer]
    - FIX missing headers [#461](https://github.com/sofa-framework/sofa/pull/461)
    - Update libQGLViewer to 2.7.1 [#545](https://github.com/sofa-framework/sofa/pull/545)

**Applications**
- [CGALPlugin]
    - Fix build problem [#351](https://github.com/sofa-framework/sofa/pull/351)
    - FIX build error with CGAL > 4.9.1 [#515](https://github.com/sofa-framework/sofa/pull/515)
- [CImgPlugin]
    - Use sofa cmake command to create proper package [#544](https://github.com/sofa-framework/sofa/pull/544)
    - Refactor CImg & CImgPlugin [#562](https://github.com/sofa-framework/sofa/pull/562)
    - prevent INT32 redefinition by libjpeg on Windows [#566](https://github.com/sofa-framework/sofa/pull/566)
- [ColladaSceneLoader]
    - FIX Assimp copy on Windows [#504](https://github.com/sofa-framework/sofa/pull/504)
- [Flexible]
    - Refactor CImg & CImgPlugin [#562](https://github.com/sofa-framework/sofa/pull/562)
- [image]
    - Refactor CImg & CImgPlugin [#562](https://github.com/sofa-framework/sofa/pull/562)
- [Meshconv]
    -  fix build if no miniflowVR [#358](https://github.com/sofa-framework/sofa/pull/358)
- [MultiThreading]
    - FIX: examples installation [#299](https://github.com/sofa-framework/sofa/pull/299)
    - Fix build with Boost 1.64.0 [#359](https://github.com/sofa-framework/sofa/pull/359)
    - FIX Windows dll loading [#507](https://github.com/sofa-framework/sofa/pull/507)
- [runSofa]
    - FIX plugin config copy on Windows [#451](https://github.com/sofa-framework/sofa/pull/451)
    - remove non-ASCII chars in string [#327](https://github.com/sofa-framework/sofa/pull/327)
    - FIX PluginRepository initialization [#502](https://github.com/sofa-framework/sofa/pull/502)
- [SofaCUDA]
    - Fix compilation problem with cuda.  [#320](https://github.com/sofa-framework/sofa/pull/320)
    - Fix: Headers of the SofaCUDA plugin are now installed in the include folder [#383](https://github.com/sofa-framework/sofa/pull/383)
    - FIX Configuration/compilation issue with CUDA plugin [#523](https://github.com/sofa-framework/sofa/pull/523)
    - Fix linearforcefield that disappears [#525](https://github.com/sofa-framework/sofa/pull/525)
- [SofaGui]
    - FIX draw scenes on classical and retina screens [#311](https://github.com/sofa-framework/sofa/pull/311)
    - Fixes #183 : Use the qt menu instead of the native one in Mac OS [#366](https://github.com/sofa-framework/sofa/pull/366)
    - fix ImageBMP issue + remove Laparoscopic stuff [#499](https://github.com/sofa-framework/sofa/pull/499)
    - Pickhandler minor fixs [#522](https://github.com/sofa-framework/sofa/pull/522)
    - Fix: Intel graphics on linux now overrides the core profile context [#526](https://github.com/sofa-framework/sofa/pull/526)
- [SofaImplicitField]
    - Fix package configuration [#422](https://github.com/sofa-framework/sofa/pull/422)
- [SofaPhysicsAPI]
    - FIX: compilation due to Sofa main API changes [#549](https://github.com/sofa-framework/sofa/pull/549)
- [SofaPython]
    - Fix python live coding that is broken [#414](https://github.com/sofa-framework/sofa/pull/414)
    - FIX crash in python script when visualizing advanced timer output [#458](https://github.com/sofa-framework/sofa/pull/458)
    - FIX error in script for plotting advancedTimer output [#519](https://github.com/sofa-framework/sofa/pull/519)
    - Fix python unicode data [#313](https://github.com/sofa-framework/sofa/pull/313)
- [SofaSPHFluid]
    - Fix invalid plugin initialization. [#467](https://github.com/sofa-framework/sofa/pull/467)
- [SofaVolumetricData]
    - Fix package configuration [#422](https://github.com/sofa-framework/sofa/pull/422)
- [SceneCreator]
    - FIX build error with AppleClang 6.0.0.6000051 [#483](https://github.com/sofa-framework/sofa/pull/483)
- [THMPGSpatialHashing]
    - Fix build with Boost 1.64.0 [#359](https://github.com/sofa-framework/sofa/pull/359)

**Scenes**
- Fix scenes [#310](https://github.com/sofa-framework/sofa/pull/310)
- Fix scenes with bad RegularGrid position relative to 270 [#324](https://github.com/sofa-framework/sofa/pull/324)
- Fix scenes errors and crashes [#505](https://github.com/sofa-framework/sofa/pull/505)
- FIX all scenes failures 17.12 [#565](https://github.com/sofa-framework/sofa/pull/565)


### Cleaning

**Kernel modules**
- [all]
    - replace a bunch of std::cerr, std::cout, prinf to use msg_* instead [#339](https://github.com/sofa-framework/sofa/pull/339)
    - More std::cout to msg_* cleaning [#370](https://github.com/sofa-framework/sofa/pull/370)
    - FIX removed compilation warnings [#386](https://github.com/sofa-framework/sofa/pull/386)
    - Clean BaseLoader & Remove un-needed #includes  [#393](https://github.com/sofa-framework/sofa/pull/393)
    - Remove last direct calls of OpenGL in modules [#425](https://github.com/sofa-framework/sofa/pull/425)
    - warning removal [#454](https://github.com/sofa-framework/sofa/pull/454)
    - Refactor SofaTest to cut dependencies [#471](https://github.com/sofa-framework/sofa/pull/471)
    - Modularizing config.h [#475](https://github.com/sofa-framework/sofa/pull/475)
    - EDIT: use PluginRepository instead of Utils::getPluginDirectory [#518](https://github.com/sofa-framework/sofa/pull/518)
- [SofaBaseLinearSolver]
    - Add comments in the CGLinearSolver [#469](https://github.com/sofa-framework/sofa/pull/469)
- [SofaBaseVisual]
    - Clean API message [#503](https://github.com/sofa-framework/sofa/pull/503)
- [SofaDefaultType]
    - remove warning generated by MapMapSparseMatrixEigenUtils [#485](https://github.com/sofa-framework/sofa/pull/485)
- [SofaHelper]
    - mute extlibs warnings [#397](https://github.com/sofa-framework/sofa/pull/397)
    - FileMonitor_windows compilation [#448](https://github.com/sofa-framework/sofa/pull/448)
    - Clean API message [#503](https://github.com/sofa-framework/sofa/pull/503)

**Other modules**
- [SofaGeneralEngine]
    - add test and minor cleaning for IndexValueMapper [#319](https://github.com/sofa-framework/sofa/pull/319)
- [SofaGeneralObjectInteraction]
    - Remove stiffness multiplicator in SpringForceField [#290](https://github.com/sofa-framework/sofa/pull/290)
- [SofaValidation]
    - move code to set default folder for monitor to init function [#500](https://github.com/sofa-framework/sofa/pull/500)

**Applications**
- [all]
    - FIX: compilation warnings [#361](https://github.com/sofa-framework/sofa/pull/361)
- [CGALPlugin]
    - Fix warnings [#361](https://github.com/sofa-framework/sofa/pull/361)
- [image]
    - Fix warnings [#361](https://github.com/sofa-framework/sofa/pull/361)
- [Registration]
    - Remove deprecated scene [#331](https://github.com/sofa-framework/sofa/pull/331)
- [SofaPython]
    - General cleaning [#304](https://github.com/sofa-framework/sofa/pull/304)
    - Fix warnings [#361](https://github.com/sofa-framework/sofa/pull/361)
    - print cleaning + SofaPython quaternion dot product [#404](https://github.com/sofa-framework/sofa/pull/404)
- [runSofa]
    - Clean : remove non-ASCII chars in string [#327](https://github.com/sofa-framework/sofa/pull/327)


____________________________________________________________



## [v17.06](https://github.com/sofa-framework/sofa/tree/v17.06)

[Full log](https://github.com/sofa-framework/sofa/compare/v16.12...v17.06)


### New features

**For users**
- [SceneCreator]
    - new methods to add basic 3D object: Cube, Cylinder, Sphere and Plane. In rigid or deformable.
- [GeneralTopology]
    - SphereGridTopology component to create sphere grids, similar to CylinderGridTopology.
- Adds a new orientedBox dataField in BoxROI so that we can use it to either defined AABoxes or OrientedBox
- Minor improvement on the way warning/error message are presented to the users in runSofa. A single panel is now used instead of of two, it is always displayed, the Panel name also contains the number of message eg: "Messages(5)"
- The Graph view is now displaying the type of message they contains.
- [runSofa]
    - Autoload plugins, described in the user-custom file 'plugin_list.conf' if present; else 'plugin_list.conf.default' containing all compiled plugins and generated automatically by CMake.

**For developpers**
- Add a Logger component that stores the history of messages into each sofa component.
- Implements new methods to write on-liner's conditional message:
     msg_info_when(level<3) << "This is a conditional info message"
- Implement an implicit version of each of the msg_* API allowing to write
     msg_info() << "Hello"  in place for msg_info(this) << Hello"
- CImgPlugin : creation of a dedicated plugin for image loading based on CImg [#185](https://github.com/sofa-framework/sofa/pull/185)
- Remove deprecated miniBoost dependency [#273](https://github.com/sofa-framework/sofa/pull/273)


### Improvements

**Modules**
- [all]
    - update containers to support c++x11 features [#113](https://github.com/sofa-framework/sofa/pull/113)
    - speed up spheres rendering + code cleaning [#170](https://github.com/sofa-framework/sofa/pull/170)
    - updates externs/gtest to a fresh checkout [#213](https://github.com/sofa-framework/sofa/pull/213)
    - auto-init/cleanup libraries [#168](https://github.com/sofa-framework/sofa/pull/168)
    - Improve and clean msg_api and logging of message (#190, #255, #275). See [documentation](https://www.sofa-framework.org/community/doc/programming-with-sofa/logger/) for more information.
    - Add CMake option to limit cores used to build specific targets [#254](https://github.com/sofa-framework/sofa/pull/254)
    - Fix rgbacolor parsing [#305](https://github.com/sofa-framework/sofa/pull/305)
    - CMake: installing gtest headers for separate plugin builds [#395](https://github.com/sofa-framework/sofa/pull/395)
- [SofaKernel]
    - Update the RichConsoleStyleMessageFormatter  [#126](https://github.com/sofa-framework/sofa/pull/126)
    - creation of a defaulttype::RGBAColor [#119](https://github.com/sofa-framework/sofa/pull/119)
    - add a new method in BaseObjectDescription [#161](https://github.com/sofa-framework/sofa/pull/161)
    - adding listener mechanism to SceneLoader [#205](https://github.com/sofa-framework/sofa/pull/205)
    - common usage for DiagonalMass and tests [#230](https://github.com/sofa-framework/sofa/pull/230)
    - add tests for DataFileName [#250](https://github.com/sofa-framework/sofa/pull/250)
    - add tests for DefaultAnimationLoop [#258](https://github.com/sofa-framework/sofa/pull/258)
    - add tests for LocalMinDistance [#258](https://github.com/sofa-framework/sofa/pull/258)
    - add a way to convert message type to string in Message.cpp [#213](https://github.com/sofa-framework/sofa/pull/213)
    - MeshSTL.cpp replace a std:cerr by a msg_error so that FIX the corresponding failing test [#213](https://github.com/sofa-framework/sofa/pull/213)
    - adding listener mechanism to SceneLoader [#204](https://github.com/sofa-framework/sofa/pull/204)
    - Grid Topologies cleanup + new SphereGrid [#164](https://github.com/sofa-framework/sofa/pull/164)
    - Add CMake option SOFA_WITH_EXPERIMENTAL_FEATURES (default OFF) to enable MechanicalObject::buildIdentityBlocksInJacobian [#276](https://github.com/sofa-framework/sofa/pull/276)
- [SofaGraphComponents]
    - add tests for RequiredPlugin [#258](https://github.com/sofa-framework/sofa/pull/258)
- [SofaHelper]
    - GLSL: load shader source code from a standard string [#158](https://github.com/sofa-framework/sofa/pull/158)
- [SofaBaseTopology]
    - GridTopology : implement "flat" grids in 1 or 2 dimension by using setting grid resolution to "1" in the corresponding axis, and associated examples [#270](https://github.com/sofa-framework/sofa/pull/270)
    - add tests for RegularGridTopology [#270](https://github.com/sofa-framework/sofa/pull/270)
- [SofaEngine]
    - BREAKING: Add oriented box feature to BoxROI [#108](https://github.com/sofa-framework/sofa/pull/108)
- [SofaConstraint]
    - add instantiation of constraint corrections with Vec2f [#157](https://github.com/sofa-framework/sofa/pull/157)
- [SofaOpenglVisual]
    - add tests for ClipPlane [#258](https://github.com/sofa-framework/sofa/pull/258)
- [SofaVolumetricData]
    - add tests for DistanceGrid [#258](https://github.com/sofa-framework/sofa/pull/258)
    - add tests for Light [#258](https://github.com/sofa-framework/sofa/pull/258)
- [SofaBoundaryCondition]
    - add tests for ConstantForceField, some of them are OpenIssue demonstrating existing problem, as crashing sofa when using negative or too large values in indices  [#258](https://github.com/sofa-framework/sofa/pull/258)
- [CI]
    - improvement of all test scripts

**Applications**
- [GUI]
    - mouse events are now transmitted to the scene with QtGLViewer [#132](https://github.com/sofa-framework/sofa/pull/132)
- [SceneCreator]
    - Cosmetic changes and remove un-needed include [#169](https://github.com/sofa-framework/sofa/pull/169)
- [SofaPython]
    - Macros to bind "sequence" types [#165](https://github.com/sofa-framework/sofa/pull/165)
    - ModuleReload [#214](https://github.com/sofa-framework/sofa/pull/214)
    - light module reload [#202](https://github.com/sofa-framework/sofa/pull/202)
    - change the way createObject() handle its arguments to simplify scene writing + batch of tests [#286](https://github.com/sofa-framework/sofa/pull/286)
- [SofaTest]
    - add Backtrace::autodump to all tests to ease debugging [#191](https://github.com/sofa-framework/sofa/pull/191)
    - add automatic tests for updateForceMask [#209](https://github.com/sofa-framework/sofa/pull/209)
    - add tests on PluginManager [#240](https://github.com/sofa-framework/sofa/pull/240)
    - TestMessageHandler : new and robust implementation to connect msg_* message to test failure  [#213](https://github.com/sofa-framework/sofa/pull/213)
    - update to use the new TestMessageHandler where msg_error generates test failures [#213](https://github.com/sofa-framework/sofa/pull/213)
    - add tests for TestMessageHandler [#213](https://github.com/sofa-framework/sofa/pull/213)
- [SofaCUDA]
    - FIX NVCC flags for debug build on Windows [#300](https://github.com/sofa-framework/sofa/pull/300)


### Bug Fixes

**Modules**
- Warnings have been fixed [#229](https://github.com/sofa-framework/sofa/pull/229)
- [all]
    - check that SofaPython is found before lauching the cmake sofa_set_python_directory command [#137](https://github.com/sofa-framework/sofa/pull/137)
    - use the cmake install DIRECTORY instead of FILES to preserve the files hierarchy when installing [#138](https://github.com/sofa-framework/sofa/pull/138)
    - fixing issue related to parsing attributes with atof/atoi [#161](https://github.com/sofa-framework/sofa/pull/161)
    - unify color datafield [#206](https://github.com/sofa-framework/sofa/pull/206)
    - Fix CMakeLists bug on Sofa.ini and installedSofa.ini creation [#291](https://github.com/sofa-framework/sofa/pull/291)
    - Fix a lot of failing tests (#271, #279)
    - Fix compilation with SOFA_FLOATING_POINT_TYPE as float [#262](https://github.com/sofa-framework/sofa/pull/262)
    - CMake: Fix and clean boost, when using Sofa as an external lib [#421](https://github.com/sofa-framework/sofa/pull/421)
- [SofaKernel]
    - Fix the Filemonitor_test random failure on MacOs [#143](https://github.com/sofa-framework/sofa/pull/143)
    - implement a numerical integration for triangle [#249](https://github.com/sofa-framework/sofa/pull/249)
    - add brace initializer to helper::vector class [#252](https://github.com/sofa-framework/sofa/pull/252)
    - Activates thread-safetiness on MessageDispatcher. [#257](https://github.com/sofa-framework/sofa/pull/257)
    - Fix getRelativePath() in DataFileName [#250](https://github.com/sofa-framework/sofa/pull/250)
    - FileRepository::getRelativePath() lowering the case on WIN32 is now a deprecated behavior [#264](https://github.com/sofa-framework/sofa/pull/264)
    - Fix FileRepository should not be optional [#122](https://github.com/sofa-framework/sofa/pull/122)
    - FileMonitor: fix the recurrent problem with file 'SofaKernel/framework/framework_test/resources/existing.txt' pointed in Issue #146 [#258](https://github.com/sofa-framework/sofa/pull/258)
    - Fix wrong inline in exported functions [#449](https://github.com/sofa-framework/sofa/pull/449)
- [SofaFramework]
    - fix the integration scheme for Quaternion [#172](https://github.com/sofa-framework/sofa/pull/172) and fix values with which the quaternion is being compared after creation from euler angles
- [SofaHelper]
    - VisualToolGL: fix single primitive calls [#293](https://github.com/sofa-framework/sofa/pull/293)
    - ImagePNG: Fix library linking in debug configuration under MSVS [#298](https://github.com/sofa-framework/sofa/pull/298)
- [SofaBaseMechanics]
    - MechanicalObject: cleaning: symbols & include [#249](https://github.com/sofa-framework/sofa/pull/249)
- [SofaPhysicsAPI]
    - fix compilation of the project [#167](https://github.com/sofa-framework/sofa/pull/167)
- [SofaUserInteraction]
    - MouseInteractor: FIX the mouse picking on Mechanical Object [#282](https://github.com/sofa-framework/sofa/pull/282)

**Applications**
- [image]
    - Fixes #135 : Check that SofaPython is found before including python directory [#137](https://github.com/sofa-framework/sofa/pull/137)
    - Fixes #136 : Use the cmake install DIRECTORY instead of FILES [#138](https://github.com/sofa-framework/sofa/pull/138)
- [LeapMotion]
    - FIX compilation for LeapMotion plugin due to moved files [#296](https://github.com/sofa-framework/sofa/pull/296)
- [runSofa]
    - Fix minor consistency issues related to the readOnly flag [#115](https://github.com/sofa-framework/sofa/pull/115)
- [SofaTest]
    - repair the minor API breaks introduced by PR #213 [#269](https://github.com/sofa-framework/sofa/pull/269)

**Scenes**
- Components/engine/GenerateGrid.scn was fixed [#303](https://github.com/sofa-framework/sofa/pull/303)


### Cleaning

**Modules**
- [all]
    - clean the consistency issues related to the readOnly flag [#115](https://github.com/sofa-framework/sofa/pull/115)
    - Clean licenses [#139](https://github.com/sofa-framework/sofa/pull/139)
- [SofaKernel]
    - clean DefaultPipeline.cpp/h (API BREAKING)
    - clean the attributes names in BoxROI (API BREAKING)
    - TetrahedronFEMForceField clean code [#270](https://github.com/sofa-framework/sofa/pull/270)
    - GridTopology : clean the code & factor the constructor [#270](https://github.com/sofa-framework/sofa/pull/270)
    - RegularGridTopology : clean the constructor's code & remove NDEBUG code [#270](https://github.com/sofa-framework/sofa/pull/270)
    - MechanicalObject : removal of code specific to the grid [#270](https://github.com/sofa-framework/sofa/pull/270)

- [SofaVolumetricData]
    - Light: clean and strenghening the interface [#258](https://github.com/sofa-framework/sofa/pull/258)
    - DistanceGrid
- [SofaBoundaryCondition]
    - ConstantForceField: clean to follow sofa guideline & fix the "visible dependencies" [#258](https://github.com/sofa-framework/sofa/pull/258)
    - ConstantForceField: replace the "points" attribute by "indices" with backward compatibility & deprecation message [#258](https://github.com/sofa-framework/sofa/pull/258)

**Applications**
- [SceneCreator]
    - clean with cosmetic changes and removed un-needed includes
- [SofaPython]
    - cleaning data binding [#166](https://github.com/sofa-framework/sofa/pull/166)


### Moved files

- The module handling HighOrderTopologies moved into a [separate repository](https://github.com/sofa-framework/plugin.HighOrder) [#222](https://github.com/sofa-framework/sofa/pull/222)


____________________________________________________________



## [v16.12](https://github.com/sofa-framework/sofa/tree/v16.12)

**Last commit: on Jan 08, 2017**  
[Full log](https://github.com/sofa-framework/sofa/compare/v16.08...v16.12)

### Environment
- **C++11 is now mandatory**. This implies some changes in building tools.
    - Generator: CMake 3.1 or higher.
    - Compiler: GCC 4.8 or higher, Clang 3.4 or higher, Microsoft Visual C++ 2013 or higher.


### New features for users

- new Geomagic plugin: supporting latest versions of force feedback interfaces from Geomagic
- add ForceMaskOff, a component to locally (in a branch of the scene graph) cancel the force mask
- live-coding for python
- live-coding for GLSL
- new component MakeAlias
- new component MakeDataAlias
- improved error message & console rendering



### New features for developpers

- Preliminary Markdown support in the msg_* API. You can now write much better formatting & alignement as well as adding URL to documentations related to  the error.
- class RichStyleConsoleFormatter which interprete the markdowns in the message and format this to a resizable console with nice alignement.
- class CountingMessageHandler (count the number of message for each message type)
- class RoutingMessageHandler (to implement context specific routing of the messages to different handler)
- class ExpectMessage and MessageAsATestFailure can be used to check that a component did or didn't send a message and generate a test failure.
- FileMonitor is now implemented on MacOS & Windows (for live-coding features, for example)
- RequiredPlugin: modified API to take a list of plugins to load
- Implements the move semantics on sofa::helper::vector

### Improvements

- **372 new tests**: DAGNode, MeshObj, DiagonalMass, BoxROI, ComplementaryROI, DifferenceEngine, BilateralInteractionConstraint, Quaternion, ImagePNG, etc.
- 184/480 components have an associated example
- [SofaKernel]
    - replace raw pointers with a smart ones
    - add a ComponentState attribute to BaseObject
    - BaseData::typeName is now public: useful to debug
    - implement DataTrackerEngine, a kind of DataEngine but that is not a BaseObject
    - fix SVector<std::string>. The string serialization changed
- [SofaRigid]
    - in case jetJs is called several times per step
- [SofaGeneralLoader]
    - MeshVTKLoader can now read FIELD data of legacy file. Lookup tables are ignored.
- [SofaPython]
    - binding AssembledSystem as a new class in python
    - binding VisualModel::exportOBJ
    - binding for DataFileNameVector
    - add Compliant.getImplicitAssembledSystem(node)
    - SofaNumpy: bind/share a c++ 1d array as a numpy array
    - script.Controller :  handle optional arguments before createGraph
- [image]
    - raw import: add commented basic size verifications (could be performed in debug)
- [Flexible]
    - add support for shapefunction viewer
    - new feature of strain smoothing
    - improve readme file
- [Compliant]
    - simulation unit: convert gravity, dt
    - MaskMapping: every input are now mapped
    - add LinearDiagonalCompliance component
    - fix use of VLA in python mappings
    - improve readme file

### Bug Fixes

- fix ConstantForceField::updateForceMask()
- fix ObjExporter memory leak
- [SofaOpenGLVisual] OglTexture: fix possible memory leaks
- [Compliant]
    - clean python


### Cleaning

- clean the compilation when SOFA_NO_OPENGL flag is activated
- clean the config.h (SOFAGUI_HAVE_QWT)
- remove boost library links (includes still required). boost chrono is not required anymore.
- remove unused POINT_DATA_VECTOR_ACCESS macro
- make miniflowVR build optional (default OFF)
- [SofaKernel]
    - remove last direct opengl calls in modules
    - add deprecation message on MechanicalObject attributes
- [SofaBaseVisual] clean BaseCamera: remove direct opengl calls
- [SofaHaptics] boost-thread is not used any more, clean cmake
- [SofaGeneralLoader] STLLoader: fixing binary loading in debug and cleaning examples
- [SofaPython]
    - remove ScriptEnvironment i.e. automatic initialization of Node
    - Node::isInitialized(), not used anymore
- [Flexible]
    - clean relativeStrainMapping


### Moved files

- move CImg from extlibs to image plugin extlibs


### Documentation

- Add the contribution and guidelines : **CONTRIBUTING.md** and **GUIDELINES.md**
- Add the configuration required (ex: C++, compiler versions)
- Add a page to use SOFA in Matlab
- Improve Logger documentation
- Add a page to use SOFA in Matlab


____________________________________________________________



## [v16.08](https://github.com/sofa-framework/sofa/tree/v16.08)

**Last commit: on Jul 28, 2016**  
[Full log](https://github.com/sofa-framework/sofa/compare/v15.12...v16.08)

### New features

- SOFA on GitHub - [https://github.com/sofa-framework/sofa](https://github.com/sofa-framework/sofa)
- creation of a RigidScale plugin: implementing mappings, especially allowing to get the DOF with Rigid+Scale type, while reusing affine DOF (Rigid+Scale+Shear) already implemented in Flexible
- creation of a LeapMotion plugin: allowing to integrate a Leap in your SOFA simulation
- add the DrawTool: DrawTool is an interface, describing an API to display primitives on screen. For now, only the OpenGL (fixed-pipeline version) implementation has been made.
- add a Logger
- add the diffusion effect in SOFA (heat transfer)
- SOFA_USE_MASK compilation variable to activate or de-activate the masks in SOFA
- DataTracker: simple and elegant way to track Data in Engine
- extlibs: update cimg to version 1.7.3
- Add guidelines for contributions in CONTRIBUTING.md

### Moved files

- Kernel modules of SOFA (SofaFramework, SofaBase, SofaCommon and SofaSimulation) have been moved to one common module SofaKernel, located at _sofa_root/SofaKernel_. SofaKernel is a pure LGPL module.
- code in _sofa_root/modules/sofa/simulation/_ has been splitted into three modules of SofaKernel: SofaSimulationCommon, SofaSimulationTree, SofaSimulationGraph
- MOVE the SofaPardiso module as a plugin
- Move OglTetrahedralModel into a new plugin called VolumetricRendering

- Minor moves
    - Move ColorMap code to helper and let (Ogl)ColorMap from SofaOpenGLVisual doing OpenGL stuff
    - Move TorsionForceField and ComplementaryROI into SOFA (those two components where in a deprecated repository _sofa_root/modules/sofa/components/_)

- [Compliant]
    - moving propagate_constraint_force_visitor in a helper file and minor cleaning of CompliantImplicitSolver

### Improvements

- test examples are now running (on Jenkins for the Continuous Integration)
- Add unit test for quaternions
- Improving default mouse manipulation while picking a dof
- MouseWheel events now propagated

- Minor improvements
    - runSofa: force loading the SofaPython plugin if existing
    - runSofa: adding clang-style formatting (option '-z clang')
    - MechanicalObject: adding more visualisation colors for Rigids
    - SofaPluginManager: Clear description and components when removing last plugin
    - CMake: removing "-Wno-deprecated-register" compiler option that is only known by a few compilers
    - Collision:  add function setConstraintId in BaseConstraintSet
    - SPtr: up to 10 parameters in constructor
    - Add function in EigenBaseSparseMatrix in order to use eigen matrices with async solvers
    - Add CUSPARSESolver in SofaCudaSolversPlugin, this solver uses cusparse library to solve a sparse triangular system on the GPU
    - MeshBoundaryROI: allows specifying an input subset
    - ColorMap: with face culling enabled
    - ColorMap: adding a scale for the legend range values
    - ImageViewer: adding new boolean data field displaying meshes directly on the slices
    - ProjectionToPlaneMultiMapping: adding a projection factor to perform tricky stuff such as planar symmetry
    - ProjectionToTargetPlaneMapping: adding a factor to perform planar symmetry for example
    - DataDisplay: can now be displayed in wireframe
    - DataDisplay: display used topology
    - DataDisplay: fix and improving shading a bit
    - SofaEngine: add selectLabelROI engine
    - SofaEngine: add SelectConnectedLabelsROI
    - SofaBoundaryCondition: it is now possible to hide fixedconstraint (default to shown as before)
    - Mat.h: adding tensor product between vectors
    - ForceField: adding const getMState()
    - VolumetricRendering: Initialize tetra/hexa Data<> (to be able to link them as data in scenes)
    - SofaBaseVisual: Add modelview and projection matrices as data output
    - FrameBufferObject: check (and use) the default framebuffer for the current display window
    - SofaOpenGLVisual: add link to a potential external shader for VisualManagerSecondaryPass
    - Add OglTexturePointer
    - adding SquareDistanceMapping to compute the square distance between 2 points.
    - add OrderIndependentTransparency Manager (using two passes instead of three)
    - add OglOITShader to customize the shading of transparent surfaces
    - ProjectionToTargetLineMapping and ProjectionToTargetPlaneMapping with precomputed constant Jacobians and using last origin and direction/normal for extra points
    - Adding a timer embedding all the animation loop step but would need further doc.
    - display of indices has been improved
    - Add an example using cloth springs: examples/Components/forcefield/ClothBendingForceField.py
    - Improving a few examples by making the embedding nodes as non pickable (tag 'NoPicking')
    - Add BaseNode::getRootPath
    - Improving performances: - Message::class is now an enum
    - Updated draw method of PointSetGeometryAlgorithms, QuadSetGeometryAlgorithms, TetrahedronSetGeometryAlgorithms, TriangleSetGeometryAlgorithms
    - Add Blender-like orientation helper in the bottom-left part of the screen while drawing bbox (QtGlViewer)
    - Add Blender-like orientation helper in the bottom-left part of the screen while drawing bbox (QtViewer)
    - add of GUIEvent into the STLExporter
    - Make the code compatible with ClipPlane (using ClipVertex in shaders, which is deprecated for GLSL > 1.4)
    - Optimize the callto C-PYTHON side when the functions are not implemented in the python side
    - Add color attribute support (and default color if not present in the node)
    - Reactivate color map in TetraFEM, as it does not depend on SofaOpenGLVisual anymore
    - indices data field for UniformMass
    - analyze matrix only if number of non-zeroes has changed and no iterative refinement
    - update the applyConstraint methods according to the actual API
    - Adding ProjectionTo{Plane|Line}MultiMapping where the plane (origin,normal) and the line (origin,direction) are dofs
    - add MeshBoundaryROI with an example

- [Tests]
    - for (multi)Mapping test, check the size of the mapping output is correct
    - adding TestMessageHandler that raises a gtest failure each time an error message is processed
    - test for node removal
    - test for removal of a node containing an UncoupledConstraintCorrection (for now the test fails because there is a problem with the removal of that component)
    - add of Multi2Mapping_test
    - add DistanceMapping_test

- [SofaPython]
    - logger: cleaning emitter
    - sml.Mesh: adding load function
    - sml: python set for tags is created by objects themselves
    - sml: add tag to JointGeneric
    - sml: add the printLog flag
    - sml: setup units in BaseScene for all sml Scene class
    - sml: mesh has a clear id
    - sml:insertVisual: bug fix for solid with multiple meshes (just impacting the Display scene)
    - sml: handy constructor for Dof creation
    - sml: like <mesh> <image> can be defined in <solid>
    - sml: add a utility function: look into the valueByTag dictionary for a tag contained in tags
    - sml: adding a warning if a vertex group is empty
    - sml: can have offsets under solids
    - sml: remove deprecated setTagFromTag() method
    - API: add subsetFromDeformables function
    - binding Node::isInitialized
    - binding loadPythonSceneWithArguments
    - adding a binding to get the pointer of a Data (with its dimensions and type)
    - adding binding of BaseMapping::getJs (as dense matrices for now)
    - adding python functions to convert a Data in a numpy array with shared memory
    - adding a visitor to set all mstates as non pickable (such as picking will only be performed with collision models)
    - add tags to mesh groups
    - add a groupsByTag dict to easily iterate over groups knowing a tag
    - add of SceneDataIO to save and load the current state of a simulation
    - add of the method getDataFields
    - adding automatically tested examples
    - add tags to MeshAttributes
    - add a helper PythonScriptFunction_call and PythonScriptFunction_callNoResult to call a python controller function from c++ code
    - PythonScriptHelper -> PythonScriptControllerHelper: PythonScriptHelper: add convertion for float and std::string
    - Add the timingEnabled attribute to the PythonScriptController to control if the script is gather timing statistics
    - adding python module to load .obj files
    - adding BaseContext_getObject_noWarning that returns None w/o warning if the object is not found
    - improving a bit the conversion from a cpp Base* to a PyObject* when the cpp Base* type is (even partially) known.
    - adding a test to show how to bind a component outside of SofaPython
    - Factory: conversion shortcuts for known types
    - PythonScriptController: if the filename is let empty, the controller is supposed to be in an already loaded file, to avoid to read the same file several times
    - adding "loadPlugin" function to the Sofa python module
    - Add a getObjects() method to python BaseContext interface. Allow selection of objects based on type and name.
    - object and type names are now both optional when calling BaseContext_getObjects()
    - search direction can now optionally be passed to BaseContext_getObjects()
    - at object creation failure, print additional error messages from BaseObjectDescription
    - adding special Data types in the PythonFactory, so more specific cases can be added outside of the plugin SofaPython.
    - adding Node::addObject_noWarning not to print a warning when adding an object to an already initialized node
    - add of a method which compute quaternion from a line define by a director vector
    - add of few new features to save and load the current state of simulation

- [Flexible]
    - adding FlexibleCorotationalMeshFEMForceField (meta-forcefield). Not optimized but working
    - add of RigidScale mapping in addition to their tests and examples
    - HexaFEM comparison: each method has its own solver and uses the same decomposition so the only difference came from the deformation tensor used to find the rotation.
    - Flexible: WIP adding a meta-forcefield to compute regular corotational FEM. The idea is to use Flexible's components internally without adding extra computation neither extra memory
    - API: add strain offseting option
    - API: strainMappings as data members
    - API: use branching images for mass computation
    - API: make AffineMass, ShapeFunction, Behavior work in more cases
    - API: make Behavior work in simple cases with no label image
    - materials: removed checking for changed parameters at each step.
    - add example showing how to better handle partially filled elements using fine volumes
    - optimizing FlexibleCorotationalMeshFEMForceField by preassembling constant sub-graph
    - refactoring of MassFromDensity
    - adding a warning when creating a UniformMass on Affine/Quadratic frames.
    - add FEMDof class to python API
    - DeformationMapping: print a warning if a child particle has no parent
    - adding HEML implementation of St Venant-Kirchhoff materials (for tetrahedra).
    - if correct weights are given in mapping as input data, use it (even if a shapefunction is found)
    - use sout for logging
    - compute tangents for VisualModel loaded using loadVisual python function
    - transformEngine for Affine frames

- [Compliant]
    - sml: export of meshes
    - sml: the solids tags to be simulated as rigids are gathered in a set()
    - sml: geometricStiffness option
    - sml: using logger
    - implementing compliance unassembled API
    - in the python API, joints can be created in compliance or not
    - API: write a addSpring() in GenericRigidJoint, reuse it in children classes where possible
    - API: simplify usage of jointCompliance specification by tag
    - API: relative offset position is given to the AssembledRigidRigidMapping, and then computed at init into the MO
    - API: modifying the API to move an Offset
    - API: adding Offset::moveOffset to apply a rigid transformation to an offset (in its local frame)
    - API: new parameter to add non-mechanical Offsets and MappedPoints
    - API: collision mesh and visual model can be added to a Rigid Offset
    - adding an automatic execution of a scene based on a sml description
    - Constraint: adding a typeid for faster Constraint type comparisons
    - adding short name to create a ConstantCompliantPseudoStaticSolver
    - using tag on joints in a generic way, set their compliance / isCompliance value
    - added machinery to map data to numpy arrays, see example/numpy_data.py
    - added easy pure python mappings, see examples/numpy_mapping.py
    - added pure python forcefields, see examples/numpy_forcefield.py
    - AssembledRigidRigidMapping autoresize
    - geometric stiffness in python mappings
    - insertMergeRigid is coherent with solid tags usage
    - CompliantPseudoStaticSolver: avoiding an unnecessary propagation when the stopping criterion is the nb of iterations.
    - visualization in DifferenceFromTargetMapping
    - Frame.py: adding a function to force quaternion normalization (to avoid numerical drift)
    - added SimpleAnimationLoop
    - adding RigidRestJointMapping to compute a joint between a rigid body's current position and its rest position.
    - large compliances are considered as null stiffnesses
    - Offset default to isMechanical=True
    - python quaternion log
    - added nlnscg acceleration
    - pure python constraints
    - .inl for python mappings
    - adding Addition[Multi]Mapping
    - implementing AssembledMultiMapping::applyDJT
    - adding DotProduct[Multi]Mapping (with tests)
    - adding NormalizationMapping to map a 3d vector to its normalization
    - adding ConstantAssembled[Multi]Mapping
    - adding SafeDistanceMapping: a distance mapping that becomes a difference mapping for too small distances.
    - adding SafeDistanceFromTargetMapping
    - using the new SofaPython API
    - SafeDistanceFromTargetMapping can now be "safe" by giving the edge directions when they are known
    - adding the SofaCompliant python module (first module created outside of SofaPython!)
    - adding DotProductFromTargetMapping (with test)
    - adding RigidJointFromTargetMapping and RigidJointFromWorldFrameMapping
    - add of complementary API to create deformable articulated systems
    - adding NegativeUnilateralConstraint to guarantee negativeness
    - adding PenaltyForceField and using it in penalty contact response
    - add of two file from the SohusimDev plugin

- [image]
    - API: Sampler.addMechanicalObject() more versatile
    - API: refactor python API
    - API: add addClosingVisual()
    - add function in python API to retrieve perspective property
    - MeshToImageEngine: move getValue out of for loops
    - add a python ImagePlaneController
    - Data<Image<T>> are now specifically bound in python
    - remove pthread and X11 dependencies
    - add metaimage tags that may be used to define orientation
    - add python function to retrieve image type
    - simpler imagePlane python controller
    - add a createTransferFunction method
    - improved cutplane texture resolution
    - half perspective, half orthographic image transforms
    - add imageCoordValuesFromPositions engine

### Bug Fixes

- [PluginManager] crashed when a plugin was removed
- [SofaCUDA] fix the compilation using SofaCUDA on Windows
- unstable behavior of masks - USE-MASK variable added
- fix DAGNode traversal when a visitor is run from a node with a not up-to-date descendancy
- fix flaws in glText (memory leak and an other bug)
- EigenBaseSparseMatrix: fix parallel matrix-vector product
- XML export

- Minor fix
    - Sofa helper: leak when drawing 3d text
    - compilation with SofaDumpVisitor flag enabled
    - compilation of BezierTriangleSetGeometryAlgorithms (color changed from Vec3f to Vec4f)
    - runSofa: viewport aspect issue and loss of interaction in QtGLViewer
    - BoxROI: visual bounding box
    - SofaMiscForceField on Windows
    - VisualVisitor traversal order
    - SphereROI: indices out when multiple spheres
    - bug in RestShapeSpringsForceField
    - Remove VariationalSymplecticSolver.h from the package in SofaCommon (to fix history)
    - some static analysis warnings and errors (including memory leaks)
    - MeshROI: remove unnecessary sqrt
    - SphereROI: set centers' size to radii if only one radius is provided
    - ARTrack plugin compilation
    - bug in MeshNewProximityIntersection involving false positive detection with triangles containing an angle > 90°
    - path to images for html description pop up window
    - OglModel hasTexture
    - DataDisplay: normal computation
    - DataDisplay: crash when the component was dynamically added to a scene
    - visual bug with OglModel when putOnlyTexCoords is enabled with no texture
    - Order Independent Transparency for old graphics card
    - sofa::gui::glut applying changes in BaseCamera
    - computation of the bounding box in PlaneForceField
    - SofaHelper: Fix bug with FBO (causing some weird things when using textures)
    - corrected the visualization of heterogeneous tetrahedra
    - SofaOpenGLVisual: Fix Spotlight source drawing + add some log for ShaderElement
    - OmniDriverEmu plugin and examples
    - scene test: ICPRegistration_ShapeMatching.scn is ignored
    - Vec: 'normalized' function constness
    - SpotLight: direction normalization
    - ProjectionTo{Plane,Line}MultiMapping Jacobian insertion order
    - SofaGeneralRigid: bug in ArticulatedSystemMapping
    - SofaEngine: BoxROI instantiation
    - SofaBaseCollision: Fix computeBBox in SPhereModel
    - bug in MechanicalPickParticlesWithTagsVisitor input tags were not respected
    - SofaOpenGLVisual: fix light's modelview matrix computation (lookat data was not checked)
    - StateMask method clean needs to resize m_size to 0

- [Test-CI]
    - fix crash UncoupledConstraintCorrection_test
    - fix SofaComponentBase_test on windows
    - fix Mapping and MultiMapping tests
    - fix MultiMapping::applyDJT test
    - fix {Difference,Addition}Mapping when a pair is composed of the same index.
    - fix tested scenes selection
    - removed OptiTrackNatNet from "options" configurations
    - ignore some OptiTrackNatNet scenes testing

- [SofaPython]
    - fix GridTopology type on the python side
    - fix OBJ loader
    - fix loading a scene from a .py in a Node independently from the awful Simulation singleton.
    - fix SofaPython.Tools.localPath in some situations
    - fix BaseContext_getObjects so it can select objects from a base class name (and adding an example)
    - quaternion: fix singularity in from_line function

- [Flexible]
    - fix case sensitive issues
    - API: fix a bug with colors when reloading a scene
    - fix the bulk factor in NeoHookean material
    - fix NeoHookean traction test
    - testing detachFromGraph
    - BaseDeformationMapping: remove debug message, fix usage of sout (no need for testing f_printLog)
    - fix test compilation w/o image plugin
    - fix loadVisual
    - fix bug in topologygausspointsampler (computation of volume integrals for irregular hexa)


- [Compliant]
    - CompliantImplicitSolver: fix Lagrange multipliers clear when reseting the simulation
    - auto-fix init errors in RigidMass
    - Frame.py: adding tolist() conversion
    - fix Jacobian reserved size
    - fix contacts and associated test

- [image]
    - python tools: fix bug in mhd parsing
    - MeshToImage: fix bresenham, be sure dmax>=1
    - fix resampling of projection images, and marching cubes default parameter
    - fix bug in mhd file loader
    - fix rasterization when using vertex colors

### Cleaning

- warnings were removed
- dead branches were removed
- the ‘using’ directives were removed in header files
- the repository sofa_root/modules/sofa/components has been cleaned (deprecated components removed)
- clean many SOFA examples
- removing "using helper::vector" from VecTypes.h
- SofaQtGui: Remove qt3 remnants in ui files

- Minor clean
    - clean SofaBaseCollision of OpenGL dependency
    - cleaning Material::setColor
    - Base: write sout as info (rather than warnings)
    - clean and fix RestShapeSpringsForceField draw functions
    - Remove useless tests, optimize and fix potential bugs
    - cleanup, precompute barycenters for tetra and hexa
    - SofaBaseVisual: clean up and make consistent BaseCamera's code: clean QtViewer projection (remove OpenGL functions)
    - SofaBaseVisual: clean and fix BaseCamera Z clipping
    - SofaOpenGLVisual: cleanup Lights (remove Glu calls and set matrices as data) + Fix typo in Camera
    - quaternion to euler: not need for the hack since atan2 is used

- [SofaPython]
    - clean examples
    - clean the hard-binding example
    - clean noPicking visitor

- [Flexible]
    - remove unecessary data for Gauss points visualization
    - some clean regarding openmp parallelisation
    - clean metaFF
    - remove unecessary apply in reinit
    - clean FlexibleCorotationalMeshFEMForceField

- [Compliant]
    - clean RigidJoint{Multi}Mapping


____________________________________________________________



## [v15.12](https://github.com/sofa-framework/sofa/tree/v15.12)

[Full log](https://github.com/sofa-framework/sofa/compare/v15.09...v15.12)


____________________________________________________________



## [v15.09](https://github.com/sofa-framework/sofa/tree/v15.09)

[Full log](https://github.com/sofa-framework/sofa/compare/release-v15.12...v15.09)
