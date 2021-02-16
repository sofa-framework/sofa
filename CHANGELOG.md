# SOFA Changelog



## [v20.12](https://github.com/sofa-framework/sofa/tree/v20.12)

[Full log](https://github.com/sofa-framework/sofa/compare/v20.06...v20.12)


### SOFA-NG
Follow the SOFA-NG project on [its board](https://github.com/sofa-framework/sofa/projects/9) and [its main issue](https://github.com/sofa-framework/sofa/issues/1527).
- [SofaMisc] Pluginize all modules [#1306](https://github.com/sofa-framework/sofa/issues/1306)
- [SofaGeneral] Pluginize all modules [#1529](https://github.com/sofa-framework/sofa/issues/1529)
- [SofaCommon] Pluginize all modules [#1597](https://github.com/sofa-framework/sofa/issues/1597)
- [SofaBase] Package all modules [#1633](https://github.com/sofa-framework/sofa/issues/1633)
- [modules] Set relocatable flags to pluginized modules [#1604](https://github.com/sofa-framework/sofa/pull/1604)
- [SofaCommon] Make SofaCommon a deprecated collection [#1609](https://github.com/sofa-framework/sofa/pull/1609)
- [SofaGeneral] Make SofaGeneral a deprecated collection [#1596](https://github.com/sofa-framework/sofa/pull/1596)
- [SofaGeneral] Move BVH-format feature from Helper to SofaGeneralRigid [#1644](https://github.com/sofa-framework/sofa/pull/1644)


### Breaking
**Architecture**
- [SofaMacros] Refactor for better target and package management [#1433](https://github.com/sofa-framework/sofa/pull/1433)

**Modules**
- [All] Change index_type from size_t to uint [#1514](https://github.com/sofa-framework/sofa/pull/1514)
- [All] Deprecate m_componentstate and rename to d_componentState [#1358](https://github.com/sofa-framework/sofa/pull/1358)
- [All] Factorize and rename TopologyObjectType into TopologyElementType [#1593](https://github.com/sofa-framework/sofa/pull/1593)
- [All] Remove topologyAlgorithms classes [#1546](https://github.com/sofa-framework/sofa/pull/1546)
- [All] Standardize index type for Vector/Matrix templates [#1453](https://github.com/sofa-framework/sofa/pull/1453)
- [All] Uniform size type  [#1515](https://github.com/sofa-framework/sofa/pull/1515)
- **[SofaKernel]** Refactor BaseData to use DataLink [#1491](https://github.com/sofa-framework/sofa/pull/1491)
- **[SofaKernel]** ♻️ FIX & CLEANUP BoxROI [#1482](https://github.com/sofa-framework/sofa/pull/1482)
- **[SofaKernel]**[SofaCore][SofaLoader][SofaGeneralLoader] SOFA with callbacks [#1408](https://github.com/sofa-framework/sofa/pull/1408)

**Plugins / Projects**
- [ColladaSceneLoader][SofaAssimp] Move ColladaSceneLoader plugin content into SofaAssimp plugin [#1360](https://github.com/sofa-framework/sofa/pull/1360)
- [plugins] Remove plugins to be deleted [#1439](https://github.com/sofa-framework/sofa/pull/1439)


### Improvements
**Architecture**
- [All] Accelerating CMake generation [#1464](https://github.com/sofa-framework/sofa/pull/1464)
- [SofaMacros] Handle COMPONENTS (needed by SofaPython3) [#1671](https://github.com/sofa-framework/sofa/pull/1671)

**Modules**
- [All] Replace last use of Qwt by QtCharts and remove internal library [#1512](https://github.com/sofa-framework/sofa/pull/1512)
- [SofaBaseCollision] Add option to use of normal orientation in triangle and self-colliding cube [#1559](https://github.com/sofa-framework/sofa/pull/1559)
- **[SofaCore]** Add virtual getPathName function in Base [#1455](https://github.com/sofa-framework/sofa/pull/1455)
- [SofaGeneralLoader] Add option for transform in SphereLoader to match other loaders API [#1495](https://github.com/sofa-framework/sofa/pull/1495)
- [SofaGeneralLoader] allow ReadState at init [#1654](https://github.com/sofa-framework/sofa/pull/1654)
- [SofaHaptics] Add multithread test on LCPForceFeedback component [#1581](https://github.com/sofa-framework/sofa/pull/1581)
- [SofaHaptics] Add simple tests on LCPForceFeedback component [#1576](https://github.com/sofa-framework/sofa/pull/1576)
- [SofaImplicitField] Add new ImplicitFields and getHessian to ScalarField [#1510](https://github.com/sofa-framework/sofa/pull/1510)
- **[SofaKernel]** ADD: add polynomial springs force fields [#1342](https://github.com/sofa-framework/sofa/pull/1342)
- **[SofaKernel]** Add DataLink object & PathResolver. [#1485](https://github.com/sofa-framework/sofa/pull/1485)
- **[SofaKernel]** Add setLinkedBase method in BaseLink [#1436](https://github.com/sofa-framework/sofa/pull/1436)
- **[SofaKernel]** Add whole program optimization (aka link-time optimization) for msvc [#1468](https://github.com/sofa-framework/sofa/pull/1468)
- **[SofaKernel]** Exposing Data in ContactListener. [#1678](https://github.com/sofa-framework/sofa/pull/1678)
- **[SofaKernel]** Filerepository gettemppath [#1383](https://github.com/sofa-framework/sofa/pull/1383)
- **[SofaKernel]** Set read-only all data defined by the file loaded [#1660](https://github.com/sofa-framework/sofa/pull/1660)
- [SofaQtGui] Restore GraphWidget for Momentum and Energy using QtCharts instead of Qwt [#1508](https://github.com/sofa-framework/sofa/pull/1508)

**Plugins / Projects**
- [Compliant] Add WinchMultiMapping and ContactMultiMapping [#1557](https://github.com/sofa-framework/sofa/pull/1557)


### Bug Fixes
**Architecture**
- [CMake] FIX non-existent target with sofa_add_plugin [#1584](https://github.com/sofa-framework/sofa/pull/1584)
- [CMake] Fix Cmake configure step with SOFA_WITH_DEPRECATED_COMPONENTS [#1452](https://github.com/sofa-framework/sofa/pull/1452)

**Extlibs**
- [extlibs/gtest] Fix the broken sofa_create_package_with_targets in gtest [#1457](https://github.com/sofa-framework/sofa/pull/1457)

**Modules**
- [All] issofa_bugfix: cleans and fixes [#218](https://github.com/sofa-framework/sofa/pull/218)
- [SofaBaseLinearSolver] Fix logging info with SPARSEMATRIX_VERBOSE  [#1715](https://github.com/sofa-framework/sofa/pull/1715)
- [SofaBaseMechanics] Use d_showColor for indices instead of arbitrary white [#1511](https://github.com/sofa-framework/sofa/pull/1511)
- [SofaBaseMechanics] 🐛 FIX draw function in UniformMass [#1480](https://github.com/sofa-framework/sofa/pull/1480)
- [SofaCarving] Fix method doCarve should be called at AnimateEndEvent [#1532](https://github.com/sofa-framework/sofa/pull/1532)
- **[SofaCore]** FIX const correctness in DataTracker [#1488](https://github.com/sofa-framework/sofa/pull/1488)
- **[SofaCore]** FIX simu unload crash caused by missing checks on slaves ptrs [#1445](https://github.com/sofa-framework/sofa/pull/1445)
- **[SofaFramework]** Fix deprecated_as_error macro for MSVC [#1658](https://github.com/sofa-framework/sofa/pull/1658)
- [SofaGUI] Fix Cmake files for out-of-tree compilation [#1570](https://github.com/sofa-framework/sofa/pull/1570)
- [SofaGeneralAnimationLoop] Fix mechanical matrix mapper [#1587](https://github.com/sofa-framework/sofa/pull/1587)
- [SofaGeneralEngine] Fix BarycentricMapperEngine parse() function [#1516](https://github.com/sofa-framework/sofa/pull/1516)
- [SofaGeneralLoader] fix GIDMeshLoader and add example [#1554](https://github.com/sofa-framework/sofa/pull/1554)
- [SofaHelper/image] Fix unit tests [#1585](https://github.com/sofa-framework/sofa/pull/1585)
- **[SofaHelper]** Add SOFA/bin to SOFA_PLUGIN_PATH [#1718](https://github.com/sofa-framework/sofa/pull/1718)
- **[SofaHelper]** Link necessary Boost macro with SofaHelper (for Windows) [#1578](https://github.com/sofa-framework/sofa/pull/1578)
- **[SofaKernel]**[SofaGuiQt] Qt viewer with intel drivers [#1690](https://github.com/sofa-framework/sofa/pull/1690)
- **[SofaKernel]** Add updateOnTransformChange update callback on MeshLoader. [#1459](https://github.com/sofa-framework/sofa/pull/1459)
- **[SofaKernel]** Data file repository now looks into the SOFA install directory [#1656](https://github.com/sofa-framework/sofa/pull/1656)
- **[SofaKernel]** Improve check for already registered plugins [#1472](https://github.com/sofa-framework/sofa/pull/1472)
- **[SofaKernel]** In DataFileName, the name FILE used in the PathType enum could be ambigous  [#1465](https://github.com/sofa-framework/sofa/pull/1465)
- **[SofaKernel]** 🐛 Break link when passing a nullptr to setLinkedBase [#1479](https://github.com/sofa-framework/sofa/pull/1479)
- **[SofaKernel]**[SofaGeneralRigid] Minor fixes in ArticulatedSystemMapping and JointSpringForceField [#1448](https://github.com/sofa-framework/sofa/pull/1448)
- **[SofaKernel]** Implement an update mechanism on component: RequiredPlugin [#1458](https://github.com/sofa-framework/sofa/pull/1458)
- **[SofaKernel]** Switch to include_guard() instead of include_guard(GLOBAL) [#1467](https://github.com/sofa-framework/sofa/pull/1467)
- [SofaMacros] FIX RELOCATABLE_INSTALL_DIR target property [#1631](https://github.com/sofa-framework/sofa/pull/1631)
- [SofaMacros] FIX deprecated macro sofa_generate_package [#1446](https://github.com/sofa-framework/sofa/pull/1446)
- [SofaMacros] FIX libs copy and plugin deps finding [#1708](https://github.com/sofa-framework/sofa/pull/1708)
- [SofaMacros] FIX missing lib copy on Windows [#1711](https://github.com/sofa-framework/sofa/pull/1711)
- [SofaMacros] FIX plugins RPATH [#1619](https://github.com/sofa-framework/sofa/pull/1619)
- [SofaMacros] Improve RPATH coverage on MacOS [#1713](https://github.com/sofa-framework/sofa/pull/1713)
- [SofaMacros] Recursive deps search for RPATH [#1710](https://github.com/sofa-framework/sofa/pull/1710)
- [SofaOpenglVisual] OglViewport: a fix for compatibility with modern OpenGL [#1500](https://github.com/sofa-framework/sofa/pull/1500)
- [SofaSimulationGraph] No reason to have moveChild in private [#1470](https://github.com/sofa-framework/sofa/pull/1470)

**Plugins / Projects**
- [CGALPlugin] Fix compilation for CGal version > 5 [#1522](https://github.com/sofa-framework/sofa/pull/1522)
- [CImgPlugin] CLEAN dependencies in CMakeLists [#1651](https://github.com/sofa-framework/sofa/pull/1651)
- [Flexible] FIX deps to pluginized modules [#1590](https://github.com/sofa-framework/sofa/pull/1590)
- [Geomagic] Fix scenes ForceFeedBack behavior due to a change in UncoupledConstraintCorrection [#1435](https://github.com/sofa-framework/sofa/pull/1435)
- [OmniDriverEmul] Fix thread behavior and replace boost/pthread by std::thread [#1665](https://github.com/sofa-framework/sofa/pull/1665)
- [SofaOpenCL] Fix Cmake configuration [#1647](https://github.com/sofa-framework/sofa/pull/1647)
- [SofaPython] Small fixes to import plugin and remove scene warnings [#1466](https://github.com/sofa-framework/sofa/pull/1466)
- [runSofa] CLEAN unused dep to SofaGeneralLoader [#1648](https://github.com/sofa-framework/sofa/pull/1648)
- [SofaSPHFluid] Fix compilation with std::execution [#1542](https://github.com/sofa-framework/sofa/pull/1542)

**Examples / Scenes**
- [examples] Fix HexahedronForceFieldTopologyChangeHandling scene [#1573](https://github.com/sofa-framework/sofa/pull/1573)

**Scripts / Tools**
- [scripts] Update licenseUpdater [#1498](https://github.com/sofa-framework/sofa/pull/1498)


### Cleanings
**Architecture**
- [SofaMacros] Split SofaMacros.cmake into multiple files [#1477](https://github.com/sofa-framework/sofa/pull/1477)
- [SofaMacros] Use IN_LIST (CMake >= 3.3) [#1484](https://github.com/sofa-framework/sofa/pull/1484)

**Modules**
- [All] Bunch of removal of sout/serr in the whole code base [#1513](https://github.com/sofa-framework/sofa/pull/1513)
- [All] Fix compilation with flag NO_OPENGL [#1478](https://github.com/sofa-framework/sofa/pull/1478)
- [All] Fix many warnings [#1682](https://github.com/sofa-framework/sofa/pull/1682)
- [All] Remove SMP-related Code [#1613](https://github.com/sofa-framework/sofa/pull/1613)
- [All] Replace all sofa::defaulttypes::RGBAColor to sofa::helper::types::RGBAColor [#1463](https://github.com/sofa-framework/sofa/pull/1463)
- [Doc] Remove Inria Foundation mention from CONTRIBUTING [#1451](https://github.com/sofa-framework/sofa/pull/1451)
- [SofaBaseTopology] Fix ambiguity causing compilation error with MSVC [#1577](https://github.com/sofa-framework/sofa/pull/1577)
- [SofaBaseTopology] Rework method getIntersectionPointWithPlane [#1545](https://github.com/sofa-framework/sofa/pull/1545)
- [SofaBaseVisual][SofaDeformable] Clean some codes [#1449](https://github.com/sofa-framework/sofa/pull/1449)
- [SofaDeformable] Update RestShapeSpringsForceField [#1431](https://github.com/sofa-framework/sofa/pull/1431)
- [SofaGeneralEngine] Improve mesh barycentric mapper engine [#1487](https://github.com/sofa-framework/sofa/pull/1487)
- [SofaGeneralEngine] Remove useless create() function in some components [#1622](https://github.com/sofa-framework/sofa/pull/1622)
- [SofaGuiQt] Move libQGLViewer lib into its module [#1617](https://github.com/sofa-framework/sofa/pull/1617)
- [SofaHaptics] Small fix on LCPForceFeedback haptic rendering [#1537](https://github.com/sofa-framework/sofa/pull/1537)
- **[SofaHelper]** DrawTool uses RGBAColor now (instead of Vec4f) [#1626](https://github.com/sofa-framework/sofa/pull/1626)
- **[SofaHelper]** Remove OpenGL dependency in vector_device [#1646](https://github.com/sofa-framework/sofa/pull/1646)
- **[SofaKernel]** Clean namespace BarycentricMapper [#1571](https://github.com/sofa-framework/sofa/pull/1571)
- **[SofaKernel]** Factorize some code for maintenance [#1569](https://github.com/sofa-framework/sofa/pull/1569)
- **[SofaKernel]** Refactor the FileRepository constructors [#1610](https://github.com/sofa-framework/sofa/pull/1610)
- **[SofaKernel]** Remove core::Plugin/core::PluginManager [#1612](https://github.com/sofa-framework/sofa/pull/1612)
- **[SofaKernel]** Remove parentBaseData in  BaseData.h [#1490](https://github.com/sofa-framework/sofa/pull/1490)
- **[SofaKernel]** Remove support for BaseData in Link.h [#1503](https://github.com/sofa-framework/sofa/pull/1503)
- **[SofaKernel]** Remove un-needed StringUtil.h in Base.h [#1519](https://github.com/sofa-framework/sofa/pull/1519)
- **[SofaKernel]** Remove un-needed class reflection system. [#1541](https://github.com/sofa-framework/sofa/pull/1541)
- [SofaLoader] Use references in Meshloader [#1627](https://github.com/sofa-framework/sofa/pull/1627)
- [modules] Minor fixes [#1441](https://github.com/sofa-framework/sofa/pull/1441)

**Plugins / Projects**
- [plugins] Reactivate SofaPython3 [#1574](https://github.com/sofa-framework/sofa/pull/1574)
- [Geomagic] Update demo scenes to use direct solvers. [#1533](https://github.com/sofa-framework/sofa/pull/1533)
- [InvertibleFVM] Externalize the plugin [#1443](https://github.com/sofa-framework/sofa/pull/1443)


____________________________________________________________



## [v20.06](https://github.com/sofa-framework/sofa/tree/v20.06)

[Full log](https://github.com/sofa-framework/sofa/compare/v19.12...v20.06)


### Breaking

**Architecture**
- [All] CMake and includes massive clean [#1397](https://github.com/sofa-framework/sofa/pull/1397)
- [CMake] Remove the use of an internal Eigen3 version and instead use the one installed on the system. [#1281](https://github.com/sofa-framework/sofa/pull/1281)
- [CMake] Remove Collections [#1314](https://github.com/sofa-framework/sofa/pull/1314)
- [Modularization] SofaNonUniformFem + SofaAdvanced removal [#1344](https://github.com/sofa-framework/sofa/pull/1344)
- [Modularization] SofaValidation [#1302](https://github.com/sofa-framework/sofa/pull/1302)

**Modules**
- [All] BaseClass reflection refactoring [#1283](https://github.com/sofa-framework/sofa/pull/1283)
- [All] Remove Aspects from Sofa [#1269](https://github.com/sofa-framework/sofa/pull/1269)
- [All] Remove compilation warnings related to collision models [#1301](https://github.com/sofa-framework/sofa/pull/1301)
- [All] Update code base according to refactoring done in PR1283. [#1330](https://github.com/sofa-framework/sofa/pull/1330)
- [All] Remove all deprecation warnings after v1912 [#1241](https://github.com/sofa-framework/sofa/pull/1241)
- [All] v19.12 changes + DocBrowser by Defrost [#1275](https://github.com/sofa-framework/sofa/pull/1275)
- **[SofaBaseMechanics]** Change data name: handleTopologicalChanges in UniformM [#1291](https://github.com/sofa-framework/sofa/pull/1291)
- **[SofaCore/Visual]** Add new functions in drawTool (BREAKING) [#1252](https://github.com/sofa-framework/sofa/pull/1252)
- [SofaGeneralEngine] Refresh MeshBarycentricMapperEngine [#1404](https://github.com/sofa-framework/sofa/pull/1404)
- **[SofaCore]** ADD directory DataFileNames [#1407](https://github.com/sofa-framework/sofa/pull/1407)
- **[SofaKernel]** Refactor DDGNode [#1372](https://github.com/sofa-framework/sofa/pull/1372)
- **[SofaKernel]** Totally remove the macro CHECK_TOPOLOGY from BaseMeshTopology [#1399](https://github.com/sofa-framework/sofa/pull/1399)
- **[SofaKernel]** Update EulerExplicitSolver [#1260](https://github.com/sofa-framework/sofa/pull/1260)
- **[SofaKernel]** implement activer's code at CollisionModel. [#1259](https://github.com/sofa-framework/sofa/pull/1259)
 
**Plugins**
- [SofaCUDA] Arch auto-detection for nvcc [#1336](https://github.com/sofa-framework/sofa/pull/1336)


### Improvements

**Architecture**
- [CMake] Create an IDE folder for all relocatable_install targets [#1405](https://github.com/sofa-framework/sofa/pull/1405)

**Modules**
- **[SofaBaseVisual]** Add camera gizmo in base camera [#1253](https://github.com/sofa-framework/sofa/pull/1253)
- **[SofaCore]** Remove warning in ExecParam [#1325](https://github.com/sofa-framework/sofa/pull/1325)
- **[SofaCore]** ADD: DataCallback system in Base [#1406](https://github.com/sofa-framework/sofa/pull/1406)
- **[SofaDefaultType]** Add a Ray type. [#1251](https://github.com/sofa-framework/sofa/pull/1251)
- **[SofaHelper]** Add the obj id to labels when available [#1256](https://github.com/sofa-framework/sofa/pull/1256)
- **[SofaHelper]** Add auto-friendly getWriteAccessors/getReadAcessor... [#1254](https://github.com/sofa-framework/sofa/pull/1254)
- **[SofaKernel]** Set the default compilation mode to c++17. [#1249](https://github.com/sofa-framework/sofa/pull/1249)
- **[SofaKernel]** Add DataTypeInfo for BoundingBox [#1373](https://github.com/sofa-framework/sofa/pull/1373)
- **[SofaKernel]** Cleaner output when the creation of an object fails [#1266](https://github.com/sofa-framework/sofa/pull/1266)
- **[SofaSimulationCore]** Add a way to fill a multi vector from a base vector with a MechanicalOperations (mops). [#1248](https://github.com/sofa-framework/sofa/pull/1248)

**Plugins / Projects**
- [plugins] Fix warnings for option compilation [#1316](https://github.com/sofa-framework/sofa/pull/1316)
- [sofa-launcher] Change doc on sofa-launcher to markdown [#1311](https://github.com/sofa-framework/sofa/pull/1311)
- [SofaCUDA] Compilation without OpenGL [#1242](https://github.com/sofa-framework/sofa/pull/1242)

**Examples / Scenes**
- [examples] Add a scene illustrating ShewchukPCG [#1420](https://github.com/sofa-framework/sofa/pull/1420)


### Bug Fixes

**Architecture**
- [CMake] Add SofaNonUniformFem to the dependencies of SofaMisc [#1384](https://github.com/sofa-framework/sofa/pull/1384)
- [SofaFramework/CMake] Create configuration type dir before copying lib [#1347](https://github.com/sofa-framework/sofa/pull/1347)
- [extlibs/gtest] Add character test in gtest paramName to allow dash character [#1265](https://github.com/sofa-framework/sofa/pull/1265)

**Modules**
- [All] Clean warnings for option config, again [#1339](https://github.com/sofa-framework/sofa/pull/1339)
- [All] Fix SOFA_LIBSUFFIX used in Debug by PluginManager [#1313](https://github.com/sofa-framework/sofa/pull/1313)
- [All] Overridden 'canCreate' methods should always log an error message when they fail [#1294](https://github.com/sofa-framework/sofa/pull/1294)
- **[SofaBaseTopology]** Fix GridTopology edge computation [#1323](https://github.com/sofa-framework/sofa/pull/1323)
- **[SofaBaseTopology]**[SofaExporter] Fix failing tests due to changes in topology [#1335](https://github.com/sofa-framework/sofa/pull/1335)
- [SofaConstraint] Fix test further to #1369 [#1386](https://github.com/sofa-framework/sofa/pull/1386)
- **[SofaEigen2Solver]** Fix CMake test on the version of Eigen [#1416](https://github.com/sofa-framework/sofa/pull/1416)
- **[SofaEngine]** Fix engine unit tests [#1280](https://github.com/sofa-framework/sofa/pull/1280)
- **[SofaEngine]** Fix Engine_test [#1338](https://github.com/sofa-framework/sofa/pull/1338)
- **[SofaFramework]** Windows/VS: Remove warnings flags from definitions [#1368](https://github.com/sofa-framework/sofa/pull/1368)
- [SofaGuiQt] Fix compilation for SOFA_DUMP_VISITOR_INFO [#1415](https://github.com/sofa-framework/sofa/pull/1415)
- [SofaGuiQt] Fix node graph [#1424](https://github.com/sofa-framework/sofa/pull/1424)
- [SofaHeadlessRecorder] Update headless recorder to use the new ffmpeg recorder [#1396](https://github.com/sofa-framework/sofa/pull/1396)
- **[SofaHelper]** AdvancedTimer wasn't using the good timer ids for the label assignments [#1244](https://github.com/sofa-framework/sofa/pull/1244)
- **[SofaHelper]** Fix unloading with PluginManager [#1274](https://github.com/sofa-framework/sofa/pull/1274)
- **[SofaHelper]** Fix fixed_array compilation with VS2019 [#1426](https://github.com/sofa-framework/sofa/pull/1426)
- **[SofaKernel]** Fix hexahedra detection in BoxROI [#1417](https://github.com/sofa-framework/sofa/pull/1417)
- **[SofaKernel]** Fix minor bug in BoxROI and add warning message in RestShapeSpringsForceField [#1391](https://github.com/sofa-framework/sofa/pull/1391)
- **[SofaKernel]** Fixes a bug where the camera was not moving with the Qt viewer [#1377](https://github.com/sofa-framework/sofa/pull/1377)
- **[SofaKernel]** Improve error message when a component cannot be created. [#1332](https://github.com/sofa-framework/sofa/pull/1332)
- **[SofaKernel]** Remove the installation of external system libraries [#1387](https://github.com/sofa-framework/sofa/pull/1387)
- **[SofaKernel]** Set default visibility to SOFA_EXPORT_DYNAMIC_LIBRARY [#1410](https://github.com/sofa-framework/sofa/pull/1410)
- [SofaMiscTopology] Fix bug in TopologicalChangeProcessor [#1247](https://github.com/sofa-framework/sofa/pull/1247)
- **[SofaSimpleFEM]** Small Fix [#1403](https://github.com/sofa-framework/sofa/pull/1403)
- **[SofaSimulationCore]** FIX resizing of bboxes in UpdateBoundingBoxVisitor [#1268](https://github.com/sofa-framework/sofa/pull/1268)
- [SofaTopologyMapping] Fix Tetra2TriangleTopologicalMapping [#1319](https://github.com/sofa-framework/sofa/pull/1319)

**Plugins / Projects**
- [Geomagic] Fix several wrong behaviors in driver code [#1378](https://github.com/sofa-framework/sofa/pull/1378)
- [MeshSTEPLoader] FIX OCC version check [#1312](https://github.com/sofa-framework/sofa/pull/1312)
- [MeshSTEPLoader] FIX build with OCC >= 7.4 [#1310](https://github.com/sofa-framework/sofa/pull/1310)
- [Modeler] FIX link error on Windows [#1282](https://github.com/sofa-framework/sofa/pull/1282)
- [SofaMiscCollision] Fix topological changes in TetrahedronCollisionModel  [#1354](https://github.com/sofa-framework/sofa/pull/1354)
- [SofaSphFluid] Fix broken behavior for ParticleSink and ParticleSource [#1278](https://github.com/sofa-framework/sofa/pull/1278)
- [SofaSphFluid] FIX .scene-tests [#1317](https://github.com/sofa-framework/sofa/pull/1317)
- [SofaOpenCL] Make it work with 20.06 [#1361](https://github.com/sofa-framework/sofa/pull/1361)
- [SofaPython] Restrict the plugin and its dependers to C++11 [#1284](https://github.com/sofa-framework/sofa/pull/1284)

**Examples / Scenes**
- [examples] Fix SurfacePressureForceField example [#1273](https://github.com/sofa-framework/sofa/pull/1273)
- [examples] Fix caduceus [#1392](https://github.com/sofa-framework/sofa/pull/1392)
- [examples] Update the scene StandardTetrahedralFEMForceField.scn [#1064](https://github.com/sofa-framework/sofa/pull/1064)


### Cleanings

**Architecture**

**Modules**
- [All] Clean namespace for some classes [#1422](https://github.com/sofa-framework/sofa/pull/1422)
- [All] Fix warnings due to visibility attribute [#1421](https://github.com/sofa-framework/sofa/pull/1421)
- [All] Clean due to doc [#1398](https://github.com/sofa-framework/sofa/pull/1398)
- [All] Clean warnings [#1376](https://github.com/sofa-framework/sofa/pull/1376)
- [All] Fix minor warnings [#1388](https://github.com/sofa-framework/sofa/pull/1388)
- [All] Fix warnings generated due to change in Aspects [#1329](https://github.com/sofa-framework/sofa/pull/1329)
- [All] Minor changes in comment or format [#1411](https://github.com/sofa-framework/sofa/pull/1411)
- [All] Multiple fixes scenes [#1289](https://github.com/sofa-framework/sofa/pull/1289)
- [All] Remove all DISPLAY_TIME define [#1267](https://github.com/sofa-framework/sofa/pull/1267)
- [All] Remove some compilation warning. [#1343](https://github.com/sofa-framework/sofa/pull/1343)
- [All] Replace usage of sleep functions for the std one [#1394](https://github.com/sofa-framework/sofa/pull/1394)
- [All] Uniform use of M_PI [#1264](https://github.com/sofa-framework/sofa/pull/1264)
- [All] Update header for cleaner future management [#1375](https://github.com/sofa-framework/sofa/pull/1375)
- [All] replace all serr by msg_error/msg_warning [#1236](https://github.com/sofa-framework/sofa/pull/1236)
- [SofaConstraint] Set the use of integration factor true by default [#1369](https://github.com/sofa-framework/sofa/pull/1369)
- **[SofaDefaultType]** BoundingBox : Remove annoying warnings [#1425](https://github.com/sofa-framework/sofa/pull/1425)
- [SofaGeneralEngine] Fix draw of the sphere in SphereROI [#1318](https://github.com/sofa-framework/sofa/pull/1318)
- [SofaGeneralEngine] Remove remaining BoxROI after SofaEngine move [#1333](https://github.com/sofa-framework/sofa/pull/1333)
- [SofaGeneralLoader] Allow flip normals in Gmsh and STL loaders [#1418](https://github.com/sofa-framework/sofa/pull/1418)
- [SofaGui] Pass QDocBrowser as an option [#1315](https://github.com/sofa-framework/sofa/pull/1315)
- **[SofaKernel]** Add Data bool d_checkTopology in Topology container to replace the use of CHECK_TOPOLOGY macro [#1351](https://github.com/sofa-framework/sofa/pull/1351)
- **[SofaKernel]** Clean occurrences of CHECK_TOPOLOGY macro in code [#1352](https://github.com/sofa-framework/sofa/pull/1352)
- **[SofaKernel]** Clean of Material.h/cpp [#1346](https://github.com/sofa-framework/sofa/pull/1346)
- **[SofaKernel]** Remove X11 dependency when SOFA_NO_OPENGL is enabled. [#1370](https://github.com/sofa-framework/sofa/pull/1370)
- **[SofaKernel]** Who hates warnings? [#1258](https://github.com/sofa-framework/sofa/pull/1258)
- **[SofaKernel]** replace all serr by msg_error/msg_warning [#1237](https://github.com/sofa-framework/sofa/pull/1237)
- [SofaSparseSolver] Move CSparse and metis into SofaSparseSolver [#1357](https://github.com/sofa-framework/sofa/pull/1357)

**Plugins / Projects**
- [CGALPlugin] Clean and pluginization [#1308](https://github.com/sofa-framework/sofa/pull/1308)
- [Geomagic] Move all code related to device model display in a dedicated class. [#1366](https://github.com/sofa-framework/sofa/pull/1366)
- [Geomagic] Fix compilation [#1393](https://github.com/sofa-framework/sofa/pull/1393)
- [ManifoldTopologies] Remove CHECK_TOPOLOGY macro occurrences [#1353](https://github.com/sofa-framework/sofa/pull/1353)
- [ManifoldTopologies] Update the license in init.cpp [#1414](https://github.com/sofa-framework/sofa/pull/1414)
- [OpenCTMPlugin] Fix compilation and clean before moving out [#1359](https://github.com/sofa-framework/sofa/pull/1359)
- [PluginExample] Update PluginExample [#1356](https://github.com/sofa-framework/sofa/pull/1356)
- [Regression] Update hash [#1321](https://github.com/sofa-framework/sofa/pull/1321)
- [SofaSphFluid] Clean SofaFluid plugin compilation warning. [#1296](https://github.com/sofa-framework/sofa/pull/1296)

**Examples / Scenes**
- [examples] Fix and remove 3 scenes with deprecated component [#1355](https://github.com/sofa-framework/sofa/pull/1355)
- [examples] Remove useless files and add MeshMatrixMass example [#1257](https://github.com/sofa-framework/sofa/pull/1257)
- [scenes] Fix scenes from alias [#1292](https://github.com/sofa-framework/sofa/pull/1292)
- [scenes] Remove scene warnings due to Rigid template [#1295](https://github.com/sofa-framework/sofa/pull/1295)
- [scenes] Fix alias warnings in scenes [#1279](https://github.com/sofa-framework/sofa/pull/1279)


____________________________________________________________



## [v19.12](https://github.com/sofa-framework/sofa/tree/v19.12)

[Full log](https://github.com/sofa-framework/sofa/compare/v19.06...v19.12)


### Breaking

**Architecture**
- [All] Improve extlibs integration [#1137](https://github.com/sofa-framework/sofa/pull/1137)
- [packages] Move all SofaComponent* + rename SofaAllCommonComponents [#1155](https://github.com/sofa-framework/sofa/pull/1155)

**Modules**
- [All] Add SingleLink to Topology to reveal all hidden GetMeshTopology [#1183](https://github.com/sofa-framework/sofa/pull/1183)
- [All] Remove ExtVecType [#1055](https://github.com/sofa-framework/sofa/pull/1055)
- [All] up change on GetMeshTopology [#1223](https://github.com/sofa-framework/sofa/pull/1223)
- [SofaBoundaryConditions] Apply doInternalUpdate API to ConstantForceField [#1145](https://github.com/sofa-framework/sofa/pull/1145)
- **[SofaKernel]** Replacing const char* with strings for group / help / widget etc. [#1152](https://github.com/sofa-framework/sofa/pull/1152)
- **[SofaKernel]** ADD: static method in events to retrieve the classname [#1118](https://github.com/sofa-framework/sofa/pull/1118)
- **[SofaKernel]** Set BaseData to non-persistant by default [#1191](https://github.com/sofa-framework/sofa/pull/1191)
- **[SofaKernel]** fix root's getPathName [#1146](https://github.com/sofa-framework/sofa/pull/1146)


### Improvements

**Architecture**
- [CMake] v19.06 changes [#1114](https://github.com/sofa-framework/sofa/pull/1114)
- [extlibs] Set Eigen as external project + upgrade to 3.2.10 [#1101](https://github.com/sofa-framework/sofa/pull/1101)
- [extlibs] Upgrade Qwt extlib from 6.1.2 to 6.1.4 [#1136](https://github.com/sofa-framework/sofa/pull/1136)

**Modules**
- [All] Add SingleLink to Topology to reveal hidden GetMeshTopology Part 2 [#1199](https://github.com/sofa-framework/sofa/pull/1199)
- [All] Add update internal mechanism [#1131](https://github.com/sofa-framework/sofa/pull/1131)
- [All] Update the SOFA Guidelines [#1135](https://github.com/sofa-framework/sofa/pull/1135)
- **[SofaBaseMechanics]** Add topological change in barycentric mapping [#1203](https://github.com/sofa-framework/sofa/pull/1203)
- **[SofaBaseMechanics]** Use doUpdateInternal API in DiagonalMass [#1150](https://github.com/sofa-framework/sofa/pull/1150)
- **[SofaBaseMechanics]** Use doUpdateInternal API in UniformMass [#1149](https://github.com/sofa-framework/sofa/pull/1149)
- **[SofaBaseTopology]** Add new geometric methods in TetrahedronSetGeometryAlgorythms [#1160](https://github.com/sofa-framework/sofa/pull/1160)
- **[SofaCore]** Remove thread specific declaration [#1129](https://github.com/sofa-framework/sofa/pull/1129)
- [SofaGeneralEngine] Added Rigid to Euler orientation export [#1141](https://github.com/sofa-framework/sofa/pull/1141)
- [SofaHaptics] Add mutex and option to lock the ForceFeedback computation [#1157](https://github.com/sofa-framework/sofa/pull/1157)
- **[SofaKernel]** ADD: DataTypeInfo<vector<string>> & improved  doc [#1113](https://github.com/sofa-framework/sofa/pull/1113)
- **[SofaKernel]** Add a strict option to the BoxROI to prevent partially inside element to be in the box. [#1127](https://github.com/sofa-framework/sofa/pull/1127)
- **[SofaKernel]** Add fixed_array_algorithm + RGBAColor::lighten [#1172](https://github.com/sofa-framework/sofa/pull/1172)
- **[SofaKernel]** Add new events to detect Initialization & Simulation start. [#1173](https://github.com/sofa-framework/sofa/pull/1173)
- **[SofaKernel]** Add option in StiffSpringFF to track list of input springs [#1093](https://github.com/sofa-framework/sofa/pull/1093)
- **[SofaKernel]** Change several AdvancedTimer logs for a better tracking [#1094](https://github.com/sofa-framework/sofa/pull/1094)
- **[SofaKernel]** Consistent SofaFramework modules [#1200](https://github.com/sofa-framework/sofa/pull/1200)
- **[SofaKernel]** Make componentState a real data field [#1168](https://github.com/sofa-framework/sofa/pull/1168)
- [SofaMiscForceField] Use doUpdateInternal API in MeshMatrixMass [#1151](https://github.com/sofa-framework/sofa/pull/1151)
- [SofaQtQuick] Pass extra command-line arguments for python scenes in a more high-level function call [#992](https://github.com/sofa-framework/sofa/pull/992)
- [SofaSphFluid] Add sprite-based point render [#1194](https://github.com/sofa-framework/sofa/pull/1194)
- [SofaSphFluid] Update rendering & other [#1215](https://github.com/sofa-framework/sofa/pull/1215)

**Plugins / Projects**
- [runSofa] Fix DataWidget display Speicherleck and long loading [#1181](https://github.com/sofa-framework/sofa/pull/1181)

**Examples / Scenes**
- [Examples] Add some mesh and PR1000 demo scene [#1112](https://github.com/sofa-framework/sofa/pull/1112)


### Bug Fixes

**Architecture**
- [CMake]**[SofaFramework]** Remove FFMPEG_exec target from the dependencies of SofaFramework [#1177](https://github.com/sofa-framework/sofa/pull/1177)
- [CMake] FIX Eigen finding [#1175](https://github.com/sofa-framework/sofa/pull/1175)
- [CMake] FIX unknown compiler option on VS2015 [#1192](https://github.com/sofa-framework/sofa/pull/1192)
- [SofaMacros] FIX default module version [#1123](https://github.com/sofa-framework/sofa/pull/1123)
- [SofaMacros] FIX sofa_set_install_relocatable escaped chars [#1154](https://github.com/sofa-framework/sofa/pull/1154)

**Modules**
- [All] Fix warnings [#1206](https://github.com/sofa-framework/sofa/pull/1206)
- [All] Fix warnings [#1167](https://github.com/sofa-framework/sofa/pull/1167)
- [All] Fix some warnings and OglAttribute handleTopologyChange [#1159](https://github.com/sofa-framework/sofa/pull/1159)
- [SofaBoundaryCondition] Fix FixedRotationConstraint when using more than one locked axis [#1119](https://github.com/sofa-framework/sofa/pull/1119)
- **[SofaBaseMechanics]** Make Uniform and DiagonalMass compatible with topo change [#1212](https://github.com/sofa-framework/sofa/pull/1212)
- **[SofaBaseTopology]** Fix SparseGrid obj loading + tests [#1231](https://github.com/sofa-framework/sofa/pull/1231)
- [SofaComponentAll] FIX SofaAllCommonComponents backward compatibility [#1204](https://github.com/sofa-framework/sofa/pull/1204)
- [SofaConstraint] Fix UncoupledConstraintCorrection topology change handling [#1115](https://github.com/sofa-framework/sofa/pull/1115)
- [SofaConstraint] Fix crash with PrecomputedConstraintCorrection [#1230](https://github.com/sofa-framework/sofa/pull/1230)
- **[SofaCore]** FIX decode functions in BaseClass [#1222](https://github.com/sofa-framework/sofa/pull/1222)
- **[SofaDefaulttype]** FIX too many ExtVec warnings with GCC [#1140](https://github.com/sofa-framework/sofa/pull/1140)
- [SofaExporter] Move bindings from SofaPython [#1095](https://github.com/sofa-framework/sofa/pull/1095)
- **[SofaFramework]** Add other orders for fromEuler() for Quaternions. [#1221](https://github.com/sofa-framework/sofa/pull/1221)
- **[SofaFramework]** Install the SofaSimulationCore target back into the SofaFramework package [#1182](https://github.com/sofa-framework/sofa/pull/1182)
- [SofaGuiQt] Fix unexpected symbol in CMakeLists [#1132](https://github.com/sofa-framework/sofa/pull/1132)
- [SofaGui] FIX missing find_package in SofaGuiConfig.cmake.in [#1198](https://github.com/sofa-framework/sofa/pull/1198)
- [SofaGui] Fix VideoRecorder [#1138](https://github.com/sofa-framework/sofa/pull/1138)
- [SofaGui] Prevent the GuiManager to store a pointer for the valid gui name [#1108](https://github.com/sofa-framework/sofa/pull/1108)
- [SofaHeadlessRecorder] FIX headlessRecorder_test [#1174](https://github.com/sofa-framework/sofa/pull/1174)
- **[SofaHelper]** FIX Eigen install path [#1240](https://github.com/sofa-framework/sofa/pull/1240)
- **[SofaKernel]** Add bloc access in basematrix [#1143](https://github.com/sofa-framework/sofa/pull/1143)
- **[SofaKernel]** Changes for Visual Studio and c++17 [#1162](https://github.com/sofa-framework/sofa/pull/1162)
- **[SofaKernel]** FIX regex in SofaMacros.cmake [#1161](https://github.com/sofa-framework/sofa/pull/1161)
- **[SofaKernel]** Fix alloc size [#1142](https://github.com/sofa-framework/sofa/pull/1142)
- **[SofaKernel]** Fix some AdvanceTimer log missing [#1158](https://github.com/sofa-framework/sofa/pull/1158)
- **[SofaKernel]** Fix useless warnings [#1144](https://github.com/sofa-framework/sofa/pull/1144)
- **[SofaKernel]** Several fix in draw methods and topology warnings [#1111](https://github.com/sofa-framework/sofa/pull/1111)
- **[SofaKernel]** Small Fix in CollisionModel [#1202](https://github.com/sofa-framework/sofa/pull/1202)
- **[SofaKernel]** Use links for input and output topologies of the barycentric mapping [#1125](https://github.com/sofa-framework/sofa/pull/1125)
- [SofaMisc] Fix compilation with SOFA_NO_OPENGL [#1193](https://github.com/sofa-framework/sofa/pull/1193)
- **[SofaSimulationGraph]** Fix CollisionGroupManager wrong search of deformable object node [#1060](https://github.com/sofa-framework/sofa/pull/1060)
- **[SofaSimulationGraph]** Stop DAGNode get parent topology process in BarycentricMapping [#1176](https://github.com/sofa-framework/sofa/pull/1176)
- [SofaSphFluid] Clean, Fix, Update ParticleSink [#1195](https://github.com/sofa-framework/sofa/pull/1195)

**Plugins / Projects**
- [All] Fix minor compilation issue in plugins [#1106](https://github.com/sofa-framework/sofa/pull/1106)
- [Carving plugin] Small fix at init. [#1110](https://github.com/sofa-framework/sofa/pull/1110)
- [Cgal plugin] Fix windows cmake dll path and add a scene example [#958](https://github.com/sofa-framework/sofa/pull/958)
- [Regression_test] Update regression test references for CollisionGroup [#1102](https://github.com/sofa-framework/sofa/pull/1102)


### Cleanings

**Architecture**
- [CMake] Use cmake_dependent_option for plugin tests [#1164](https://github.com/sofa-framework/sofa/pull/1164)

**Modules**
- [All] Fix order warnings [#1239](https://github.com/sofa-framework/sofa/pull/1239)
- [All] Fix override warning in option mode [#1210](https://github.com/sofa-framework/sofa/pull/1210)
- [All] Small cleaning on sout and serr [#1234](https://github.com/sofa-framework/sofa/pull/1234)
- [All] Standardize epsilons in SOFA [#1049](https://github.com/sofa-framework/sofa/pull/1049)
- [All] Code cleaning of multiple classes [#1116](https://github.com/sofa-framework/sofa/pull/1116)
- [All] Remove deprecated macro SOFA_TRANGLEFEM [#1233](https://github.com/sofa-framework/sofa/pull/1233)
- [All] Remove references to "isToPrint" because it's broken [#1197](https://github.com/sofa-framework/sofa/pull/1197)
- [All] Replace NULL by nullptr [#1179](https://github.com/sofa-framework/sofa/pull/1179)
- [All] Try to reduce the number of compilation warnings [#1196](https://github.com/sofa-framework/sofa/pull/1196)
- [SceneCreator] Pluginizing... [#1109](https://github.com/sofa-framework/sofa/pull/1109)
- **[SofaBaseLinearSolver]** Remove virtual function BaseLinearSolver::isMultiGroup [#1063](https://github.com/sofa-framework/sofa/pull/1063)
- **[SofaBaseLinearSolver][FullMatrix]**  Restore fast clear function [#1128](https://github.com/sofa-framework/sofa/pull/1128)
- **[SofaFramework]** Remove (painful) check/warning with Rigids [#1229](https://github.com/sofa-framework/sofa/pull/1229)
- [SofaGUI] Split OpenGL and Qt dependency [#1121](https://github.com/sofa-framework/sofa/pull/1121)
- [SofaGeneralObjectInteraction] Create delegate functions in AttachConstraint [#1185](https://github.com/sofa-framework/sofa/pull/1185)
- [SofaGraphComponent] Update sceneCheckerAPI and deprecate MatrixMass [#1107](https://github.com/sofa-framework/sofa/pull/1107)
- [SofaHAPI] Fixes for HAPI [#1189](https://github.com/sofa-framework/sofa/pull/1189)
- **[SofaKernel]** ADD const specifier on notify methods in Node [#1169](https://github.com/sofa-framework/sofa/pull/1169)
- **[SofaKernel]** Remove deprecated SOFA_DEBUG macro  [#1232](https://github.com/sofa-framework/sofa/pull/1232)
- **[SofaMeshCollision]** Clean deprecated code [#1201](https://github.com/sofa-framework/sofa/pull/1201)
- [SofaSphFluid] Clean code of ParticleSource and update scenes [#1190](https://github.com/sofa-framework/sofa/pull/1190)
- [SofaSphFluid] Reorder plugin code and scenes files [#1165](https://github.com/sofa-framework/sofa/pull/1165)
- [SofaSphFluid] Several clean in the plugin components [#1186](https://github.com/sofa-framework/sofa/pull/1186)
- [SofaSphFluid] missing namespace [#1188](https://github.com/sofa-framework/sofa/pull/1188)
- [SofaTest] CLEAN msg in Multi2Mapping_test [#1097](https://github.com/sofa-framework/sofa/pull/1097)
- [SofaTopologyMapping] Cleanups of some topological mappings + better initialization [#1126](https://github.com/sofa-framework/sofa/pull/1126)
- [SofaViewer] Prevent the GUI to ouput every CTRL actions in the console [#1130](https://github.com/sofa-framework/sofa/pull/1130)

**Plugins / Projects**
- [CGALPlugin] Some cleanups to CylinderMesh [#1124](https://github.com/sofa-framework/sofa/pull/1124)
- [CGal plugin][CImgPlugin] Image data moved from Image/ to CImgPlugin/ [#1104](https://github.com/sofa-framework/sofa/pull/1104)
- [Geomagic] Reorder plugin files for better modularization [#1208](https://github.com/sofa-framework/sofa/pull/1208)
- [ManifoldTopologies] Undust and clean [#1156](https://github.com/sofa-framework/sofa/pull/1156)

**Examples / Scenes**
- [Scenes] Clean some alias warnings [#1098](https://github.com/sofa-framework/sofa/pull/1098)
- [scenes] Change OglModel to use a MeshObjLoader instead of loading the mesh internally. [#1096](https://github.com/sofa-framework/sofa/pull/1096)


____________________________________________________________



## [v19.06](https://github.com/sofa-framework/sofa/tree/v19.06)

[Full log](https://github.com/sofa-framework/sofa/compare/v18.12...v19.06)


### Breaking

**Modules**
- [All] Run clang-tidy and update license headers [#899](https://github.com/sofa-framework/sofa/pull/899)
- [All] Refactor the loading of Xsp files. [#918](https://github.com/sofa-framework/sofa/pull/918)
- **[SofaBaseTopology]** Change triangles orientation in tetrahedron [#878](https://github.com/sofa-framework/sofa/pull/878)
- **[SofaBaseTopology]** Major Change in Topology Containers [#967](https://github.com/sofa-framework/sofa/pull/967)
- **[SofaKernel]** Refactor the MutationListener [#917](https://github.com/sofa-framework/sofa/pull/917)
- **[SofaKernel]** Some Topology cleaning... [#866](https://github.com/sofa-framework/sofa/pull/866)
- [SofaOpenglVisual] Fix ogl perf problem [#1069](https://github.com/sofa-framework/sofa/pull/1069)


### Modularizations

- [SofaExporter] Modularize (+minor dependency cleaning) [#915](https://github.com/sofa-framework/sofa/pull/915)
- [SofaHaptics] Modularize sofa haptics [#945](https://github.com/sofa-framework/sofa/pull/945)
- [SofaOpenglVisual] Pluginize. [#1080](https://github.com/sofa-framework/sofa/pull/1080)


### Improvements

**Architecture**
- [CMake] Rework sofa_generate_package [#951](https://github.com/sofa-framework/sofa/pull/951)
- [CMake] SofaMacros.cmake: deprecating sofa_create_package [#909](https://github.com/sofa-framework/sofa/pull/909)

**Modules**
- [All] Improve install and packaging [#1018](https://github.com/sofa-framework/sofa/pull/1018)
- [All] Plugins finding and loading [#913](https://github.com/sofa-framework/sofa/pull/913)
- [All] Replace deprecated c++ standard binder component [#908](https://github.com/sofa-framework/sofa/pull/908)
- **[SofaBaseMechanics]** BarycentricMapping: spatial hashing, handle limit cases [#896](https://github.com/sofa-framework/sofa/pull/896)
- **[SofaBaseTopology]** Clean Topology logs and add AdvanceTimer logs [#874](https://github.com/sofa-framework/sofa/pull/874)
- **[SofaBaseVisual]** Add default texcoord in VisualModel [#933](https://github.com/sofa-framework/sofa/pull/933)
- [SofaConstraint] ADD control on constraint force in UniformConstraint [#1027](https://github.com/sofa-framework/sofa/pull/1027)
- **[SofaCore]** Add possibilities to draw lines on surfaces in DrawTool [#937](https://github.com/sofa-framework/sofa/pull/937)
- **[SofaCore]** Collision visitor primitive tests count [#930](https://github.com/sofa-framework/sofa/pull/930)
- **[SofaCore]** ADD Datacallback and datalink [#911](https://github.com/sofa-framework/sofa/pull/911)
- [SofaEngine] Avoid Crash in BoxROI when rest_position is not yet defined [#1031](https://github.com/sofa-framework/sofa/pull/1031)
- [SofaExporter] Add option for Regression_test to check first and last iteration [#1061](https://github.com/sofa-framework/sofa/pull/1061)
- [SofaGeneralAnimationLoop] Improve MechanicalMatrixMapper [#882](https://github.com/sofa-framework/sofa/pull/882)
- [SofaGraphComponent] Run SceneChecker at each load [#938](https://github.com/sofa-framework/sofa/pull/938)
- [SofaGuiQt] Change the keyboard shortcut associated to camera mode [#997](https://github.com/sofa-framework/sofa/pull/997)
- [SofaGuiQt] Add a profiling window based on AdvanceTimer records [#1028](https://github.com/sofa-framework/sofa/pull/1028)
- **[SofaKernel]** Some small changes in debug topology drawing [#952](https://github.com/sofa-framework/sofa/pull/952)
- **[SofaKernel]** Update Static Solver [#950](https://github.com/sofa-framework/sofa/pull/950)
- **[SofaKernel]** Rename TModels into CollisionModels and update all scenes [#1034](https://github.com/sofa-framework/sofa/pull/1034)
- **[SofaKernel]** Add a new video recorder class VideoRecorderFFMPEG [#883](https://github.com/sofa-framework/sofa/pull/883)
- **[SofaSimulationCore]** Cpu task and scheduled thread support [#970](https://github.com/sofa-framework/sofa/pull/970)
- **[SofaSimulationCore]** call BaseObject::draw() during the Transparent pass [#929](https://github.com/sofa-framework/sofa/pull/929)
- [SofaTopologyMapping] Clean, fix, upgrade Tetra2TriangleTopologicalMapping [#876](https://github.com/sofa-framework/sofa/pull/876)

**Plugins / Projects**
- [Geomagic] Add some better check at init and method to free driver [#925](https://github.com/sofa-framework/sofa/pull/925)
- [Icons] EDIT Sofa icons [#881](https://github.com/sofa-framework/sofa/pull/881)
- [MultiThreading] TaskAllocator Interface [#906](https://github.com/sofa-framework/sofa/pull/906)
- [PluginExample] Update example + add comments [#1053](https://github.com/sofa-framework/sofa/pull/1053)
- [Regression] ADD Regression as external project [#1052](https://github.com/sofa-framework/sofa/pull/1052)
- [runSofa] ADD possibility to jump to source/instanciation of selected component [#1013](https://github.com/sofa-framework/sofa/pull/1013)
- [SofaCUDA] Fix cuda with latest API [#912](https://github.com/sofa-framework/sofa/pull/912)
- [SofaPython] Add Sofa.hasViewer function [#964](https://github.com/sofa-framework/sofa/pull/964)
- [SofaPython] Change Base.addNewData [#1004](https://github.com/sofa-framework/sofa/pull/1004)

**Examples / Scenes**
- [examples] Rename TModels into CollisionModels and update all scenes [#1034](https://github.com/sofa-framework/sofa/pull/1034)


### Bug Fixes

**Architecture**
- [CMake] Add check to prevent the inclusion of non-existant file in cmake 3.13 [#897](https://github.com/sofa-framework/sofa/pull/897)
- [CMake] Fix relocatable plugins [#1059](https://github.com/sofa-framework/sofa/pull/1059)
- [CMake] FIX: exporting options in SofaFrameworkConfig.cmake [#927](https://github.com/sofa-framework/sofa/pull/927)
- [CMake] FIX: wrong paths of installed headers in SofaBaseMechanics [#887](https://github.com/sofa-framework/sofa/pull/887)
- [CMake] FIX build/install plugins directory [#959](https://github.com/sofa-framework/sofa/pull/959)

**Modules**
- [All] Three small fixes in SofaBaseLinearSolver, SofaBoundaryCondition, runSofa [#931](https://github.com/sofa-framework/sofa/pull/931)
- [All] FIXES made for RoboSoft2019 [#1003](https://github.com/sofa-framework/sofa/pull/1003)
- [All] Fix some warnings [#873](https://github.com/sofa-framework/sofa/pull/873)
- [All] Several bug fixes [#985](https://github.com/sofa-framework/sofa/pull/985)
- [All] Some fixes to have a ... green dashboard! [#982](https://github.com/sofa-framework/sofa/pull/982)
- [All] Fix compilation with SOFA_NO_OPENGL flag [#1032](https://github.com/sofa-framework/sofa/pull/1032)
- [SofaConstraint] Convert static sized arrays to dynamic ones in GenericConstraintSolver [#920](https://github.com/sofa-framework/sofa/pull/920)
- **[SofaBaseMechanics]** Fix barycentric mapping again [#924](https://github.com/sofa-framework/sofa/pull/924)
- **[SofaBaseTopology]** Fix Crash when loading a vtk file generated by Gmsh using TetrahedronSetTopologyContainer as container [#1008](https://github.com/sofa-framework/sofa/pull/1008)
- **[SofaBaseTopology]** Fix right setDirty/clean topologyData  [#889](https://github.com/sofa-framework/sofa/pull/889)
- **[SofaBaseTopology]**[DrawTools] Some fix/update in topology internal draw methods. [#877](https://github.com/sofa-framework/sofa/pull/877)
- **[SofaBaseTopology]** Yet another fix in Tetra2triangleTopologicalMapping [#998](https://github.com/sofa-framework/sofa/pull/998)
- **[SofaBaseTopology]** Clean, fix, upgrade Triangle2EdgeTopologicalMapping [#875](https://github.com/sofa-framework/sofa/pull/875)
- **[SofaBaseTopology]** Fix crashes in Tetra2TriangleTopologicalMapping  [#960](https://github.com/sofa-framework/sofa/pull/960)
- [SofaBoundaryCondition] Fix draw function in ConstantForcefield [#1017](https://github.com/sofa-framework/sofa/pull/1017)
- **[SofaDeformable]** FIX issue 928 [#942](https://github.com/sofa-framework/sofa/pull/942)
- **[SofaDeformable]** Merge 2 ctor in SpringForceField [#948](https://github.com/sofa-framework/sofa/pull/948)
- [SofaExporter] FIX: out-of-tree include of SofaExporter header files [#975](https://github.com/sofa-framework/sofa/pull/975)
- [SofaGeneralLoader] Compute subElement by default for Gmsh format [#986](https://github.com/sofa-framework/sofa/pull/986)
- [SofaGeneralObjectInteraction] Fix AttachConstraint in case of FreeMotion (LM solving) [#949](https://github.com/sofa-framework/sofa/pull/949)
- [SofaGeneralObjectInteraction] Fix attach constraint radius [#650](https://github.com/sofa-framework/sofa/pull/650)
- [SofaGui] Fix missing profiling timers for BatchGUI and HeadlessRecorder [#890](https://github.com/sofa-framework/sofa/pull/890)
- [SofaGuiGlut] Fix compilation [#1044](https://github.com/sofa-framework/sofa/pull/1044)
- [SofaGuiQt] FIX: component/nodes ordering in runSofa scene graph [#1001](https://github.com/sofa-framework/sofa/pull/1001)
- [SofaGuiQt] REMOVE: public export of target SofaExporter [#963](https://github.com/sofa-framework/sofa/pull/963)
- [SofaGuiQt] Fix: several QWidget do not have a parent [#1030](https://github.com/sofa-framework/sofa/pull/1030)
- **[SofaHelper]** FIX compilation on Visual Studio 2015 with QWT plugin [#935](https://github.com/sofa-framework/sofa/pull/935)
- **[SofaHelper]** FIX WinDepPack INSTALL_INTERFACE [#1042](https://github.com/sofa-framework/sofa/pull/1042)
- **[SofaHelper]** REMOVE PluginManager::m_searchPaths [#947](https://github.com/sofa-framework/sofa/pull/947)
- **[SofaKernel]** Clean & Fix TopologyChangeVisitor and StateChangeVisitor behavior [#880](https://github.com/sofa-framework/sofa/pull/880)
- **[SofaKernel]** Clean output data when doUpdate in BoxROI [#1056](https://github.com/sofa-framework/sofa/pull/1056)
- **[SofaKernel]** FIX deprecation message related to template types. [#939](https://github.com/sofa-framework/sofa/pull/939)
- **[SofaKernel]** FIX in TetrahedronFEMForceField & TetrahedronSetTopologyAlgorithm [#973](https://github.com/sofa-framework/sofa/pull/973)
- **[SofaKernel]** FIX operator>> in Mat.h and add corresponding test. [#993](https://github.com/sofa-framework/sofa/pull/993)
- **[SofaKernel]** FIX: A few fix to compile on Mac OSX Xcode 9 and Linux gcc 7.3.0 [#969](https://github.com/sofa-framework/sofa/pull/969)
- **[SofaKernel]** FIX: force name data to contain something [#1009](https://github.com/sofa-framework/sofa/pull/1009)
- **[SofaKernel]** Fix error in MapperHexahedron and MapperQuad barycentric coef computation [#1057](https://github.com/sofa-framework/sofa/pull/1057)
- **[SofaKernel]** Fix: remove unwanted AdvanceTimer::begin command [#1029](https://github.com/sofa-framework/sofa/pull/1029)
- **[SofaKernel]** Remove warnings [#968](https://github.com/sofa-framework/sofa/pull/968)
- **[SofaKernel]** several small fix [#953](https://github.com/sofa-framework/sofa/pull/953)
- [SofaLoader] Fix positions when handleSeams is activated in MeshObjLoader [#923](https://github.com/sofa-framework/sofa/pull/923)
- [SofaMeshCollision] Fix TriangleModel to handle topology changes [#903](https://github.com/sofa-framework/sofa/pull/903)
- **[SofaSimulationCore]** Remove unjustified Assert in getSimulation() [#1082](https://github.com/sofa-framework/sofa/pull/1082)
- **[SofaSimulationCore]** FIX CollisionVisitor::processCollisionPipeline [#962](https://github.com/sofa-framework/sofa/pull/962)
- [SofaTests] Fix small bugs in the Multi2Mapping_test [#1078](https://github.com/sofa-framework/sofa/pull/1078)

**Plugins / Projects**
- [CImgPlugin] FIX: messed up package prefix in CImg [#921](https://github.com/sofa-framework/sofa/pull/921)
- [Geomagic] FIX compilation error in Geomagic plugin with removal of SOFA_FLOAT/DOUBLE [#898](https://github.com/sofa-framework/sofa/pull/898)
- [image] Fix image_gui plugin loading [#1015](https://github.com/sofa-framework/sofa/pull/1015)
- [image] Message API is needed even if no python [#1068](https://github.com/sofa-framework/sofa/pull/1068)
- [runSofa] FIX the opening of ModifyObject view. [#1010](https://github.com/sofa-framework/sofa/pull/1010)
- [runSofa] Fix runSofa -a option with a gui. [#1058](https://github.com/sofa-framework/sofa/pull/1058)
- [runSofa] User experience fixes in the ModifyData view. [#1011](https://github.com/sofa-framework/sofa/pull/1011)
- [Sensable] Fix the compilation of the Sensable plugin [#1019](https://github.com/sofa-framework/sofa/pull/1019)
- [SofaCUDA] Compilation error fix (CudaStandardTetrahedralFEMForceField.cu) [#991](https://github.com/sofa-framework/sofa/pull/991)
- [SofaCUDA] Fix several Cuda example scenes [#1000](https://github.com/sofa-framework/sofa/pull/1000)
- [SofaCUDA] Fix windows compilation. [#966](https://github.com/sofa-framework/sofa/pull/966)
- [SofaPython] FIX allow the derivTypeFromParentValue to work with node. [#984](https://github.com/sofa-framework/sofa/pull/984)
- [SofaPython] FIX example broken by PR#459 [#1020](https://github.com/sofa-framework/sofa/pull/1020)
- [SofaPython] FIX the broken Binding_Data::setValue()  [#1006](https://github.com/sofa-framework/sofa/pull/1006)
- [SofaPython] Fix duplicate symbol [#1036](https://github.com/sofa-framework/sofa/pull/1036)
- [SofaPython] FIX: removing PythonLibs target from SofaPython [#891](https://github.com/sofa-framework/sofa/pull/891)
- [SofaPython] REMOVE: public export of target SofaExporter [#963](https://github.com/sofa-framework/sofa/pull/963)

**Examples / Scenes**
- [examples] Remove warnings in Demos/ scenes [#1021](https://github.com/sofa-framework/sofa/pull/1021)
- [scenes] Fix chainAll demo scenario [#987](https://github.com/sofa-framework/sofa/pull/987)


### Cleanings

**Modules**
- [All] For each data field's with a "filename" alias flip it with the data's name.  [#1024](https://github.com/sofa-framework/sofa/pull/1024)
- [All] Quick changes diffusion and mass [#983](https://github.com/sofa-framework/sofa/pull/983)
- [All] Remove duplicate ctor + prettify some code [#1054](https://github.com/sofa-framework/sofa/pull/1054)
- [All] Replace serr with the new msg_error() API. [#916](https://github.com/sofa-framework/sofa/pull/916)
- [All] Several STC fixes [#1048](https://github.com/sofa-framework/sofa/pull/1048)
- [All] Sofa defrost sprint week2 [#884](https://github.com/sofa-framework/sofa/pull/884)
- [All] minor cleaning of warnings and bugfix [#886](https://github.com/sofa-framework/sofa/pull/886)
- [All] Remove bunch of warnings (again) [#1065](https://github.com/sofa-framework/sofa/pull/1065)
- [All] remove #ifdef SOFA_HAVE_GLEW [#1077](https://github.com/sofa-framework/sofa/pull/1077)
- **[SofaLoader]** Change error into warning in MeshVTKLoader [#1037](https://github.com/sofa-framework/sofa/pull/1037)
- [SofaConstraint] Replaced sout calls by msg_info() in LCPConstraintSolver [#981](https://github.com/sofa-framework/sofa/pull/981)
- [SofaGeneralLinearSolver] Clean BTDLinearSolver [#907](https://github.com/sofa-framework/sofa/pull/907)
- [SofaHaptics] Replace deprecated INCLUDE_ROOT_DIR in CMakeLists.txt [#1023](https://github.com/sofa-framework/sofa/pull/1023)
- **[SofaKernel]** Brainless Warnings cleaning [#971](https://github.com/sofa-framework/sofa/pull/971)
- **[SofaKernel]** Minor code refactor in BaseData & new StringUtils functions. [#860](https://github.com/sofa-framework/sofa/pull/860)
- **[SofaKernel]** Refactor DataTrackerEngine so it match the DataCallback [#1073](https://github.com/sofa-framework/sofa/pull/1073)
- **[SofaKernel]** Remove annoying warning [#1062](https://github.com/sofa-framework/sofa/pull/1062)
- **[SofaKernel]** Remove boost::locale dependency [#1033](https://github.com/sofa-framework/sofa/pull/1033)
- **[SofaKernel]** Remove usage of helper::system::atomic<int> (replaced by STL's) [#1035](https://github.com/sofa-framework/sofa/pull/1035)
- **[SofaKernel]** Several changes in Topology components [#999](https://github.com/sofa-framework/sofa/pull/999)
- **[SofaKernel]** minor cleaning in mesh loader [#1025](https://github.com/sofa-framework/sofa/pull/1025)
- **[SofaKernel]** Remove multigroup option in MatrixLinearSolver [#901](https://github.com/sofa-framework/sofa/pull/901)
- [SofaRigid] Clean JointSpringFF [#850](https://github.com/sofa-framework/sofa/pull/850)
- [SofaRigid] Cosmetic clean in RigidRigidMapping & msg_* update. [#1005](https://github.com/sofa-framework/sofa/pull/1005)
- [SofaSimpleFem] Use msg and size_t in TetraDiff [#1016](https://github.com/sofa-framework/sofa/pull/1016)

**Plugins / Projects**
- [image] Add warning guiding users regarding pluginization of DiffusionSolver [#1067](https://github.com/sofa-framework/sofa/pull/1067)
- [Modeler] Deactivate Modeler by default, since it is deprecated [#972](https://github.com/sofa-framework/sofa/pull/972)

**Examples / Scenes**
- [Scenes] Apply script on all scenes using VisualModel/OglModel [#1081](https://github.com/sofa-framework/sofa/pull/1081)


____________________________________________________________



## [v18.12](https://github.com/sofa-framework/sofa/tree/v18.12)

[Full log](https://github.com/sofa-framework/sofa/compare/v18.06...v18.12)


### Deprecated

**Removed in v18.12**
- [SofaBoundaryCondition] BuoyantForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaBoundaryCondition] VaccumSphereForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
- **[SofaHelper]** Utils::getPluginDirectory() [#518](https://github.com/sofa-framework/sofa/pull/518) - Use PluginRepository.getFirstPath() instead
- [SofaMisc] ParallelCGLinearSolver [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscForceField] ForceMaskOff [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscForceField] LineBendingSprings [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscForceField] WashingMachineForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
- ~~[SofaMiscForceField] LennardJonesForceField [#457](https://github.com/sofa-framework/sofa/pull/457)~~
- [SofaMiscMapping] CatmullRomSplineMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] CenterPointMechanicalMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] CurveMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] ExternalInterpolationMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] ProjectionToLineMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] ProjectionToPlaneMapping
- ~~[SofaOpenglVisual] OglCylinderModel [#457](https://github.com/sofa-framework/sofa/pull/457)~~
- ~~[SofaOpenglVisual] OglGrid [#457](https://github.com/sofa-framework/sofa/pull/457)~~
- ~~[SofaOpenglVisual] OglRenderingSRGB [#457](https://github.com/sofa-framework/sofa/pull/457)~~
- ~~[SofaOpenglVisual] OglLineAxis [#457](https://github.com/sofa-framework/sofa/pull/457)~~
- ~~[SofaOpenglVisual] OglSceneFrame [#457](https://github.com/sofa-framework/sofa/pull/457)~~
- [SofaUserInteraction] ArticulatedHierarchyBVHController [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] ArticulatedHierarchyController [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] DisabledContact [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] EdgeSetController [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] GraspingManager [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] InterpolationController [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] MechanicalStateControllerOmni [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] NodeToggleController [#457](https://github.com/sofa-framework/sofa/pull/457)


### Breaking

**Modules**
- **[SofaBaseMechanics]**[SofaMiscForceField] Homogeneization of mass components in SOFA [#637](https://github.com/sofa-framework/sofa/pull/637)
- **[SofaBaseMechanics]** Clean barycentric mapping [#797](https://github.com/sofa-framework/sofa/pull/797)
- [SofaBoundaryCondition] Refactor FixedPlaneConstraint (breaking)  [#803](https://github.com/sofa-framework/sofa/pull/803)
- **[SofaFramework]** [BREAKING] Replacing DataEngine with SimpleDataEngine [#814](https://github.com/sofa-framework/sofa/pull/814)
- **[SofaFramework]** [BREAKING] Rename: data tracker has changed [#822](https://github.com/sofa-framework/sofa/pull/822)
- [SofaPreconditioner] modularization [#668](https://github.com/sofa-framework/sofa/pull/668)
- [SofaSparseSolver] modularization [#668](https://github.com/sofa-framework/sofa/pull/668)


### Improvements

**Architecture**
- [CMake] use ccache if available [#692](https://github.com/sofa-framework/sofa/pull/692)
- [Cmake] Add a findCython.cmake [#734](https://github.com/sofa-framework/sofa/pull/734)
- [CMake] ADD QtIFW generator + improvements [#796](https://github.com/sofa-framework/sofa/pull/796)
- [SofaMacros] ADD CMake macro to create pybind11 & cython targets and modules #859(https://github.com/sofa-framework/sofa/pull/859)

**Modules**
- [All] Use drawtool everywhere [#704](https://github.com/sofa-framework/sofa/pull/704)
- [All] Sofa add mechanical matrix mapper [#721](https://github.com/sofa-framework/sofa/pull/721)
- **[SofaBaseTopology]** Add battery of tests on topology containers [#708](https://github.com/sofa-framework/sofa/pull/708)
- **[SofaBaseTopology]** Topology change propagation to Mechanical State [#838](https://github.com/sofa-framework/sofa/pull/838)
- **[SofaBaseMechanics]** Optimize barycentric mapping initialization [#798](https://github.com/sofa-framework/sofa/pull/798)
- [SofaBoundaryCondition] Factorize partial fixedconstraint [#718](https://github.com/sofa-framework/sofa/pull/718)
- [SofaConstraint] add a force data field in SlidingConstraint [#780](https://github.com/sofa-framework/sofa/pull/780)
- [SofaConstraint] ADD Data to show constraint forces [#840](https://github.com/sofa-framework/sofa/pull/840)
- [SofaConstraint] allow call of constraints' storeLambda() [#854](https://github.com/sofa-framework/sofa/pull/854)
- **[SofaCore]** Add new (simpler) DataEngine implementation [#760](https://github.com/sofa-framework/sofa/pull/760)
- [SofaExporter] ADD in WriteState all required tests on data and clean export with msg API [#714](https://github.com/sofa-framework/sofa/pull/714)
- **[SofaFramework]** Improve external dirs fetching in SofaMacros [#759](https://github.com/sofa-framework/sofa/pull/759)
- [SofaGeneralAnimationLoop] Improvement on MMMapper [#772](https://github.com/sofa-framework/sofa/pull/772)
- **[SofaHelper]** EDIT FileSystem and FileRepository for regression tests [#830](https://github.com/sofa-framework/sofa/pull/830)
- **[SofaKernel]** Improve Displayflags [#671](https://github.com/sofa-framework/sofa/pull/671)
- **[SofaKernel]** Add a "sofa_add_module" in SofaMacro.cmake [#732](https://github.com/sofa-framework/sofa/pull/732)
- **[SofaKernel]** use string in base object description [#862](https://github.com/sofa-framework/sofa/pull/862)
- [SofaMeshCollision] TriangleModel optimization when topology changes occur [#839](https://github.com/sofa-framework/sofa/pull/839)
- [SofaSparseSolver] ADD saveMatrixToFile to SparseLDLSolver [#845](https://github.com/sofa-framework/sofa/pull/845)
- [SofaTest] ADD a PrintTo method so test failure shows human readable informations. [#730](https://github.com/sofa-framework/sofa/pull/730)
- [VisualModel] Improve the messages when loading mesh inside VisualModel [#778](https://github.com/sofa-framework/sofa/pull/778)
- [WriteState] minor fix with the time attribute, default values [#776](https://github.com/sofa-framework/sofa/pull/776)

**Plugins / Projects**
- [Geomagic] ADD an inputForceFeedback data in Geomagic [#648](https://github.com/sofa-framework/sofa/pull/648)
- [Geomagic] Fix dll export and some enhancements [#786](https://github.com/sofa-framework/sofa/pull/786)
- [MultiThreading] removed the boost thread dependency [#701](https://github.com/sofa-framework/sofa/pull/701)
- [MultiThreading] New components and Task scheduler classes refactoring  [#745](https://github.com/sofa-framework/sofa/pull/745)
- [MultiThreading] Add Image plugin Data types in DataExchange component [#770](https://github.com/sofa-framework/sofa/pull/770)
- [MultiThreading] TaskScheduler Interface [#775](https://github.com/sofa-framework/sofa/pull/775)
- [runSofa] Add data field value change on mouse move [#750](https://github.com/sofa-framework/sofa/pull/750)
- [SofaCarving] Refresh and enhancement [#712](https://github.com/sofa-framework/sofa/pull/712)
- [SofaCarving] plugin enhancement [#787](https://github.com/sofa-framework/sofa/pull/787)
- [SofaPython] ADD forwarding of onMouseMove event into the script controller [#731](https://github.com/sofa-framework/sofa/pull/731)
- [SofaPython] ADD: Bindings for BoundingBox [#736](https://github.com/sofa-framework/sofa/pull/736)
- [SofaPython][PSDE] Psde derive io [#742](https://github.com/sofa-framework/sofa/pull/742)
- [SofaPython][PSDE] Update on demand as designed initially [#751](https://github.com/sofa-framework/sofa/pull/751)
- [SofaPython] ADD a custom __dir__ method in Binding_Base. [#762](https://github.com/sofa-framework/sofa/pull/762)
- [SofaPython] add getLinkedBase to the binding of a link. [#843](https://github.com/sofa-framework/sofa/pull/843)
- [SofaPython] ADD binding python to getCategories [#868](https://github.com/sofa-framework/sofa/pull/868)


### Bug Fixes

**Architecture**
- [CMake] FIX: cyclic recursion [#766](https://github.com/sofa-framework/sofa/pull/766)
- [CMake] Backport fixes [#791](https://github.com/sofa-framework/sofa/pull/791)
- [CMake] Fix compilation issues due to CPackNSIS [#867](https://github.com/sofa-framework/sofa/pull/867)
- [CMake] Add check to prevent the inclusion of non-existant file in cmake 3.13 [#897](https://github.com/sofa-framework/sofa/pull/897)

**Modules**
- [All] ISSofa bugfix, lot of fixes [#756](https://github.com/sofa-framework/sofa/pull/756)
- [All] FIX Windows linkage [#910](https://github.com/sofa-framework/sofa/pull/910)
- [SofaGuiQt] Change method to allow antialiased screenshots in QtViewer [#728](https://github.com/sofa-framework/sofa/pull/728)
- **[SofaBaseMechanics]** Fix warning scene mass [#779](https://github.com/sofa-framework/sofa/pull/779)
- **[SofaBaseMechanics]** FIX DiagonalMass_test [#832](https://github.com/sofa-framework/sofa/pull/832)
- **[SofaBaseTopology]** Fix SparseGridTopology initialization with an input mesh [#670](https://github.com/sofa-framework/sofa/pull/670)
- [SofaBoundaryCondition] FIX AffineMovementConstraint orientation issue [#824](https://github.com/sofa-framework/sofa/pull/824)
- [SofaCarving] Modify the CMake config file to allow other plugins link to Sofa Carving  [#781](https://github.com/sofa-framework/sofa/pull/781)
- **[SofaCore]** FIX: enable ExtVecXf mappings with double floating type [#827](https://github.com/sofa-framework/sofa/pull/827)
- [SofaDeformable] Fix MeshSpring ForceField and Loader [#815](https://github.com/sofa-framework/sofa/pull/815)
- **[SofaFramework]** Keep SOFA_EXTERN_TEMPLATE macro definition [#870](https://github.com/sofa-framework/sofa/pull/870)
- [SofaGui] ADD option to enable VSync (default: OFF) [#722](https://github.com/sofa-framework/sofa/pull/722)
- [SofaOpenglVisual] Rollback removal of Ogl components [#905](https://github.com/sofa-framework/sofa/pull/905)
- **[SofaKernel]** FIX bug in toEulerVector [#399](https://github.com/sofa-framework/sofa/pull/399)
- **[SofaKernel]** FIX segfault created by static initialisers on OSX/clang compiler [#642](https://github.com/sofa-framework/sofa/pull/642)
- **[SofaKernel]** Fix: correct path writing in sofa_set_python_directory macro [#763](https://github.com/sofa-framework/sofa/pull/763)
- **[SofaKernel]** Check if Quaternion has norm 1 [#790](https://github.com/sofa-framework/sofa/pull/790)
- [SofaPreconditioner] FIX ShewchukPCGLinearSolver [#737](https://github.com/sofa-framework/sofa/pull/737)

**Plugins / Projects**
- [CGALPlugin] fix compilation [#783](https://github.com/sofa-framework/sofa/pull/783)
- [CGALPlugin] Fix compilation for cgal 4.12+ (cgal::polyhedron_3 is now forward declared) [#812](https://github.com/sofa-framework/sofa/pull/812)
- [CImgPlugin] CMake/Mac: lower priority for XCode's libpng [#720](https://github.com/sofa-framework/sofa/pull/720)
- [Geomagic] Fix broken behavior [#761](https://github.com/sofa-framework/sofa/pull/761)
- [Geomagic] Fix scenes [#858](https://github.com/sofa-framework/sofa/pull/858)
- [Multithreading] FIX compiling error on Mac [#727](https://github.com/sofa-framework/sofa/pull/727)
- [MultiThreading] Fix Livers scenes crash  [#792](https://github.com/sofa-framework/sofa/pull/792)
- [runSofa] ADD Modules to plugin_list.conf.default [#753](https://github.com/sofa-framework/sofa/pull/753)
- [SofaPython][examples] FIX: Fixing the scene... and the typo in the name [#765](https://github.com/sofa-framework/sofa/pull/765)
- [Tutorials] FIX oneTetrahedron and chainHybrid [#773](https://github.com/sofa-framework/sofa/pull/773)

**Examples / Scenes**
- [examples] Fix TopologySurfaceDifferentMesh.scn [#716](https://github.com/sofa-framework/sofa/pull/716)
- [examples] Fix the examples  missing a <RequiredPlugin name="SofaSparseSolver"/> [#748](https://github.com/sofa-framework/sofa/pull/748)
- [examples] Fix scenes having issue with CollisionGroup [#821](https://github.com/sofa-framework/sofa/pull/821)


### Cleanings

**Modules**
- [All] Fix some recent compilation warnings [#726](https://github.com/sofa-framework/sofa/pull/726)
- [All] Replace some int/unit by size_t [#729](https://github.com/sofa-framework/sofa/pull/729)
- [All] Fix some C4661 warnings [#738](https://github.com/sofa-framework/sofa/pull/738)
- [All] FIX warnings [#739](https://github.com/sofa-framework/sofa/pull/739)
- [All] Fix some compilation warnings [#755](https://github.com/sofa-framework/sofa/pull/755)
- [All] FIX a bunch of static-analysis errors (cppcheck) [#782](https://github.com/sofa-framework/sofa/pull/782)
- [All] Remove SOFA_DECL_CLASS and SOFA_LINK_CLASS [#837](https://github.com/sofa-framework/sofa/pull/837)
- [All] Remove SOFA_SUPPORT_MOVING_FRAMES and SOFA_SUPPORT_MULTIRESOLUTION [#849](https://github.com/sofa-framework/sofa/pull/849)
- **[SofaBaseCollision]** CLEAN CylinderModel [#847](https://github.com/sofa-framework/sofa/pull/847)
- **[SofaBaseLinearSolver]** CLEAN GraphScatteredTypes [#844](https://github.com/sofa-framework/sofa/pull/844)
- **[SofaFramework]** Various cleaning (non-breaking) [#841](https://github.com/sofa-framework/sofa/pull/841)
- **[SofaFramework]** CLEAN: removing unused PS3 files [#851](https://github.com/sofa-framework/sofa/pull/851)
- [SofaGeneralSimpleFEM] Clean BeamFemForceField [#846](https://github.com/sofa-framework/sofa/pull/846)
- **[SofaHelper]** Change drawTriangle and drawQuad with internal functions [#813](https://github.com/sofa-framework/sofa/pull/813)
- **[SofaHelper]** Update ComponentChange with removed Components [#905](https://github.com/sofa-framework/sofa/pull/905)
- **[SofaKernel]** Remove commented code since years in SofaBaseMechanics [#733](https://github.com/sofa-framework/sofa/pull/733)
- **[SofaKernel]** Move ScriptEvent class from SofaPython to core/objectModel [#764](https://github.com/sofa-framework/sofa/pull/764)
- [SofaMiscFem] Clean BaseMaterial::handleTopologyChange [#817](https://github.com/sofa-framework/sofa/pull/817)
- [SofaMiscMapping] Clean DeformableOnRigidFrameMapping [#848](https://github.com/sofa-framework/sofa/pull/848)
- [SofaSimpleFem] Stc clean simplefem [#826](https://github.com/sofa-framework/sofa/pull/826)

**Plugins / Projects**
- [Multithreading] Move TaskScheduler files from MultiThreading plugin to SofaKernel [#805](https://github.com/sofa-framework/sofa/pull/805)

**Examples / Scenes**
- [examples] Remove scenes about deprecated components [#922](https://github.com/sofa-framework/sofa/pull/922)


____________________________________________________________



## [v18.06](https://github.com/sofa-framework/sofa/tree/v18.06)

[Full log](https://github.com/sofa-framework/sofa/compare/v17.12...v18.06)


### Deprecated

**Will be removed in v18.06**
- **[SofaHelper]** Utils::getPluginDirectory() [#518](https://github.com/sofa-framework/sofa/pull/518) - Use PluginRepository.getFirstPath() instead

**Will be removed in v18.12**
- [SofaBoundaryCondition] BuoyantForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaBoundaryCondition] VaccumSphereForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMisc] ParallelCGLinearSolver [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscForceField] ForceMaskOff [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscForceField] LineBendingSprings [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscForceField] WashingMachineForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscForceField] LennardJonesForceField [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] CatmullRomSplineMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] CenterPointMechanicalMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] CurveMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] ExternalInterpolationMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] ProjectionToLineMapping [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaMiscMapping] ProjectionToPlaneMapping
- [SofaOpenglVisual] OglCylinderModel [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaOpenglVisual] OglGrid [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaOpenglVisual] OglRenderingSRGB [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaOpenglVisual] OglLineAxis [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaOpenglVisual] OglSceneFrame
- [SofaUserInteraction] ArticulatedHierarchyBVHController [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] ArticulatedHierarchyController [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] DisabledContact [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] EdgeSetController [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] GraspingManager [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] InterpolationController [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] MechanicalStateControllerOmni [#457](https://github.com/sofa-framework/sofa/pull/457)
- [SofaUserInteraction] NodeToggleController [#457](https://github.com/sofa-framework/sofa/pull/457)


### Breaking

**Modules**
- [SofaConstraint] Update FreeMotionAnimationLoop so that it can compute a linearised version of the constraint force. [#459](https://github.com/sofa-framework/sofa/pull/459)
- **[SofaCore]** Update FreeMotionAnimationLoop so that it can compute a linearised version of the constraint force. [#459](https://github.com/sofa-framework/sofa/pull/459)
- **[SofaHelper]** Unifying the way we report file related errors [#669](https://github.com/sofa-framework/sofa/pull/669)


### Improvements

**Architecture**
- [CMake] ADD external projects handling [#649](https://github.com/sofa-framework/sofa/pull/649)
- [CMake] ADD the CMAKE_WARN_DEPRECATED option in SOFA [#662](https://github.com/sofa-framework/sofa/pull/662)
- [CMake] Improve SOFA installation and packaging [#635](https://github.com/sofa-framework/sofa/pull/635)
- [CMake] Cleans for packaging [#789](https://github.com/sofa-framework/sofa/pull/789)

**Modules**
- [All] Refactoring in Camera, BackgroundSetting and Light [#676](https://github.com/sofa-framework/sofa/pull/676)
- **[SofaBaseLinearSolver]** Improve warning emission for CG [#658](https://github.com/sofa-framework/sofa/pull/658)
- **[SofaBaseLinearSolver]** Add ability to activate printing of debug information at runtime [#667](https://github.com/sofa-framework/sofa/pull/667)
- [SofaGeneralImplicitOdeSolver] FIX data field name in VariationalSymplecticSolver [#624](https://github.com/sofa-framework/sofa/pull/624)
- [SofaGraphComponent] ADD alias usage detection [#702](https://github.com/sofa-framework/sofa/pull/702)
- **[SofaLoader]** ADD support to load VTK polylines in legacy formated files [#576](https://github.com/sofa-framework/sofa/pull/576)
- [SofaMiscMapping] Fix rigid barycentric mapping [#710](https://github.com/sofa-framework/sofa/pull/710)
- **[SofaHelper]** PluginManager now checks for file existence instead of library extension match. [#621](https://github.com/sofa-framework/sofa/pull/621)

**Plugins / Projects**
- [HeadlessRecorder] ADD frameskip option to headless recorder [#615](https://github.com/sofa-framework/sofa/pull/615)
- [HeadlessRecorder] Remove avcodec dependency in HeadlessRecorder.h [#752](https://github.com/sofa-framework/sofa/pull/752)
- [runSofa] Save&restore the scenegraph state when live-code & add info panel [#657](https://github.com/sofa-framework/sofa/pull/657)
- [SofaPython] PythonScriptDataEngine (PSDE) [#583](https://github.com/sofa-framework/sofa/pull/583)
- [SofaPython] Small fix & new features. [#656](https://github.com/sofa-framework/sofa/pull/656)

**Tools**
- [tools] FIX sofa-launcher stdout [#592](https://github.com/sofa-framework/sofa/pull/592)


### Bug Fixes

**Modules**
- [All] FIX VS2017 build (Windows) [#630](https://github.com/sofa-framework/sofa/pull/630)
- [All] Fix computeBBox() [#634](https://github.com/sofa-framework/sofa/pull/634)
- [All] FIX warnings [#584](https://github.com/sofa-framework/sofa/pull/584)
- [All] Various small changes in error messages & bugfix  from defrost branches [#660](https://github.com/sofa-framework/sofa/pull/660)
- [SofaConstraint] FIX: Moving semicolon under preprocessor define [#680](https://github.com/sofa-framework/sofa/pull/680)
- **[SofaEngine]** FIX Bug in BoxROI that is not properly initialized [#627](https://github.com/sofa-framework/sofa/pull/627)
- **[SofaFramework]** Fix plugin list configuration [#645](https://github.com/sofa-framework/sofa/pull/645)
- [SofaGraphComponent] FIX SceneChecker_test + ADD alias test [#711](https://github.com/sofa-framework/sofa/pull/711)
- [SofaGraphComponent] FIX SceneCheck build on MacOS [#719](https://github.com/sofa-framework/sofa/pull/719)
- [SofaGuiQt] FIX missing resources [#758](https://github.com/sofa-framework/sofa/pull/758)
- [SofaGeneralEngine] FIX disabled tests [#675](https://github.com/sofa-framework/sofa/pull/675)
- **[SofaHelper]** More robust method to test end of string [#617](https://github.com/sofa-framework/sofa/pull/617)
- **[SofaKernel]** FIX macro issue resulted from the #include cleaning. [#672](https://github.com/sofa-framework/sofa/pull/672)
- [SofaMiscFem] FIX dependencies [#588](https://github.com/sofa-framework/sofa/pull/588)
- [SofaOpenglVisual] FIX MacOS crash in batch mode [#646](https://github.com/sofa-framework/sofa/pull/646)
- **[SofaSimulationGraph]** FIX dependencies [#588](https://github.com/sofa-framework/sofa/pull/588)
- [SofaSparseSolver] FIX SparseLDL crash and add proper SOFA_FLOAT/DOUBLE mangement [#655](https://github.com/sofa-framework/sofa/pull/655)

**Plugins / Projects**
- [CGALPlugin] FIX compilation issue with recent version of CGAL (4.11) & Ubunu 18.04 LTS [#664](https://github.com/sofa-framework/sofa/pull/664)
- [CImgPlugin] Export CImg_CFLAGS [#595](https://github.com/sofa-framework/sofa/pull/595)
- [CImgPlugin] FIX CMakeLists install fail since pluginization [#609](https://github.com/sofa-framework/sofa/pull/609)
- [CImgPlugin] FIX malformed cflag append [#622](https://github.com/sofa-framework/sofa/pull/622)
- [HeadlessRecorder] Fix headless recorder stream definition [#666](https://github.com/sofa-framework/sofa/pull/666)
- [MultiThreading] FIX: add createSubelements param in MeshGmshLoader [#626](https://github.com/sofa-framework/sofa/pull/626)
- [runSofa] Fix compilation when SofaGuiQt is not activated [#599](https://github.com/sofa-framework/sofa/pull/599)
- [runSofa] ADD infinite iterations option to batch gui [#613](https://github.com/sofa-framework/sofa/pull/613)
- [runSofa] FIX missing resources [#758](https://github.com/sofa-framework/sofa/pull/758)
- [SofaDistanceGrid] ADD .scene-tests to ignore scene [#594](https://github.com/sofa-framework/sofa/pull/594)
- [SofaPython] FIX build for MacOS >10.13.0 [#614](https://github.com/sofa-framework/sofa/pull/614)

**Examples / Scenes**
- FIX collision of the fontain example [#612](https://github.com/sofa-framework/sofa/pull/612)
- FIX failing scenes on CI [#641](https://github.com/sofa-framework/sofa/pull/641)
- FIX missing RequiredPlugin [#628](https://github.com/sofa-framework/sofa/pull/628)

**Extlibs**
- [extlibs/gtest] Update gtest & clean the CMakeLists.txt [#604](https://github.com/sofa-framework/sofa/pull/604)


### Cleanings

**Architecture**
- [CMake] Remove the option SOFA_GUI_INTERACTION and its associated codes/macro [#643](https://github.com/sofa-framework/sofa/pull/643)

**Modules**
- [All] CMake: Remove COMPONENTSET, keep DEPRECATED [#586](https://github.com/sofa-framework/sofa/pull/586)
- [All] CLEAN topology classes [#693](https://github.com/sofa-framework/sofa/pull/693)
- **[SofaHelper]** CLEAN commented code and double parentheses in Messaging.h [#587](https://github.com/sofa-framework/sofa/pull/587)
- **[SofaKernel]** Header include cleanup [#638](https://github.com/sofa-framework/sofa/pull/638)
- **[SofaKernel]** Remove unused function "renumberConstraintId" [#691](https://github.com/sofa-framework/sofa/pull/691)

**Plugins / Projects**
- [CImgPlugin] Less scary config warnings [#607](https://github.com/sofa-framework/sofa/pull/607)
- [HeadlessRecorder] Handle errors in target config [#608](https://github.com/sofa-framework/sofa/pull/608)
- [SofaGUI] Move GlutGUI to projects and remove all glut references in SofaFramework [#598](https://github.com/sofa-framework/sofa/pull/598)
- [SofaGUI] CMake: Remove useless if block in qt CMakelists.txt [#590](https://github.com/sofa-framework/sofa/pull/590)
- [SofaPhysicsAPI] FIX: remove the include of glut [#659](https://github.com/sofa-framework/sofa/pull/659)


____________________________________________________________



## [v17.12](https://github.com/sofa-framework/sofa/tree/v17.12)

[Full log](https://github.com/sofa-framework/sofa/compare/v17.06...v17.12)


### Deprecated

**Kernel modules**
- Will be removed in v17.12
    - [All]
        - SMP support [#457](https://github.com/sofa-framework/sofa/pull/457 - no more maintained)
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
- [All]
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
- [All]
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
- [All]
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

**Plugins / Projects**
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
- [All]
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
- [All]
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

**Plugins / Projects**
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


### Cleanings

**Kernel modules**
- [All]
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

**Plugins / Projects**
- [All]
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
- [All]
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

**Plugins / Projects**
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
- [All]
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

**Plugins / Projects**
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


### Cleanings

**Modules**
- [All]
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

**Plugins / Projects**
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


### Cleanings

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

### Cleanings

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
