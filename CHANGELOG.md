# SOFA Changelog


## [v24.06.00]( https://github.com/sofa-framework/sofa/tree/v24.06.00 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v23.12..v24.06 )

### Highlighted contributions

- [GitHub] Add a python-based action managing discussions [#4268](https://github.com/sofa-framework/sofa/pull/4268)
- [plugins] Update license to LGPL of remaining files [#4425](https://github.com/sofa-framework/sofa/pull/4425)
- [LinearSystem] Introduce components to map matrices [#4490](https://github.com/sofa-framework/sofa/pull/4490)
- [SofaSphFluid] Externalize SofaSphFluid [#4526](https://github.com/sofa-framework/sofa/pull/4526)
- [LinearSystem] Introduce ConstantSparsityProjectionMethod [#4552](https://github.com/sofa-framework/sofa/pull/4552)
- [HyperElastic] Introduce stable Neo Hookean hyperelastic material [#4603](https://github.com/sofa-framework/sofa/pull/4603)
- [LinearSystem] Cache mapped mass matrix [#4625](https://github.com/sofa-framework/sofa/pull/4625)
- [Constraint.Lagrangian] Add fixed lagrangian constraint [#4646](https://github.com/sofa-framework/sofa/pull/4646)


### Breaking

- [Type] RGBAColor: remove inheritance from type::fixed_array and use std::array to store its components [#4263](https://github.com/sofa-framework/sofa/pull/4263)
- [All] ColorMap uses RGBAColor [#4270](https://github.com/sofa-framework/sofa/pull/4270)
- [Type] Refactor Vec [#4296](https://github.com/sofa-framework/sofa/pull/4296)
- [SolidMechanics.Springs] Cleaning of QuadularBendingSprings [#4304](https://github.com/sofa-framework/sofa/pull/4304)
- [Topology.Container.Dynamic] GeometryAlgorithms: support generic State [#4362](https://github.com/sofa-framework/sofa/pull/4362)
- [LinearSolver.Direct] Introduce other ordering methods in SparseLDL through Eigen [#4370](https://github.com/sofa-framework/sofa/pull/4370)
- [Constraint.Lagrangian.Solver] Fix assert in GenericConstraintSolver [#4389](https://github.com/sofa-framework/sofa/pull/4389)
- [Type] Refactor Mat [#4396](https://github.com/sofa-framework/sofa/pull/4396)
- [LinearSystem] Detect changes in sparsity pattern when using ConstantSparsityPatternSystem [#4428](https://github.com/sofa-framework/sofa/pull/4428)
- [Core][LinearSolver] Introduction of ordering method components [#4477](https://github.com/sofa-framework/sofa/pull/4477)
- [Simulation.Graph] Move SimpleApi into its own library [#4495](https://github.com/sofa-framework/sofa/pull/4495)
- [all] Lifecycle for v24.06 (2): remove Compat [#4533](https://github.com/sofa-framework/sofa/pull/4533)
- [all] Add PenalityContact vector Data display [#4637](https://github.com/sofa-framework/sofa/pull/4637)
- [GUI.Component] Add virtualization for attachment performer [#4638](https://github.com/sofa-framework/sofa/pull/4638)
- [GUI.Qt] Some cleaning in qt RealGui [#4641](https://github.com/sofa-framework/sofa/pull/4641)
- [FEM.HyperElastic] Convert string Data to OptionsGroup [#4651](https://github.com/sofa-framework/sofa/pull/4651)
- [AnimationLoop] Remove duplicated Constraint Visitors  [#4715](https://github.com/sofa-framework/sofa/pull/4715)


### Improvements

- [Lagrangian.Correction] Add callbacks to check zero compliance  [#4205](https://github.com/sofa-framework/sofa/pull/4205)
- [SolidMechanics.Spring] Implement buildStiffnessMatrix for TriangularBendingSprings [#4295](https://github.com/sofa-framework/sofa/pull/4295)
- [SolidMechanics] Implement buildStiffnessMatrix for PolynomialSpringsFF [#4301](https://github.com/sofa-framework/sofa/pull/4301)
- [Config] CMake: support interface-type target for install macro [#4356](https://github.com/sofa-framework/sofa/pull/4356)
- [Core] BaseMapping: link mapConstraints to the "meta-alias" isMechanical [#4360](https://github.com/sofa-framework/sofa/pull/4360)
- [github] Add new action checking PR age [#4386](https://github.com/sofa-framework/sofa/pull/4386)
- [Topology.Mapping] Edge2QuadTopologicalMapping: use States directly [#4388](https://github.com/sofa-framework/sofa/pull/4388)
- [README] Minor update badges [#4417](https://github.com/sofa-framework/sofa/pull/4417)
- [StateContainer] Accelerate copy of MatrixDeriv for CRS matrices [#4443](https://github.com/sofa-framework/sofa/pull/4443)
- [GitHub] Add new actions to connect GitHub and Discord [#4445](https://github.com/sofa-framework/sofa/pull/4445)
- [All] Display reference name when pulling external plugins [#4507](https://github.com/sofa-framework/sofa/pull/4507)
- [Config] Add TSAN option [#4534](https://github.com/sofa-framework/sofa/pull/4534)
- [Project] Start dev phase v24.06 [#4544](https://github.com/sofa-framework/sofa/pull/4544)
- [runsofa] Allowing multiple argv [#4591](https://github.com/sofa-framework/sofa/pull/4591)
- [GUI] Add ConstraintAttachButtonSetting [#4601](https://github.com/sofa-framework/sofa/pull/4601)
- [plugins] Add fetchable ModelOrderReduction [#4608](https://github.com/sofa-framework/sofa/pull/4608)
- [GUI.Qt] Add clickable link to online documentation [#4650](https://github.com/sofa-framework/sofa/pull/4650)
- [CMake] Start finding Qt6 then Qt5 [#4686](https://github.com/sofa-framework/sofa/pull/4686)
- [examples] Use MeshMatrixMass in hyperelastic examples [#4687](https://github.com/sofa-framework/sofa/pull/4687)


### Bug Fixes

- [Geometry] Update method intersectionWithEdge signature and redirect all methods to it in EdgeSetGeometryAlgorithms [#4194](https://github.com/sofa-framework/sofa/pull/4194)
- [SofaSPHFluid] Add option in ParticleSource to add/remove random values in the particles generation. Remove noise for CI scenes [#4316](https://github.com/sofa-framework/sofa/pull/4316)
- [ODESolver] Fix compilation with SOFA_NO_VMULTIOP  [#4325](https://github.com/sofa-framework/sofa/pull/4325)
- [Topology.Mapping] Edge2QuadTopologicalMapping: use Links for required QuadSet components [#4361](https://github.com/sofa-framework/sofa/pull/4361)
- [MultiThreading] Fix failing test on ParallelImplementationsRegistry [#4392](https://github.com/sofa-framework/sofa/pull/4392)
- [Helper] Fix Colormap when using HSV to RGB [#4408](https://github.com/sofa-framework/sofa/pull/4408)
- [All] Fix subplugin regression tests [#4420](https://github.com/sofa-framework/sofa/pull/4420)
- [sofaInfo] Fix compilation and behavior [#4422](https://github.com/sofa-framework/sofa/pull/4422)
- [Core] Call super init() in MultiMapping and Multi2Mapping [#4432](https://github.com/sofa-framework/sofa/pull/4432)
- [Helper] Fix new[]/delete mismatch [#4447](https://github.com/sofa-framework/sofa/pull/4447)
- [CMake] Fix SOFA install [#4451](https://github.com/sofa-framework/sofa/pull/4451)
- [GitHub] Fix action using github.context [#4456](https://github.com/sofa-framework/sofa/pull/4456)
- [Helper] PluginManager: Check symbol (real) location and avoid calling wrong entrypoint (Mac/Linux) [#4466](https://github.com/sofa-framework/sofa/pull/4466)
- [GitHUb] Fix actions using cron condition [#4468](https://github.com/sofa-framework/sofa/pull/4468)
- [GitHub] Fix PR messages not properly sent [#4475](https://github.com/sofa-framework/sofa/pull/4475)
- [GitHub] Use finally pull_request_target [#4476](https://github.com/sofa-framework/sofa/pull/4476)
- [SofaCUDA] Fix spatial grid compilation with double [#4478](https://github.com/sofa-framework/sofa/pull/4478)
- [image] Fix compilation on Windows [#4484](https://github.com/sofa-framework/sofa/pull/4484)
- [Testing] Fix installation of gtest headers [#4489](https://github.com/sofa-framework/sofa/pull/4489)
- [Helper] PluginManager Fix test in debug [#4491](https://github.com/sofa-framework/sofa/pull/4491)
- [Topology] Fix method isQuadDeulaunayOriented and its use in splitAlongPath [#4494](https://github.com/sofa-framework/sofa/pull/4494)
- [GitHub] Fix review request Discord msg [#4502](https://github.com/sofa-framework/sofa/pull/4502)
- [Simulation.Core] RequiredPlugin: Avoid calling loadPlugin() twice on start-up [#4509](https://github.com/sofa-framework/sofa/pull/4509)
- [Share] Remove DefaultCollisionGroupManager in the default scenes [#4521](https://github.com/sofa-framework/sofa/pull/4521)
- [all] Fix some warnings [#4529](https://github.com/sofa-framework/sofa/pull/4529)
- [GitHub] Fix GH Discussions Announcements for Discord [#4530](https://github.com/sofa-framework/sofa/pull/4530)
- [Simulation.Core] Make TSAN pass for caduceus [#4535](https://github.com/sofa-framework/sofa/pull/4535)
- [Config] Add cxxopts cmake find module [#4538](https://github.com/sofa-framework/sofa/pull/4538)
- [Sofa.Testing] Add SimpleApi in Config.cmake.in [#4542](https://github.com/sofa-framework/sofa/pull/4542)
- [script] Stale action: remove temporary layer and activate closing [#4560](https://github.com/sofa-framework/sofa/pull/4560)
- [README] Update link to Discord [#4562](https://github.com/sofa-framework/sofa/pull/4562)
- [plugins] Add SimpleApi as a dependency in SofaCarving_test [#4572](https://github.com/sofa-framework/sofa/pull/4572)
- [Constraint.Lagrangian.Solver] Make problemId counter id atomic [#4581](https://github.com/sofa-framework/sofa/pull/4581)
- [Collision.Response.Contact] Make contactId thread-safe [#4582](https://github.com/sofa-framework/sofa/pull/4582)
- [Collision.Detection] Give current intersection methods as parameter to intersection functions [#4583](https://github.com/sofa-framework/sofa/pull/4583)
- [LinearAlgebra] Fix compilation of assert [#4598](https://github.com/sofa-framework/sofa/pull/4598)
- [Multithreading] fix multithread packaging [#4619](https://github.com/sofa-framework/sofa/pull/4619)
- [GitHub] Add timezone info on cron Actions [#4626](https://github.com/sofa-framework/sofa/pull/4626)
- [LinearAlgebra] Trigger error on the Eigen version [#4630](https://github.com/sofa-framework/sofa/pull/4630)
- [Core] Missing call to super init [#4634](https://github.com/sofa-framework/sofa/pull/4634)
- [Demos] Fix regression for fallingBeamLagrangianCollision.scn [#4643](https://github.com/sofa-framework/sofa/pull/4643)
- [Geomagic] Fix compilation. Replace Vector3 by Vec3 [#4661](https://github.com/sofa-framework/sofa/pull/4661)
- [CollisionOBBCapsule] Fix duplicated registration in the factory [#4709](https://github.com/sofa-framework/sofa/pull/4709)
- [Contact] Fix crash if collision models are not provided [#4710](https://github.com/sofa-framework/sofa/pull/4710)
- [Engine.Analyze] Fix initialization of distance type [#4711](https://github.com/sofa-framework/sofa/pull/4711)
- [SceneUtility] Initialize pointer to nullptr [#4712](https://github.com/sofa-framework/sofa/pull/4712)
- [SofaCUDA] extern template instantiations [#4714](https://github.com/sofa-framework/sofa/pull/4714)
- [image_gui] Make it a cleaner SOFA module [#4719](https://github.com/sofa-framework/sofa/pull/4719)
- [SensableEmulation] Fix module name [#4721](https://github.com/sofa-framework/sofa/pull/4721)
- [tests] Adapt code to any Eigen version [#4724](https://github.com/sofa-framework/sofa/pull/4724)
- [image,Mapping.NonLinear] Properly includes config.h before ObjectFactory.h [#4726](https://github.com/sofa-framework/sofa/pull/4726)
- [tests] Fix and test value type string of topology primitives [#4732](https://github.com/sofa-framework/sofa/pull/4732)
- [MultiThreading] Fix module name in BeamLinearMapping_mt [#4740](https://github.com/sofa-framework/sofa/pull/4740)
- [image_gui] Fix module name [#4741](https://github.com/sofa-framework/sofa/pull/4741)
- [Analyze] Fix typos [#4742](https://github.com/sofa-framework/sofa/pull/4742)
- [PlayBack] Add option to set period in the WriteStateCreator visitor [#4744](https://github.com/sofa-framework/sofa/pull/4744)
- [Core] Missing closing brace [#4753](https://github.com/sofa-framework/sofa/pull/4753)
- [Helper] Fix dl open when path are not canonical [#4754](https://github.com/sofa-framework/sofa/pull/4754)


### Cleaning

- [MechanicalLoad] Add buildStiffnessMatrix to TrianglePressureForceField [#4294](https://github.com/sofa-framework/sofa/pull/4294)
- [Simulation.Core] Remove usage of ill-used nodeData in MechanicalGetNonDiagonalMassesCountVisitor and MechanicalVDotVisitor [#4328](https://github.com/sofa-framework/sofa/pull/4328)
- [Simulation.Core] BaseMechanicalVisitor: Deprecate rootData [#4350](https://github.com/sofa-framework/sofa/pull/4350)
- [LinearAlgebra] Use class template argument deduction with MatrixExpr [#4351](https://github.com/sofa-framework/sofa/pull/4351)
- [LinearAlgebra] constexpr if statement when possible [#4352](https://github.com/sofa-framework/sofa/pull/4352)
- [LinearAlgebra] Factorize template specializations of filterValues in CRS matrix [#4355](https://github.com/sofa-framework/sofa/pull/4355)
- [CImgPlugin] Add fetchable CImg and remove embedded cimg [#4357](https://github.com/sofa-framework/sofa/pull/4357)
- [Mapping.Linear] Replace a few beginEdit by accessors [#4363](https://github.com/sofa-framework/sofa/pull/4363)
- [SofaSimpleGUI] Fix calls to deprecated functions [#4390](https://github.com/sofa-framework/sofa/pull/4390)
- [Type] Remove test of a deprecated constructor [#4391](https://github.com/sofa-framework/sofa/pull/4391)
- [Core] Remove pragma directive in cpp file [#4393](https://github.com/sofa-framework/sofa/pull/4393)
- [All] Rename license file into LICENSE-LGPL.md to follow github repository rules [#4410](https://github.com/sofa-framework/sofa/pull/4410)
- [SofaCUDA] Generalize the use of MessageDispatcher in SofaCUDA [#4430](https://github.com/sofa-framework/sofa/pull/4430)
- [Core] Add documentation for the method BaseMapping::buildGeometricStiffnessMatrix [#4431](https://github.com/sofa-framework/sofa/pull/4431)
- [LinearAlgebra] Factorize value filtering [#4442](https://github.com/sofa-framework/sofa/pull/4442)
- [GitHub] Clean existing actions [#4444](https://github.com/sofa-framework/sofa/pull/4444)
- [SofaCUDA] Initialize module from another module [#4453](https://github.com/sofa-framework/sofa/pull/4453)
- [SofaDistanceGrid] remove shadow variable [#4455](https://github.com/sofa-framework/sofa/pull/4455)
- [Defaulttype, LinearAlgera] Fix warnings [#4465](https://github.com/sofa-framework/sofa/pull/4465)
- [Testing] Deprecate BaseSimulationTest::importPlugin [#4467](https://github.com/sofa-framework/sofa/pull/4467)
- [All] CMake: Remove deprecation warnings [#4469](https://github.com/sofa-framework/sofa/pull/4469)
- [framework] Fix typo [#4474](https://github.com/sofa-framework/sofa/pull/4474)
- [all] Apply nested namespaces [#4482](https://github.com/sofa-framework/sofa/pull/4482)
- [all] Missing override keyword [#4483](https://github.com/sofa-framework/sofa/pull/4483)
- [all] Convert some loops to range-based [#4485](https://github.com/sofa-framework/sofa/pull/4485)
- [Config] CMake: fix warning about upper/lowercase for Difflib [#4486](https://github.com/sofa-framework/sofa/pull/4486)
- [README] Update badge with Discord [#4498](https://github.com/sofa-framework/sofa/pull/4498)
- [Plugins] Move MeshSTEPLoader to an external repository [#4504](https://github.com/sofa-framework/sofa/pull/4504)
- [Plugins] Gather external plugins configs in a same folder (proposal) [#4505](https://github.com/sofa-framework/sofa/pull/4505)
- [image] Fix scene test [#4517](https://github.com/sofa-framework/sofa/pull/4517)
- [GitHub] Update version of the actions/github-script [#4522](https://github.com/sofa-framework/sofa/pull/4522)
- [GUI] Remove the New/Save/SaveAs menu options [#4523](https://github.com/sofa-framework/sofa/pull/4523)
- [all] Additional fixes further to v24.06 lifecycle [#4540](https://github.com/sofa-framework/sofa/pull/4540)
- [Core] Update NarrowPhaseDetection.cpp use prefix [#4557](https://github.com/sofa-framework/sofa/pull/4557)
- [Geomagic] Update GeomagicDriver.cpp to fire error when hd.h is not found [#4576](https://github.com/sofa-framework/sofa/pull/4576)
- [CMake] Upgrade cmake_minimum_required to 3.22 [#4586](https://github.com/sofa-framework/sofa/pull/4586)
- [Core] Remove unused debug trace [#4590](https://github.com/sofa-framework/sofa/pull/4590)
- [Core] cmake: Remove last traces of Sofa.Component.Compat [#4596](https://github.com/sofa-framework/sofa/pull/4596)
- [LinearAlgebra] Remove empty source file [#4599](https://github.com/sofa-framework/sofa/pull/4599)
- [Type] Clean and test MatSym [#4600](https://github.com/sofa-framework/sofa/pull/4600)
- [tests] Use appropriate gtest macro [#4607](https://github.com/sofa-framework/sofa/pull/4607)
- [LinearSystem] Remove development artifact [#4620](https://github.com/sofa-framework/sofa/pull/4620)
- [all] Minor last fixes using SimpleApi [#4627](https://github.com/sofa-framework/sofa/pull/4627)
- [SolidMechanics.FEM] Small update on container access to avoid unecessary check [#4639](https://github.com/sofa-framework/sofa/pull/4639)
- [AnimationLoop] Rename Data [#4664](https://github.com/sofa-framework/sofa/pull/4664)
- [Guidelines] Format cpp code [#4695](https://github.com/sofa-framework/sofa/pull/4695)
- [all] Remove some warnings [#4698](https://github.com/sofa-framework/sofa/pull/4698)
- [examples] Run PluginFinder on examples [#4707](https://github.com/sofa-framework/sofa/pull/4707)
- [examples] Remove examples using DefaultCollisionGroupManager [#4708](https://github.com/sofa-framework/sofa/pull/4708)
- [Core] Warn when module name is empty [#4725](https://github.com/sofa-framework/sofa/pull/4725)
- ﻿[PlayBack] Update playback scenes to write/read and compare a beam under gravity motion [#4745](https://github.com/sofa-framework/sofa/pull/4745)
- [tools] Factorize and clean plugin list filtering [#4748](https://github.com/sofa-framework/sofa/pull/4748)
- [tools] Add new dependency to fix in MacOS post-install-fixup [#4749](https://github.com/sofa-framework/sofa/pull/4749)
- [all] Fix typos in comments [#4759](https://github.com/sofa-framework/sofa/pull/4759)
- [All] Delete v23.06 disabled header [#4760](https://github.com/sofa-framework/sofa/pull/4760)
- [examples] Rename example scenes of constraints [#4769](https://github.com/sofa-framework/sofa/pull/4769)


### Refactoring

- [Constraint] Make name of constraints more explicit [#4302](https://github.com/sofa-framework/sofa/pull/4302)
- [Constraint.Projective] Implement applyConstraint from new matrix assembly API [#4309](https://github.com/sofa-framework/sofa/pull/4309)
- [Spring] Extract LinearSpring class in its own file [#4454](https://github.com/sofa-framework/sofa/pull/4454)
- [VolumetricRendering][SofaCUDA] Move CUDA files related to VolumetricRendering [#4487](https://github.com/sofa-framework/sofa/pull/4487)
- [DefaultType] Extract inner classes from SolidTypes in files [#4513](https://github.com/sofa-framework/sofa/pull/4513)
- [LinearAlgebra] Refactor sparse matrix product [#4547](https://github.com/sofa-framework/sofa/pull/4547)
- [plugins] Remove ExternalBehaviorModel and ManualMapping plugins [#4565](https://github.com/sofa-framework/sofa/pull/4565)
- [LinearSystem.Direct] Remove metis dependency [#4588](https://github.com/sofa-framework/sofa/pull/4588)
- [ODESolver] Explicit link to linear solver [#4628](https://github.com/sofa-framework/sofa/pull/4628)
- [Collision.Detection.Algorithm] Rename Data [#4674](https://github.com/sofa-framework/sofa/pull/4674)
- [Collision.Detection.Intersection] Rename Data [#4675](https://github.com/sofa-framework/sofa/pull/4675)
- [Collision.Geometry+Response] Rename Data [#4676](https://github.com/sofa-framework/sofa/pull/4676)
- [Controller+Engine] Rename Data [#4678](https://github.com/sofa-framework/sofa/pull/4678)
- [IO] Rename Data [#4679](https://github.com/sofa-framework/sofa/pull/4679)
- [ODESolver] Rename Data [#4680](https://github.com/sofa-framework/sofa/pull/4680)
- [Playback] Rename Data [#4681](https://github.com/sofa-framework/sofa/pull/4681)
- [Setting] Rename Data [#4682](https://github.com/sofa-framework/sofa/pull/4682)
- [Topology] Rename Data [#4683](https://github.com/sofa-framework/sofa/pull/4683)
- [Visual] Rename Data [#4684](https://github.com/sofa-framework/sofa/pull/4684)
- [Constraint] Rename Data [#4696](https://github.com/sofa-framework/sofa/pull/4696)
- [all] Rename depreciation macros in config.h.in for unique module id… [#4755](https://github.com/sofa-framework/sofa/pull/4755)
- [examples] Rename FixedConstraint example [#4764](https://github.com/sofa-framework/sofa/pull/4764)


### Others

- [Helper] PluginManager: testing loading a plugin with a dependency on an other plugin [#4464](https://github.com/sofa-framework/sofa/pull/4464)
- [Simulation] Tests: introduce multiple parallel simulations [#4580](https://github.com/sofa-framework/sofa/pull/4580)
- [Simulation.Core] Remove useless and annoying timers [#4631](https://github.com/sofa-framework/sofa/pull/4631)
- [image] Fix module name [#4720](https://github.com/sofa-framework/sofa/pull/4720)
- [tools] Add metis relocation in post install fixup [#4767](https://github.com/sofa-framework/sofa/pull/4767)



## [v23.12.00]( https://github.com/sofa-framework/sofa/tree/v23.12.00 )

[Full log](https://github.com/sofa-framework/sofa/compare/v23.06..v23.12)

### Highlighted contributions
- [Mass] Remove DiagonalMass and replace with MeshMatrixMass (with lumping) [#3001](https://github.com/sofa-framework/sofa/pull/3001)
- [LinearAlgebra] Pull Insimo's CompressedRowSparseMatrix into the main branch [#3515](https://github.com/sofa-framework/sofa/pull/3515)
- [plugins] Add fetchable SoftRobots [#3882](https://github.com/sofa-framework/sofa/pull/3882)
- [Sofa.Simulation] First steps to remove the singleton Simulation [#3889](https://github.com/sofa-framework/sofa/pull/3889)
- [plugins] ADD external collisions plugins [#3890](https://github.com/sofa-framework/sofa/pull/3890)
- [LinearAlgebra] Implement CompressedRowSparseMatrixConstraint [#3894](https://github.com/sofa-framework/sofa/pull/3894)
- [Simulation.Core] Refactor DefaultAnimationLoop + multithreading [#3959](https://github.com/sofa-framework/sofa/pull/3959)
- [LinearSolver.Direct] Parallelization of H A^-1 H^T in SparseLDLSolver [#3986](https://github.com/sofa-framework/sofa/pull/3986)
- [LinearSystem] Optim: Only account for affected DoFs [#4001](https://github.com/sofa-framework/sofa/pull/4001)
- [GitHub] Action to check labels [#4079](https://github.com/sofa-framework/sofa/pull/4079)
- [MultiThreading] Introduce parallel CG [#4138](https://github.com/sofa-framework/sofa/pull/4138)
- [LinearSystem] Introduce constant sparsity matrix assembly [#4158](https://github.com/sofa-framework/sofa/pull/4158)
- [LinearSolver] Implement parallel inverse product for all linear solvers [#4255](https://github.com/sofa-framework/sofa/pull/4255)


### Breaking
- [Constraint.Lagrangian] Activate the export of lambda forces by default [#3857](https://github.com/sofa-framework/sofa/pull/3857)
- [all] Change variable name supportOnlySymmetricMatrix in MParams  [#3861](https://github.com/sofa-framework/sofa/pull/3861)
- [all] Unify how Animation/Visual loops are handling their "targetNode" [#3945](https://github.com/sofa-framework/sofa/pull/3945)
- [MechanicalLoad] Implement buildStiffnessMatrix for PlaneForceField [#3972](https://github.com/sofa-framework/sofa/pull/3972)
- [AnimationLoop] Change the default constraint solver in FreeMotionAnimationLoop [#3994](https://github.com/sofa-framework/sofa/pull/3994)
- [Spring] Implement buildStiffnessMatrix in PolynomialRestShapeSpringsForceField [#4009](https://github.com/sofa-framework/sofa/pull/4009)
- [MechanicalLoad] Remove the data force in ConstantFF and solves circular dependency  [#4019](https://github.com/sofa-framework/sofa/pull/4019)
- [all] Lifecycle v23.12 1/n [#4034](https://github.com/sofa-framework/sofa/pull/4034)
- [Core] Make doDrawVisual final [#4045](https://github.com/sofa-framework/sofa/pull/4045)
- [Visual] ADD a visualization flag and draw method for NarrowPhaseDetection [#4048](https://github.com/sofa-framework/sofa/pull/4048)
- [SolidMechanics.TensorMass] Implement buildStiffnessMatrix for TetrahedralTensorMassForceField [#4127](https://github.com/sofa-framework/sofa/pull/4127)
- [BatchGUI] Show progress bar [#4168](https://github.com/sofa-framework/sofa/pull/4168)
- [Constraint.Lagrangian.Solver] Another step to factorize both constraint solvers [#4213](https://github.com/sofa-framework/sofa/pull/4213)
- [all] Replace tinyxml by external tinyxml2 [#4240](https://github.com/sofa-framework/sofa/pull/4240)


### Improvements
- [all] Add namespaces in the factory and a python-like import mechanism [#2512](https://github.com/sofa-framework/sofa/pull/2512)
- [IO][Tests] Add basic test for the MeshSTLLoader [#2999](https://github.com/sofa-framework/sofa/pull/2999)
- [SofaCarving] Add an example written in python [#3457](https://github.com/sofa-framework/sofa/pull/3457)
- [MultiThreading] Implement domain decomposition for a lock-free parallelism [#3566](https://github.com/sofa-framework/sofa/pull/3566)
- [Components] Add functions to visualization of TetrahedronFemForceField [#3821](https://github.com/sofa-framework/sofa/pull/3821)
- [LinearAlgebra] Change access specifier for the method set [#3834](https://github.com/sofa-framework/sofa/pull/3834)
- [Mapping.NonLinear] Warns when non-symmetric matrix is produced [#3838](https://github.com/sofa-framework/sofa/pull/3838)
- [Spring] Limit spring force to some axis [#3849](https://github.com/sofa-framework/sofa/pull/3849)
- [Spring] Limit spring force to some axis [#3850](https://github.com/sofa-framework/sofa/pull/3850)
- [Constraint] Add tag to classify constraints [#3888](https://github.com/sofa-framework/sofa/pull/3888)
- [Spring] Rename RestShapeSpringsForceField to a more user-oriented name [#3903](https://github.com/sofa-framework/sofa/pull/3903)
- [Mapping.NonLinear] Implement buildGeometricStiffnessMatrix for DistanceFromTargetMapping + example [#3921](https://github.com/sofa-framework/sofa/pull/3921)
- [Config] Allow fast math mode for gcc/clang [#3922](https://github.com/sofa-framework/sofa/pull/3922)
- [all] Use NVI design pattern for drawVisual [#3931](https://github.com/sofa-framework/sofa/pull/3931)
- [all] Add DeprecatedData and RemovedData [#3934](https://github.com/sofa-framework/sofa/pull/3934)
- [Helper] add the experimental filesystem library for gcc-8 compilation [#3944](https://github.com/sofa-framework/sofa/pull/3944)
- [FEM.Elastic] Implement buildStiffnessMatrix in HexahedralFEMForceField [#3969](https://github.com/sofa-framework/sofa/pull/3969)
- [Spring] Implement buildStiffnessMatrix for SpringForceField [#3970](https://github.com/sofa-framework/sofa/pull/3970)
- [MechanicalLoad] Implement buildStiffnessMatrix for LinearForceField [#3973](https://github.com/sofa-framework/sofa/pull/3973)
- [Collision.Response.Contact] Implement buildStiffnessMatrix in PenalityContactForceField [#3974](https://github.com/sofa-framework/sofa/pull/3974)
- [FEM.Elastic] Implement buildStiffnessMatrix for TetrahedralCorotationalFEMForceField [#3981](https://github.com/sofa-framework/sofa/pull/3981)
- [LinearSystem.Direct] More details in the error message [#3985](https://github.com/sofa-framework/sofa/pull/3985)
- [FEM.Linear] Implement buildStiffnessMatrix and addKToMatrix for TriangularFEMForceField [#3991](https://github.com/sofa-framework/sofa/pull/3991)
- [Spring] Implement buildStiffnessMatrix for AngularSpringForceField [#3993](https://github.com/sofa-framework/sofa/pull/3993)
- [MechanicalLoad] Implement buildStiffnessMatrix for ConicalForceField [#3997](https://github.com/sofa-framework/sofa/pull/3997)
- [MechanicalLoad] Restore computation of derivatives in DiagonalVelocityDampingForceField [#3999](https://github.com/sofa-framework/sofa/pull/3999)
- [SofaCUDA] Implement buildStiffnessMatrix and buildDampingMatrix in TLED [#4000](https://github.com/sofa-framework/sofa/pull/4000)
- [MechanicalLoad] Implement buildStiffnessMatrix in EdgePressureForceField [#4004](https://github.com/sofa-framework/sofa/pull/4004)
- [MechanicalLoad] Implement buildStiffnessMatrix in EllipsoidForceField [#4005](https://github.com/sofa-framework/sofa/pull/4005)
- [Spring] Implement buildStiffnessMatrix in FastTriangularBendingSprings [#4006](https://github.com/sofa-framework/sofa/pull/4006)
- [MechanicalLoad] Implement buildStiffnessMatrix for OscillatingTorsionPressureForceField [#4007](https://github.com/sofa-framework/sofa/pull/4007)
- [SolidMechanics] Implement buildStiffnessMatrix for QuadBendingFEMForceField [#4015](https://github.com/sofa-framework/sofa/pull/4015)
- [MechanicalLoad] Implement buildStiffnessMatrix for QuadPressureForceField [#4018](https://github.com/sofa-framework/sofa/pull/4018)
- [FEM.Elastic] Add option to compute principal stress direction in TriangularFEMForceFieldOptim  [#4027](https://github.com/sofa-framework/sofa/pull/4027)
- [Geometry] Add method isPointOnEdge in Edge structure [#4028](https://github.com/sofa-framework/sofa/pull/4028)
- [Geometry] Add method intersectionWithEdge in Edge structure [#4029](https://github.com/sofa-framework/sofa/pull/4029)
- [Geometry] Add methods getBarycentricCoordinates and isPointInTriangle in Triangle class [#4053](https://github.com/sofa-framework/sofa/pull/4053)
- [examples] Introduce falling beam example [#4055](https://github.com/sofa-framework/sofa/pull/4055)
- [all] Add dates for user deprecation classes RemovedData and DeprecatedData [#4059](https://github.com/sofa-framework/sofa/pull/4059)
- [LinearSolver.Direct] More details in error message [#4060](https://github.com/sofa-framework/sofa/pull/4060)
- [GitHub] Action to check PR titles [#4081](https://github.com/sofa-framework/sofa/pull/4081)
- [MechanicalLoad] Implement buildStiffnessMatrix in SurfacePressureForceField [#4097](https://github.com/sofa-framework/sofa/pull/4097)
- [MechanicalLoad] Implement buildStiffnessMatrix for SphereForceField [#4099](https://github.com/sofa-framework/sofa/pull/4099)
- [FEM.Elastic] Implement buildStiffnessMatrix for TriangularFEMForceFieldOptim [#4105](https://github.com/sofa-framework/sofa/pull/4105)
- [plugins] Add fetchable ShapeMatchingPlugin [#4106](https://github.com/sofa-framework/sofa/pull/4106)
- [Constraint.Lagrangian.Correction] Decrease the severity of not finding a file in PrecomputedConstraintCorrection [#4108](https://github.com/sofa-framework/sofa/pull/4108)
- [FEM.HyperElastic] Implement buildStiffnessMatrix for StandardTetrahedralFEMForceField [#4110](https://github.com/sofa-framework/sofa/pull/4110)
- [MechanicalLoad] Implement buildStiffnessMatrix for TorsionForceField [#4115](https://github.com/sofa-framework/sofa/pull/4115)
- [Mass] Add a callback on the lumping data in MMMass [#4128](https://github.com/sofa-framework/sofa/pull/4128)
- [Constraint.Lagrangian.Solver] GenericConstraintSolver: use given re-ordered list for unbuilt GS [#4132](https://github.com/sofa-framework/sofa/pull/4132)
- [Core] Store default value in a Data [#4133](https://github.com/sofa-framework/sofa/pull/4133)
- [all] Use SimulationInitDoneEvent instead of AnimateBeginEvent [#4160](https://github.com/sofa-framework/sofa/pull/4160)
- [GitHub] filter action if not on sofa-framework repository  [#4171](https://github.com/sofa-framework/sofa/pull/4171)
- [LinearSystem] Assemble non-mapped and mapped matrices in parallel [#4172](https://github.com/sofa-framework/sofa/pull/4172)
- [Config] Integrate Tracy profiler [#4182](https://github.com/sofa-framework/sofa/pull/4182)
- [Config] CMake: Dont check for IPO at every configure step [#4191](https://github.com/sofa-framework/sofa/pull/4191)
- [Constraint.Lagrangian.Solver] GenericConstraintSolver: avoid repeated allocation in loops [#4195](https://github.com/sofa-framework/sofa/pull/4195)
- [LinearSolver.Direct] Better distribution of tasks among threads [#4220](https://github.com/sofa-framework/sofa/pull/4220)
- [all] Add the folder sofa-launcher to the resources component [#4245](https://github.com/sofa-framework/sofa/pull/4245)
- [MatrixAccumulator] adds 6x6 matrix handling [#4247](https://github.com/sofa-framework/sofa/pull/4247)
- [LinearAlgebra] Speedup accumulation on BTDMatrix [#4248](https://github.com/sofa-framework/sofa/pull/4248)
- [LinearAlgebra] Support 6x6 matrices accumulation in BaseMatrix [#4253](https://github.com/sofa-framework/sofa/pull/4253)
- [example] Speedup TorusFall with parallel inverse product [#4256](https://github.com/sofa-framework/sofa/pull/4256)
- [Helper] ADD option to cmake for advanced timer [#4259](https://github.com/sofa-framework/sofa/pull/4259)
- [LinearSystem] Speedup computation of Jacobian matrices  [#4317](https://github.com/sofa-framework/sofa/pull/4317)
- [FEM.Elastic] Reference instead of a copy in TriangularFEMForceField [#4332](https://github.com/sofa-framework/sofa/pull/4332)
- [All] Add GIT_REF option for external plugins [#4448](https://github.com/sofa-framework/sofa/pull/4448)
- [Testing] Externalize (find_package() or fetch) googletest/gtest [#4471](https://github.com/sofa-framework/sofa/pull/4471)


### Bug Fixes
- [test] Fix unit test on RestShapeSpringsForceField [#3864](https://github.com/sofa-framework/sofa/pull/3864)
- [test] Fix failing unit test [#3876](https://github.com/sofa-framework/sofa/pull/3876)
- [Constraint.Lagrangian.Solver] LCPConstraintSolver: Fix when mu=0 (no friction) [#3905](https://github.com/sofa-framework/sofa/pull/3905)
- [applications] remove unused variable [#3920](https://github.com/sofa-framework/sofa/pull/3920)
- [Projective] Fix and test FixedPlaneConstraint [#3925](https://github.com/sofa-framework/sofa/pull/3925)
- [MechanicalLoad] Restore addKToMatrix and test SurfacePressureForceField [#3935](https://github.com/sofa-framework/sofa/pull/3935)
- [Core.Topology] Restore invalid ids in invalid containers [#3962](https://github.com/sofa-framework/sofa/pull/3962)
- [Sofa.Component] FIX default color for TetrahedronFEMForceField [#3971](https://github.com/sofa-framework/sofa/pull/3971)
- [image] Fix crash in case no parameters provided to TransferFunction [#3976](https://github.com/sofa-framework/sofa/pull/3976)
- [Mapping.NonLinear] Fix assert in RigidMapping [#3978](https://github.com/sofa-framework/sofa/pull/3978)
- [Diffusion] Fix buildStiffnessMatrix in TetrahedronDiffusionFEMForceField [#4012](https://github.com/sofa-framework/sofa/pull/4012)
- [Core] Fix drawing of Objects when hiding Visual Models  [#4044](https://github.com/sofa-framework/sofa/pull/4044)
- [FEM.Elastic] Compute BBox in triangle FEM [#4061](https://github.com/sofa-framework/sofa/pull/4061)
- [FEM.Elastic] Fix typo in error message [#4062](https://github.com/sofa-framework/sofa/pull/4062)
- [LinearAlgebra] Fix matrix sizes when filtering [#4063](https://github.com/sofa-framework/sofa/pull/4063)
- [MechanicalLoad] Fix compilation further to new RemovedData constructor [#4094](https://github.com/sofa-framework/sofa/pull/4094)
- [MechanicalLoad] Fix plane force field buildStiffnessMatrix [#4098](https://github.com/sofa-framework/sofa/pull/4098)
- [GUI.Qt] Fix crash if filename is null [#4102](https://github.com/sofa-framework/sofa/pull/4102)
- [Tutorials] Update and fix oneTetrahedron tutorial [#4103](https://github.com/sofa-framework/sofa/pull/4103)
- [Helper] Fix wrong function called when writing jpg file [#4111](https://github.com/sofa-framework/sofa/pull/4111)
- [test] Make quaternion test deterministic and portable [#4126](https://github.com/sofa-framework/sofa/pull/4126)
- [Constraint.Lagrangian.Solver] Fix default value for computeConstraintForces Data [#4129](https://github.com/sofa-framework/sofa/pull/4129)
- [Sofa.GL] Fix doDrawVisual for OglLabel [#4142](https://github.com/sofa-framework/sofa/pull/4142)
- [SofaSphFluid] Fix: scenes warnings and failing examples [#4149](https://github.com/sofa-framework/sofa/pull/4149)
- [SofaSphFluid] Fix: internal draw method not restoring default parameters [#4150](https://github.com/sofa-framework/sofa/pull/4150)
- [Mass] Fix UniformMass vertexMass value should not be set to 0 if nbr of points reach 0 [#4151](https://github.com/sofa-framework/sofa/pull/4151)
- [Topology.Grid] Fix SparseGridTopology and SparseGridRamificationTopology crash at init if mesh file is not found [#4164](https://github.com/sofa-framework/sofa/pull/4164)
- [Topology] Rename Edge::pointBaryCoefs into Edge::getBarycentricCoordinates [#4165](https://github.com/sofa-framework/sofa/pull/4165)
- [LinearSolver.Direct] SparseLDL: Fix crash in addJMInvJtLocal [#4180](https://github.com/sofa-framework/sofa/pull/4180)
- [Helper] Fix warning in MeshTopologyLoader.cpp [#4181](https://github.com/sofa-framework/sofa/pull/4181)
- [examples] Fix warning of UncoupledCC in caduceus [#4187](https://github.com/sofa-framework/sofa/pull/4187)
- [LinearSystem] MatrixLinearSystem: add registration in the factory for BTDMatrix6 [#4189](https://github.com/sofa-framework/sofa/pull/4189)
- [SofaCarving] Fix warnings in SofaCarving_test due to use of deprecated methods [#4193](https://github.com/sofa-framework/sofa/pull/4193)
- [GUI.Qt] Fix libQGLViewer cmake install [#4198](https://github.com/sofa-framework/sofa/pull/4198)
- [Helper] Update ComponentChange for MechanicalMatrixMapper [#4235](https://github.com/sofa-framework/sofa/pull/4235)
- [LinearAlgebra] Robustify accesses to empty matrices [#4236](https://github.com/sofa-framework/sofa/pull/4236)
- [Topology] Check indices out-of-bound in TriangleSetTopologyContainer [#4242](https://github.com/sofa-framework/sofa/pull/4242)
- [github] Fix name of PR author in GitHub workflows [#4267](https://github.com/sofa-framework/sofa/pull/4267)
- [SofaCUDA] Dont use both version of cublas (legacy or v2) [#4274](https://github.com/sofa-framework/sofa/pull/4274)
- [Sofa.GUI.Qt] Add cmake module for QGLViewer [#4290](https://github.com/sofa-framework/sofa/pull/4290)
- [all] Fix warnings [#4291](https://github.com/sofa-framework/sofa/pull/4291)
- [LinearAlgera, Core] Fix linking with LTO on MacOS/Clang [#4293](https://github.com/sofa-framework/sofa/pull/4293)
- [MultiThreading] Avoid Static Initialization Order Fiasco [#4307](https://github.com/sofa-framework/sofa/pull/4307)
- [SofaCUDA] FIX compilation SofaCUDA along with SparseGrid with Cuda12 [#4319](https://github.com/sofa-framework/sofa/pull/4319)
- [SofaAssimp] Fix the FindAssimp.cmake [#4326](https://github.com/sofa-framework/sofa/pull/4326)
- [image] image_gui to compile with Qt6 [#4330](https://github.com/sofa-framework/sofa/pull/4330)
- [Haption] Partially fix the plugin [#4338](https://github.com/sofa-framework/sofa/pull/4338)
- [github] quick fix for GHD script [#4347](https://github.com/sofa-framework/sofa/pull/4347)
- [github] fix stale action [#4348](https://github.com/sofa-framework/sofa/pull/4348)
- [GUI] Fix compilation using QDocBrowser [#4354](https://github.com/sofa-framework/sofa/pull/4354)
- [VolumetricRendering] Fix the compilation [#4398](https://github.com/sofa-framework/sofa/pull/4398)
- [VolumetricRendering] Fix crashes in batch mode [#4436](https://github.com/sofa-framework/sofa/pull/4436)
- [LinearSolver.Direct] Fix metis dependency [#4450](https://github.com/sofa-framework/sofa/pull/4450)
- [Simulation.Common] Fix tinyXML2 install for windows [#4525](https://github.com/sofa-framework/sofa/pull/4525)
- [all] Install FindTinyXML2  [#4545](https://github.com/sofa-framework/sofa/pull/4545)
- [cmake] Remove error in Findcxxopt [#4554](https://github.com/sofa-framework/sofa/pull/4554)
- [GUI.qt] Set link to tinyxml2 to PRIVATE and fix config file [#4558](https://github.com/sofa-framework/sofa/pull/4558)
- [Config] Fix findcxxopt when version is not specified [#4564](https://github.com/sofa-framework/sofa/pull/4564)
- [Config] Fix Findmetis module when using config mode [#4570](https://github.com/sofa-framework/sofa/pull/4570)
- [all] Fix tinyxml2 dependency [#4574](https://github.com/sofa-framework/sofa/pull/4574)


### Cleaning
- [plugins] Remove fetching of SofaPython [#3855](https://github.com/sofa-framework/sofa/pull/3855)
- [Constraint.Lagrangian] Add messages when no compliance is given [#3858](https://github.com/sofa-framework/sofa/pull/3858)
- [all] include base class inl file [#3865](https://github.com/sofa-framework/sofa/pull/3865)
- [SofaCUDA] No longer use deprecated texture references in HexaTLED [#3868](https://github.com/sofa-framework/sofa/pull/3868)
- [SofaCUDA] Deprecated CudaTexture.h [#3869](https://github.com/sofa-framework/sofa/pull/3869)
- [Config] Fix cross-compilation for embedded external libs [#3870](https://github.com/sofa-framework/sofa/pull/3870)
- [all] Deprecate unused verbose data [#3871](https://github.com/sofa-framework/sofa/pull/3871)
- [contact] Add missing call to super init [#3884](https://github.com/sofa-framework/sofa/pull/3884)
- [examples] Fix unstable scene constantMomentum.scn [#3886](https://github.com/sofa-framework/sofa/pull/3886)
- [SolidMechanics] Use accessors & make geometrical data required in BFF [#3887](https://github.com/sofa-framework/sofa/pull/3887)
- [SofaCUDA] Replace deprecated vector types [#3902](https://github.com/sofa-framework/sofa/pull/3902)
- [Helper] Improve text message for users in ComponentChange [#3913](https://github.com/sofa-framework/sofa/pull/3913)
- [all] Minor clean of types [#3915](https://github.com/sofa-framework/sofa/pull/3915)
- [examples] Remove example for MechanicalMatrixMapper [#3919](https://github.com/sofa-framework/sofa/pull/3919)
- [MechanicalLoad] Replace doUpdateInternal by callback: ConstantFF [#3924](https://github.com/sofa-framework/sofa/pull/3924)
- [FEM.Elastic] Implement buildStiffnessMatrix for FastTetrahedralCorotationalForceField [#3929](https://github.com/sofa-framework/sofa/pull/3929)
- [all] Cosmetic: apply nested namespaces style [#3932](https://github.com/sofa-framework/sofa/pull/3932)
- [Helper] Properly deprecate an already deprecated function [#3933](https://github.com/sofa-framework/sofa/pull/3933)
- [all] Make local variables const [#3937](https://github.com/sofa-framework/sofa/pull/3937)
- [Tests] Properly remove exported files [#3942](https://github.com/sofa-framework/sofa/pull/3942)
- [all] Remove few lines of code that do nothing [#3946](https://github.com/sofa-framework/sofa/pull/3946)
- [all] Implement empty buildDampingMatrix [#3948](https://github.com/sofa-framework/sofa/pull/3948)
- [IO.Mesh] More debug info in the error message in MeshSTLLoader [#3949](https://github.com/sofa-framework/sofa/pull/3949)
- [Sofa.Simulation] Remove Node::bwdInit [#3954](https://github.com/sofa-framework/sofa/pull/3954)
- [Mapping/tests] Remove dependency on SceneCreator [#3955](https://github.com/sofa-framework/sofa/pull/3955)
- [Simulation.Core] Deprecate unused classes *VisitorScheduler [#3957](https://github.com/sofa-framework/sofa/pull/3957)
- [Helper.System] Introduce function append for paths [#3961](https://github.com/sofa-framework/sofa/pull/3961)
- [Project] Start dev phase v23.12 [#3963](https://github.com/sofa-framework/sofa/pull/3963)
- [all] Minor clean on indentation and warning [#3975](https://github.com/sofa-framework/sofa/pull/3975)
- [MechanicalLoad] Remove empty draw function in LinearForceField [#3979](https://github.com/sofa-framework/sofa/pull/3979)
- [tests] Restore commented unit tests [#3982](https://github.com/sofa-framework/sofa/pull/3982)
- [CHANGELOG] Update further to latest changes in v23.06 [#3998](https://github.com/sofa-framework/sofa/pull/3998)
- [Component] Follow changes from SOFA #3889 [#4013](https://github.com/sofa-framework/sofa/pull/4013)
- [LinearAlgebra] CompressedRowSparseMatrix: add virtual destructor [#4020](https://github.com/sofa-framework/sofa/pull/4020)
- [GuiQt] Remove some unactivated code [#4025](https://github.com/sofa-framework/sofa/pull/4025)
- [constraint] Update PrecomputedConstraintCorrection logs when loading compliance file [#4026](https://github.com/sofa-framework/sofa/pull/4026)
- [all] Lifecycle v23.12 2/n [#4040](https://github.com/sofa-framework/sofa/pull/4040)
- [GUI.Qt] Remove unused recorder feature [#4041](https://github.com/sofa-framework/sofa/pull/4041)
- [MechanicalLoad] Use prefix d_ in DiagonalVelocityDampingForceField [#4046](https://github.com/sofa-framework/sofa/pull/4046)
- [all] Lifecycle v23.12 3/n [#4056](https://github.com/sofa-framework/sofa/pull/4056)
- [all] Clean unused warnings in SOFA [#4057](https://github.com/sofa-framework/sofa/pull/4057)
- [all] Lifecycle v23.12 4/n [#4058](https://github.com/sofa-framework/sofa/pull/4058)
- [plugins] Fix OptionGroup deprecated constructor in image and SofaCUDA plugin [#4064](https://github.com/sofa-framework/sofa/pull/4064)
- [all] Update code regarding lifecycle in Topology.h [#4065](https://github.com/sofa-framework/sofa/pull/4065)
- [all] Fix warnings related to Vec and unused var [#4067](https://github.com/sofa-framework/sofa/pull/4067)
- [all] Remove some simple uses of bwdInit() [#4075](https://github.com/sofa-framework/sofa/pull/4075)
- [SofaMatrix] Remove CImgPlugin dependency [#4112](https://github.com/sofa-framework/sofa/pull/4112)
- [all] Replace deprecated wbloc function by its new function name [#4118](https://github.com/sofa-framework/sofa/pull/4118)
- [Core] VecId: set correct message for the deleted function holonomicC [#4120](https://github.com/sofa-framework/sofa/pull/4120)
- [IO.Mesh.Tests] Reduce number of logs by unactivating printlog [#4148](https://github.com/sofa-framework/sofa/pull/4148)
- [Constraint.Lagrangian.Solver] LCPConstraintSolver: remove useless computation if printLog is enabled [#4170](https://github.com/sofa-framework/sofa/pull/4170)
- [Scene] Eigen3-SVD.scn: disable printLog [#4184](https://github.com/sofa-framework/sofa/pull/4184)
- [Constraint.Lagrangian.Solver] Clean both constraint solvers [#4185](https://github.com/sofa-framework/sofa/pull/4185)
- [Topology.Container] Update and rename methods in EdgeSetGeometryAlgorithms to compute barycentric coordinates [#4190](https://github.com/sofa-framework/sofa/pull/4190)
- [Topology.Container] Remove method writeMSHfile in GeometryAlgorithms components [#4192](https://github.com/sofa-framework/sofa/pull/4192)
- [all] Fix mismatch on explicit template instantiations [#4210](https://github.com/sofa-framework/sofa/pull/4210)
- [SofaCUDA] Move cuda GUI dependent code [#4227](https://github.com/sofa-framework/sofa/pull/4227)
- [all] Forgotten scoped timers [#4237](https://github.com/sofa-framework/sofa/pull/4237)
- [all] Fix warnings [#4238](https://github.com/sofa-framework/sofa/pull/4238)
- [all] Remove TODO.md [#4244](https://github.com/sofa-framework/sofa/pull/4244)
- [Type, Helper] Remove unused __STL_MEMBER_TEMPLATES parts [#4251](https://github.com/sofa-framework/sofa/pull/4251)
- [GL] Remove deprecated (and incomplete) Color class [#4264](https://github.com/sofa-framework/sofa/pull/4264)
- [FEM, Mapping] dont mix type::fixed_array and type::Vec [#4269](https://github.com/sofa-framework/sofa/pull/4269)
- [all] Fix headeronly extlibs licenses [#4272](https://github.com/sofa-framework/sofa/pull/4272)
- [all] Externalize cxxopts [#4273](https://github.com/sofa-framework/sofa/pull/4273)
- [Topology.Mapping] SimpleTesselatedHexaTopologicalMapping: use correct type for Index [#4279](https://github.com/sofa-framework/sofa/pull/4279)
- [GUI.Qt] Minor single-line cleaning [#4308](https://github.com/sofa-framework/sofa/pull/4308)
- [Core] Minor clean of DefaultAnimationLoop  [#4314](https://github.com/sofa-framework/sofa/pull/4314)
- [Collections] Remove reference to non-existing SofaSimulation [#4320](https://github.com/sofa-framework/sofa/pull/4320)
- [All] Changed default plugins [#4322](https://github.com/sofa-framework/sofa/pull/4322)
- [ODESolver.Backward] Convert double to SReal in NewmarkImplicitSolver [#4341](https://github.com/sofa-framework/sofa/pull/4341)
- [Sofa.Testing] Do not build Sofa.Testing if SOFA_BUILD_TESTS is OFF [#4459](https://github.com/sofa-framework/sofa/pull/4459)
- [Simulation.Common] Fix downstream project compilation with tinyXML2 [#4506](https://github.com/sofa-framework/sofa/pull/4506)


### Refactoring
- [Sofa.Core] minor refactoring for Data::read() to move into BaseData the reading code [#3278](https://github.com/sofa-framework/sofa/pull/3278)
- [all] Rename DefaultContactManager into CollisionResponse [#3891](https://github.com/sofa-framework/sofa/pull/3891)
- [FEM.Elastic] Minor refactor of buildStiffnessMatrix in TetrahedronFEMForceField [#3983](https://github.com/sofa-framework/sofa/pull/3983)
- [plugins] Remove PSL files, add ExternalProjectConfig and update CMakeLists [#4047](https://github.com/sofa-framework/sofa/pull/4047)
- Revert "[Sofa.Core] minor refactoring for Data::read() to move into BaseData the reading code" [#4068](https://github.com/sofa-framework/sofa/pull/4068)
- [MechanicalLoad] Implement buildStiffnessMatrix for TaitSurfacePressureForceField [#4116](https://github.com/sofa-framework/sofa/pull/4116)
- [Constraint.Lagrangian.Solver] Unify lists of constraint corrections into a MultiLink [#4117](https://github.com/sofa-framework/sofa/pull/4117)
- [Constraint.Projective] Implement applyConstraint in LinearMovementConstraint [#4144](https://github.com/sofa-framework/sofa/pull/4144)
- [all] Generalize the usage of ScopedAdvancedTimer [#4177](https://github.com/sofa-framework/sofa/pull/4177)
- [all] Replace ScopedAdvancedTimers by macros [#4203](https://github.com/sofa-framework/sofa/pull/4203)
- [LinearSolver] Remove CSparse-based linear solvers [#4258](https://github.com/sofa-framework/sofa/pull/4258)
- [Common] Add message to make the fetch mechanism less hidden [#4310](https://github.com/sofa-framework/sofa/pull/4310)
- [Simulation.Core] Deprecate LocalStorage feature [#4327](https://github.com/sofa-framework/sofa/pull/4327)




## [v23.06.00]( https://github.com/sofa-framework/sofa/tree/v23.06.00 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v22.12..v23.06 )

### Highlighted contributions

- [LinearSystem] Refactor matrix assembly [#2777](https://github.com/sofa-framework/sofa/pull/2777) 
- [Core] Introduce parallel for each [#3548](https://github.com/sofa-framework/sofa/pull/3548) 
- [MultiThreading] Introduce ParallelTetrahedronFEMForceField [#3552](https://github.com/sofa-framework/sofa/pull/3552) 
- [examples] Apply Sofa/Component structure for all examples [#3588](https://github.com/sofa-framework/sofa/pull/3588) 


### Breaking

**Architecture**
- [all] Lifecycle v23.06 1/n [#3634](https://github.com/sofa-framework/sofa/pull/3634) 
- [all] Lifecycle v23.06 2/n [#3649](https://github.com/sofa-framework/sofa/pull/3649) 
- [all] Lifecycle v23.06 3/n [#3654](https://github.com/sofa-framework/sofa/pull/3654) 
- [all] Lifecycle v23.06 4/n  [#3655](https://github.com/sofa-framework/sofa/pull/3655) 
- [Core] Use std::function for Link's Validator (and fix UB) [#3665](https://github.com/sofa-framework/sofa/pull/3665) 
- [Helper] Improve OptionsGroup [#3737](https://github.com/sofa-framework/sofa/pull/3737) 
- [Simulation] Suggest required plugin in the syntax of the scene loader [#3799](https://github.com/sofa-framework/sofa/pull/3799) 

**Modules**
- [Mapping] Adds template to RigidMapping  [#3680](https://github.com/sofa-framework/sofa/pull/3680) 
- [Analyze] Raw pointers to Links [#3683](https://github.com/sofa-framework/sofa/pull/3683) 
- [MechanicalLoad] Stupid check for consistency between state and topology [#3692](https://github.com/sofa-framework/sofa/pull/3692) 
- [Mapping.NonLinear] Geometric stiffness method in an OptionsGroup [#3740](https://github.com/sofa-framework/sofa/pull/3740) 
- [Core.Visual, Component.Visual] Create VisualState (formerly Vec3State) [#3782](https://github.com/sofa-framework/sofa/pull/3782) 

**Plugins / Projects**
- [MultiThreading] Factorize task scheduler user [#3607](https://github.com/sofa-framework/sofa/pull/3607) 


### Improvements

**Architecture**
- [Helper] Use DataTypeInfo in NameDecoder [#3465](https://github.com/sofa-framework/sofa/pull/3465) 
- [DefaultType] RigidCoord/Deriv become iterable [#3536](https://github.com/sofa-framework/sofa/pull/3536) 
- [Core] Give threads a name on Windows [#3551](https://github.com/sofa-framework/sofa/pull/3551) 
- [Core] Display flags spelling suggestion [#3575](https://github.com/sofa-framework/sofa/pull/3575) 
- [Type] Construct matrices based on initializer-lists [#3584](https://github.com/sofa-framework/sofa/pull/3584) 
- [defaulttype] Template aliases for CRS matrices [#3592](https://github.com/sofa-framework/sofa/pull/3592) 
- [Simulation] SceneCheck can be added in plugins [#3597](https://github.com/sofa-framework/sofa/pull/3597) 
- [Contributing] Update file to mention good first issue [#3625](https://github.com/sofa-framework/sofa/pull/3625) 
- [Simulation] Error when trying to load a non-existing file [#3677](https://github.com/sofa-framework/sofa/pull/3677) 
- [all] Give reason when component cannot be created [#3682](https://github.com/sofa-framework/sofa/pull/3682) 
- [Type] Support structured binding for type::fixed_array [#3753](https://github.com/sofa-framework/sofa/pull/3753) 
- Update CONTRIBUTING.md [#3774](https://github.com/sofa-framework/sofa/pull/3774) 

**Modules**
- [Collision.Response.Contact] Implement addKToMatrix for PenalityContactForceField [#3626](https://github.com/sofa-framework/sofa/pull/3626) 
- [Mapping.NonLinear] Implement missing applyJT [#3776](https://github.com/sofa-framework/sofa/pull/3776) 
- [Constraint] Add data to access constraint forces in LCPCS [#3796](https://github.com/sofa-framework/sofa/pull/3796) 
- [Rendering3D] OglModel: Use glMapBufferRange to update buffers [#3822](https://github.com/sofa-framework/sofa/pull/3822) 
- [Logging] Suggestion to better highlight warnings in console [#3914](https://github.com/sofa-framework/sofa/pull/3914) 

**Plugins / Projects**
- [Multithreading] 2 steps to maximize coalescing memory access [#3572](https://github.com/sofa-framework/sofa/pull/3572) 
- [MultiThreading] Parallel springs [#3596](https://github.com/sofa-framework/sofa/pull/3596) 
- [MultiThreading] Prepare data for rendering in parallel [#3599](https://github.com/sofa-framework/sofa/pull/3599) 
- [MultiThreading] Implement addDForce in ParallelStiffSpringForceField [#3668](https://github.com/sofa-framework/sofa/pull/3668) 
- [SofaCUDA] Add missing templates for Cuda MechanicalObject [#3688](https://github.com/sofa-framework/sofa/pull/3688) 
- [SceneChecking] Add check when setting contactStiffness uselessly [#3843](https://github.com/sofa-framework/sofa/pull/3843) 

**Examples / Scenes**
- [example] Introduce an example for DistanceMultiMapping [#3742](https://github.com/sofa-framework/sofa/pull/3742) 
- [examples] Introduce examples for DistanceMapping and SquareDistanceMapping [#3756](https://github.com/sofa-framework/sofa/pull/3756)



### Bug Fixes

**Architecture**
- [Simulation.Graph] SimpleAPI: Remove reference to Node argument in createChild() [#3620](https://github.com/sofa-framework/sofa/pull/3620) 
- [defaulttype] Fix cuda template instantiation [#3646](https://github.com/sofa-framework/sofa/pull/3646)
- [Core] Fix UB (misaligned address) when comparing string [#3664](https://github.com/sofa-framework/sofa/pull/3664) 
- [Helper, GL] Fix various memleaks [#3671](https://github.com/sofa-framework/sofa/pull/3671) 
- [Helper] ArgumentParser: Fix map insertion when parsing more than once [#3672](https://github.com/sofa-framework/sofa/pull/3672) 
- [Defaulttype] Add generic precision aliases for Mat templates [#3675](https://github.com/sofa-framework/sofa/pull/3675) 
- [Config] Disable SOFA_EXTERN_TEMPLATE [#3678](https://github.com/sofa-framework/sofa/pull/3678) 
- [all] Simple robustification [#3685](https://github.com/sofa-framework/sofa/pull/3685) 
- [Helper] Fix FileRepository::relativeToPath [#3693](https://github.com/sofa-framework/sofa/pull/3693) 
- [Core] Fix MSVC warning in Link [#3763](https://github.com/sofa-framework/sofa/pull/3763) 
- [Core] Access cumulative sum of VecIds through proxy class [#3918](https://github.com/sofa-framework/sofa/pull/3918) 

**Modules**
- [GUI] Fix CMake Packaging [#3595](https://github.com/sofa-framework/sofa/pull/3595) 
- [Component.Engine] Replace some explicit instanciations on double with SReal [#3629](https://github.com/sofa-framework/sofa/pull/3629) 
- [Topology.Container.Grid] RegularGrid: Fix rounding errors with SReal=float [#3636](https://github.com/sofa-framework/sofa/pull/3636)
- [Collision.Geometry] Fix transparency while displaying the Bounding Collision Cubemodel [#3658](https://github.com/sofa-framework/sofa/pull/3658) 
- [MechanicalLoad] Fix ConstantForceField when no force given [#3670](https://github.com/sofa-framework/sofa/pull/3670) 
- [Collision] Make the (CubeModel) BoundingTree deterministic [#3687](https://github.com/sofa-framework/sofa/pull/3687) 
- [IO.Mesh] Fix UB when filename is empty [#3689](https://github.com/sofa-framework/sofa/pull/3689) 
- [FEM.Elastic] Check for nullptr in BeamFEMForceField [#3690](https://github.com/sofa-framework/sofa/pull/3690) 
- [Spring] Check for nullptr in RestShapeSpringsForceField [#3691](https://github.com/sofa-framework/sofa/pull/3691) 
- [Testing] Fix CMake config after removal of compat [#3694](https://github.com/sofa-framework/sofa/pull/3694) 
- [LinearAlgebra] Fix crash when matrix has no nonzero values [#3700](https://github.com/sofa-framework/sofa/pull/3700) 
- [IO.Mesh] Fix mesh creation if load called multiple times [#3702](https://github.com/sofa-framework/sofa/pull/3702) 
- [LinearSolver.Direct] Make sure the matrix is factorized in SparseCholeskySolver [#3706](https://github.com/sofa-framework/sofa/pull/3706) 
- [GUI.Qt] Register meta type to fix asynchronous Qt call [#3749](https://github.com/sofa-framework/sofa/pull/3749) 
- [Tests]  S.Components tests compiles with SReal=float [#3755](https://github.com/sofa-framework/sofa/pull/3755) 
- [Mapping.NonLinear] Fix SquareMapping applyDJT [#3761](https://github.com/sofa-framework/sofa/pull/3761)
- [LinearAlgebra] Restore insertion operator for BaseVector [#3775](https://github.com/sofa-framework/sofa/pull/3775) 
- [StateContainer] Fix bug in dynamic data registration [#3783](https://github.com/sofa-framework/sofa/pull/3783) 
- [Rendering3D] Missing StateLifeCycle [#3784](https://github.com/sofa-framework/sofa/pull/3784) 
- [Mapping.Nonlinear] Fix matrix assembly in RigidMapping [#3803](https://github.com/sofa-framework/sofa/pull/3803) 
- [Visual] VisualModelImpl: Fix updateVisual() [#3815](https://github.com/sofa-framework/sofa/pull/3815) 
- [LinearSolver.Iterative] CMake: Fix package configuration [#3840](https://github.com/sofa-framework/sofa/pull/3840) 
- [Rendering3D] OglModel: Revert back to glBufferSubData [#3841](https://github.com/sofa-framework/sofa/pull/3841) 
- [SolidMechanics.Spring] CMake: Fix package configuration [#3846](https://github.com/sofa-framework/sofa/pull/3846) 
- [Spring] Remove unused dependencies [#3848](https://github.com/sofa-framework/sofa/pull/3848) 
- [FEM.Linear] FIX & Minor refactor of buildStiffnessMatrix in TriangleFEMForceField [#3989](https://github.com/sofa-framework/sofa/pull/3989)

**Plugins / Projects**
- [SofaEulerianFluid] Fix: rename shadow variable in Fluid2D [#3561](https://github.com/sofa-framework/sofa/pull/3561) 
- [MultiThreading] ParallelBruteForceBroadPhase: Fix assertion error [#3574](https://github.com/sofa-framework/sofa/pull/3574) 
- [image] Fix Sofa.GUI dependency [#3591](https://github.com/sofa-framework/sofa/pull/3591) 
- [SofaCUDA] Missing support for double precision floating-points [#3603](https://github.com/sofa-framework/sofa/pull/3603) 
- [SofaPhysicsAPI] Fix compilation by replacing use of SofaGUI by sofa::GUI::common [#3612](https://github.com/sofa-framework/sofa/pull/3612) 
- [SofaCUDA] No longer use deprecated texture references in TetraTLED [#3650](https://github.com/sofa-framework/sofa/pull/3650) 
- [MultiThreading] Empty string instead of null pointer in DataExchange [#3686](https://github.com/sofa-framework/sofa/pull/3686) 
- [Geomagic] Fix potential crash at exit when device has not been init [#3698](https://github.com/sofa-framework/sofa/pull/3698) 
- [SofaCUDA] Fix includes pointing to compatibility layer [#3727](https://github.com/sofa-framework/sofa/pull/3727) 
- [plugins] ArticulatedSystemMapping : adds size check & fixes typo [#3751](https://github.com/sofa-framework/sofa/pull/3751) 

**Examples / Scenes**
- [All] Fix example scenes showing warnings or errors [#3526](https://github.com/sofa-framework/sofa/pull/3526) 
- [all] Update totalmass to totalMass [#3622](https://github.com/sofa-framework/sofa/pull/3622) 

**Scripts / Tools**
- [CI, Scenes] Restore custom parameters for the scene-tests [#3674](https://github.com/sofa-framework/sofa/pull/3674) 


### Cleaning

**Architecture**
- [all] Lifecycle v22.06 [#3535](https://github.com/sofa-framework/sofa/pull/3535) 
- [Type] Quat: small optimization for axisToQuat() [#3559](https://github.com/sofa-framework/sofa/pull/3559) 
- [Config] Remove unused SOFA_WITH_THREADING option [#3565](https://github.com/sofa-framework/sofa/pull/3565) 
- [Project] Start v23.06 dev phase [#3573](https://github.com/sofa-framework/sofa/pull/3573) 
- [Cmake] Update warning replacement message for deprecated macro sofa_add_XX  [#3611](https://github.com/sofa-framework/sofa/pull/3611) 
- [Sofa] Remove warnings [#3627](https://github.com/sofa-framework/sofa/pull/3627) 
- [Sofa.framework] Compile and run tests when SReal is float [#3628](https://github.com/sofa-framework/sofa/pull/3628) 
- [Type] Fix massive warning due to Mat.h [#3633](https://github.com/sofa-framework/sofa/pull/3633) 
- [Core] TLink: Fix warnings about comparisons between unnamed enums [#3714](https://github.com/sofa-framework/sofa/pull/3714) 
- [Helper] Deprecate constructor of OptionsGroup  [#3741](https://github.com/sofa-framework/sofa/pull/3741) 
- [Config] Remove suspicious Eigen macro preventing vectorization [#3780](https://github.com/sofa-framework/sofa/pull/3780) 

**Modules**
- [StateContainer] Avoid code duplication in MechanicalObject [#3541](https://github.com/sofa-framework/sofa/pull/3541) 
- [FEM.Elastic] Fix unit tests warnings [#3545](https://github.com/sofa-framework/sofa/pull/3545) 
- [GUI] GUI libraries as plugins [#3550](https://github.com/sofa-framework/sofa/pull/3550) 
- [LinearAlgebra, LinearSolver.Direct] make getSubMatrixDim() compile-time constant [#3556](https://github.com/sofa-framework/sofa/pull/3556) 
- [FEM.Elastic] Consistent default values for poisson's ratio and Young's modulus [#3563](https://github.com/sofa-framework/sofa/pull/3563) 
- [FEM.Elastic] Minor refactoring and optimization in draw [#3564](https://github.com/sofa-framework/sofa/pull/3564) 
- [all] Fix some warnings further v22.12 [#3570](https://github.com/sofa-framework/sofa/pull/3570) 
- [LinearAlgebra] Remove wrong comment [#3582](https://github.com/sofa-framework/sofa/pull/3582) 
- [LinearAlgebra] Factorize rotateMatrix in RotationMatrix [#3586](https://github.com/sofa-framework/sofa/pull/3586) 
- [Mass] Remove empty loop in addForce [#3593](https://github.com/sofa-framework/sofa/pull/3593) 
- [Spring] Remove duplicated code [#3594](https://github.com/sofa-framework/sofa/pull/3594) 
- [Components] rename shadow variable [#3606](https://github.com/sofa-framework/sofa/pull/3606) 
- [Helper,Geometry] Move proximity classes into free functions [#3666](https://github.com/sofa-framework/sofa/pull/3666) 
- [all] Remove a bunch of warnings [#3711](https://github.com/sofa-framework/sofa/pull/3711)
- [Spring] Remove variable redefinitions [#3754](https://github.com/sofa-framework/sofa/pull/3754) 
- [AnimationLoop] Call super init in FreeMotionAnimationLoop [#3791](https://github.com/sofa-framework/sofa/pull/3791) 
- [Hyperelastic] Reformat files and add override attribute [#3792](https://github.com/sofa-framework/sofa/pull/3792) 
- [Visual] VisualModelImpl: Clean and optimize slightly ComputeNormals() [#3805](https://github.com/sofa-framework/sofa/pull/3805) 
- [Constraint] Change print variable names to be consistent with doc [#3816](https://github.com/sofa-framework/sofa/pull/3816) 
- [Components] rename shadow variable [#3818](https://github.com/sofa-framework/sofa/pull/3818) 
- [All] Remove trivial warnings [#3823](https://github.com/sofa-framework/sofa/pull/3823) 
- [Lagrangian.Model] Missing _API keyword [#3833](https://github.com/sofa-framework/sofa/pull/3833) 
- [All] Remove warnings for v23.06 [#3911](https://github.com/sofa-framework/sofa/pull/3911) 

**Plugins / Projects**
- [plugins] rename shadow variable [#3581](https://github.com/sofa-framework/sofa/pull/3581) 
- [SofaCUDA] Clean removed code and add useful one [#3632](https://github.com/sofa-framework/sofa/pull/3632) 
- [GUI.Qt, Multithreading] Fix compilation when SReal=float [#3637](https://github.com/sofa-framework/sofa/pull/3637) 
- [MultiThreading] Remove obsolete documentation [#3667](https://github.com/sofa-framework/sofa/pull/3667) 
- [Multithreading] Fix AnimationLoopParallelScheduler [#3676](https://github.com/sofa-framework/sofa/pull/3676) 
- [SofaCUDA] Remove BeamLinearMapping<Rigid3fTypes,.> [#3684](https://github.com/sofa-framework/sofa/pull/3684) 
- [GUI.Qt] Clean and fix in the "inspector" panel [#3713](https://github.com/sofa-framework/sofa/pull/3713)  
- [applications] rename shadow variable [#3738](https://github.com/sofa-framework/sofa/pull/3738) 
- [SofaPhysicsAPI] Remove unreachable code [#3771](https://github.com/sofa-framework/sofa/pull/3771) 
- [SofaCarving] Refresh : update file architecture and cmake [#3798](https://github.com/sofa-framework/sofa/pull/3798) 
- [BulletCollisionDetection] Revive project [#3800](https://github.com/sofa-framework/sofa/pull/3800) 
- [PreassembledMass] Make it external [#3802](https://github.com/sofa-framework/sofa/pull/3802) 
- [SofaCUDA] Move explicit template instantiations from CudaTypes.cpp to CudaMultiMapping.cpp [#3807](https://github.com/sofa-framework/sofa/pull/3807) 
- [plugins] Remove references to old plugins [#3960](https://github.com/sofa-framework/sofa/pull/3960)
- [plugins] Remove references to external deleted plugins [#3980](https://github.com/sofa-framework/sofa/pull/3980)

**Examples / Scenes**
- [Tutorials] Fix CMake with Sofa.GUI [#3624](https://github.com/sofa-framework/sofa/pull/3624) 
- [Scenes] Use "floating-point type"-independent templates [#3635](https://github.com/sofa-framework/sofa/pull/3635) 
- [examples] Apply rest position in mappings [#3757](https://github.com/sofa-framework/sofa/pull/3757) 




### Refactoring

**Modules**
- [Collision.Detection] Rename DefaultPipeline into CollisionPipeline [#3590](https://github.com/sofa-framework/sofa/pull/3590) 
- [Mapping.NonLinear] Move DistanceMultiMapping in its own files [#3707](https://github.com/sofa-framework/sofa/pull/3707) 

**Plugins / Projects**
- [MultiThreading] Reorganize following SOFA structure [#3598](https://github.com/sofa-framework/sofa/pull/3598) 
- [SofaCUDA] Reorganize following SOFA structure (1/n) [#3601](https://github.com/sofa-framework/sofa/pull/3601) 
- [SofaCUDA] Reorganize following SOFA structure (2/n) [#3605](https://github.com/sofa-framework/sofa/pull/3605) 
- [SofaCUDA] Reorganize following SOFA structure (3/n) [#3660](https://github.com/sofa-framework/sofa/pull/3660) 
- [SofaCUDA] Reorganize following SOFA structure (4/n) [#3701](https://github.com/sofa-framework/sofa/pull/3701) 
- [SofaCUDA] Reorganize following SOFA structure (5/n) [#3758](https://github.com/sofa-framework/sofa/pull/3758) 
- [SofaCUDA] Reorganize following SOFA structure (6/n) [#3760](https://github.com/sofa-framework/sofa/pull/3760) 
- [SofaCUDA] Reorganize following SOFA structure (7/n) [#3785](https://github.com/sofa-framework/sofa/pull/3785) 
- [SofaCUDA] Reorganize following SOFA structure (8/n) [#3795](https://github.com/sofa-framework/sofa/pull/3795) 


### Tests

**Modules**
- [Solver.Direct] Unit test on topological changes leading to empty system [#3501](https://github.com/sofa-framework/sofa/pull/3501) 
- [LinearAlgebra] Unit tests for RotationMatrix [#3585](https://github.com/sofa-framework/sofa/pull/3585)
- [LinearAlgebra] Fix matrix unit test [#3832](https://github.com/sofa-framework/sofa/pull/3832) 

**Examples / Scenes**
- [examples] Introduce example and tests for SquareMapping [#3768](https://github.com/sofa-framework/sofa/pull/3768) 




## [v22.12.00]( https://github.com/sofa-framework/sofa/tree/v22.12.00 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v22.06..v22.12 )

### Highlighted contributions

- [Core, Helper] Add spelling suggestion in ObjectFactory [#3042](https://github.com/sofa-framework/sofa/pull/3042) 
- [Constraint.Lagrangian] Add the NNCG as NLCP solver in GenericCS [#3053](https://github.com/sofa-framework/sofa/pull/3053) 
- [CMake] Introduce CMake presets [#3305](https://github.com/sofa-framework/sofa/pull/3305) 
- [all] Type conversion cleaning (+ user-defined literal) [#3314](https://github.com/sofa-framework/sofa/pull/3314) 
- [Core] Convert warning to error in object factory [#3404](https://github.com/sofa-framework/sofa/pull/3404) 


### Breaking

**Architecture**
- [Core] Make some functions non-virtual [#3242](https://github.com/sofa-framework/sofa/pull/3242) 
- [Core] Add some const qualifier to collision methods & use SReal & nodiscard.  [#3270](https://github.com/sofa-framework/sofa/pull/3270) 
- [Topology] Improve TopologyHandler registration [#3271](https://github.com/sofa-framework/sofa/pull/3271) 
- [Core] Finally disable deprecated code in BaseData and ExectParam's Aspects. [#3279](https://github.com/sofa-framework/sofa/pull/3279) 
- [Simulation] Separate factory code from TaskScheduler [#3480](https://github.com/sofa-framework/sofa/pull/3480)
- [all] Replace all Vector2, Vector3, Vector4 by their short name alias Vec2, Vec3 [#3299](https://github.com/sofa-framework/sofa/pull/3299) 
- [Core] Remove memory leaks [#3183](https://github.com/sofa-framework/sofa/pull/3183) 
- [Core] Avoid extra copy in DataFileNameVector [#3188](https://github.com/sofa-framework/sofa/pull/3188) 

**Modules**
- [Constraint.Lagrangian] Create data links in ConstraintCorrection for linear solver [#3152](https://github.com/sofa-framework/sofa/pull/3152) 
- [LinearSolver] Create data links for preconditioners and ShewchukPCGLinearSolver [#3155](https://github.com/sofa-framework/sofa/pull/3155) 
- [Constraint.Lagrangian] Fix BilateralInteractionConstraint double init and clean some Data [#3327](https://github.com/sofa-framework/sofa/pull/3327) 
- [Constraint.Lagrangian] Remove merge option from BilateralInteractionConstraint [#3328](https://github.com/sofa-framework/sofa/pull/3328) 
- [Constraint.Lagrangian.Solver] Deprecate MechanicalAccumulateConstraint  [#3393](https://github.com/sofa-framework/sofa/pull/3393) 
- [TriangularFEMForceField] Avoid double write access to the TriangleInfo Data in TriangularFEMForceField [#3412](https://github.com/sofa-framework/sofa/pull/3412) 
- [all] Fix compilation with float as floating_point_type [#3435](https://github.com/sofa-framework/sofa/pull/3435) 


### Improvements

**Architecture**
- [Core] Linear time getRoot() method in BaseNode and Node [#3059](https://github.com/sofa-framework/sofa/pull/3059) 
- [Type] Add NoInit constructor for Quat class [#3217](https://github.com/sofa-framework/sofa/pull/3217) 
- [Geometry] Make global variables constexpr [#3233](https://github.com/sofa-framework/sofa/pull/3233) 
- [Core] Reduce calls to getValue in collisions [#3264](https://github.com/sofa-framework/sofa/pull/3264) 
- [Type] Fix/clean and speed up of Mat [#3280](https://github.com/sofa-framework/sofa/pull/3280) 
- [Type] Add fixedarray aliases and BoundingBox3D [#3298](https://github.com/sofa-framework/sofa/pull/3298) 
- [Config] Option to set the number of MSVC processes [#3313](https://github.com/sofa-framework/sofa/pull/3313) 
- [Helper] RAII for DrawTool state life cycle [#3338](https://github.com/sofa-framework/sofa/pull/3338) 
- [Config] Option to enable interprocedural optimization [#3345](https://github.com/sofa-framework/sofa/pull/3345) 
- [Config] Change type of CMake message when adding a module [#3381](https://github.com/sofa-framework/sofa/pull/3381) 
- [CMake] Speed-up Configuration (again) [#3382](https://github.com/sofa-framework/sofa/pull/3382) 
- [Helper] Portable thread local storage duration [#3422](https://github.com/sofa-framework/sofa/pull/3422) 
- [Helper.Accessor] Support more std vector methods in WriteAccessorVector [#3426](https://github.com/sofa-framework/sofa/pull/3426) 
- [Topology] Add mechanism to check checkTopologyInputTypes [#3428](https://github.com/sofa-framework/sofa/pull/3428) 
- [Topology.Container] Check at init for Container in Modifier [#3434](https://github.com/sofa-framework/sofa/pull/3434) 
- [Simulation] Task scheduler can accept callable [#3482](https://github.com/sofa-framework/sofa/pull/3482) 
- [Simulation] Worker threads are no longer static [#3491](https://github.com/sofa-framework/sofa/pull/3491) 
- [Core] Support getObjects on set containers [#3495](https://github.com/sofa-framework/sofa/pull/3495) 
- [Type] Conversion to scalar for Mat1x1 [#3498](https://github.com/sofa-framework/sofa/pull/3498) 
- [VectorTypeInfo] Change default handling of data buffer [#3505](https://github.com/sofa-framework/sofa/pull/3505) 

**Modules**
- [MappedMatrix] MechanicalMatrixMapper: adds option [#3173](https://github.com/sofa-framework/sofa/pull/3173) 
- [Constraint.Lagrangian.Correction] LinearSolverCC: use FullVector API if detected [#3231](https://github.com/sofa-framework/sofa/pull/3231) 
- [Topology_test] Add tests in EdgeSetTopology_test to check topological changes [#3245](https://github.com/sofa-framework/sofa/pull/3245) 
- [SolidMechanics][Spring] Implement applyRemovedEdges for SpringForceField [#3269](https://github.com/sofa-framework/sofa/pull/3269) 
- [StateContainer] Optimize vector operations [#3284](https://github.com/sofa-framework/sofa/pull/3284) 
- [Mapping.Linear] Add support for 2d in SubsetMultiMapping [#3321](https://github.com/sofa-framework/sofa/pull/3321) 
- [Constraint.Lagrangian] Update BilateralInteractionConstraint to support topological changes [#3329](https://github.com/sofa-framework/sofa/pull/3329) 
- [all] Unit tests for getTemplateName in some components [#3380](https://github.com/sofa-framework/sofa/pull/3380) 
- [all] Implement getModuleComponentList for most modules [#3386](https://github.com/sofa-framework/sofa/pull/3386) 
- [Constraint.Lagrangian.Solver] Add events in the constraint solver pipeline [#3418](https://github.com/sofa-framework/sofa/pull/3418) 
- [FEM.Elastic] Speedup hexa drawing in force field [#3420](https://github.com/sofa-framework/sofa/pull/3420) 
- [Visual] Introduce TrailRenderer [#3471](https://github.com/sofa-framework/sofa/pull/3471) 
- [Solver.Direct] Unit tests on empty system [#3500](https://github.com/sofa-framework/sofa/pull/3500) 
- [MechanicalLoad] Move getValue call out of the loops [#3503](https://github.com/sofa-framework/sofa/pull/3503) 

**Plugins / Projects**
- [SofaCUDA] Add ConstraintCorrection instantiation in CudaVec3f/CudaVec3f1 [#3004](https://github.com/sofa-framework/sofa/pull/3004) 
- [SofaCUDA] Make CudaConstantForceField compatible with CudaRigid types [#3164](https://github.com/sofa-framework/sofa/pull/3164) 
- [GUI.performer] Add remove elements function for LineCollisionModel [#3239](https://github.com/sofa-framework/sofa/pull/3239) 
- [Regression] Test Eigen solvers scenes for regression [#3326](https://github.com/sofa-framework/sofa/pull/3326)
- [Regression] Add HexahedronFEMForceFieldAndMass to regression tests [#3367](https://github.com/sofa-framework/sofa/pull/3367) 
- [SofaCarving] Update and add more tests in SofaCarving_test [#3407](https://github.com/sofa-framework/sofa/pull/3407) 
- [SofaCuda] Add method in CudaVector Accessor and CudaBilateralInteractionConstraint [#3460](https://github.com/sofa-framework/sofa/pull/3460) 
- [SofaPhysicsAPI] Add several methods using output parameters [#3520](https://github.com/sofa-framework/sofa/pull/3520) 
- [SofaPhysicsAPI] Add C bindings to access C++ api [#3539](https://github.com/sofa-framework/sofa/pull/3539) 
- [SofaPhysicsAPI] Add methods and C bindings to load SOFA ini file, plugins and messageHandler [#3540](https://github.com/sofa-framework/sofa/pull/3540) 


### Bug Fixes

**Architecture**
- [Config] Rename ide folder for libraries [#3214](https://github.com/sofa-framework/sofa/pull/3214) 
- [CMake] FIX parent modules library consistency [#3225](https://github.com/sofa-framework/sofa/pull/3225) 
- [CMake] FIX out-of-tree SofaGui and runSofa locations [#3229](https://github.com/sofa-framework/sofa/pull/3229)
- [Sofa.Type] Fix compile-time Mat and Vec [#3281](https://github.com/sofa-framework/sofa/pull/3281) 
- [Simulation.Core] Fix import of required plugins [#3322](https://github.com/sofa-framework/sofa/pull/3322) 
- [Topology] Fix internal infinite update loop in TopologySubsetData remove process [#3330](https://github.com/sofa-framework/sofa/pull/3330) 
- [Topology] Update getLastElementIndex in TopologySubsetIndices [#3331](https://github.com/sofa-framework/sofa/pull/3331) 
- [Core.Collision] Fix memory leak in NarrowPhaseDetection and IntersectorFactory [#3337](https://github.com/sofa-framework/sofa/pull/3337) 
- [Type] Remove explicit template instantiation to restore performances [#3349](https://github.com/sofa-framework/sofa/pull/3349) 
- [Topology] Fix topologyHandler removal [#3369](https://github.com/sofa-framework/sofa/pull/3369) 
- [Mapping.Linear] Fix assert [#3432](https://github.com/sofa-framework/sofa/pull/3432) 
- [DefaultType] fix rigidcoord compilation [#3462](https://github.com/sofa-framework/sofa/pull/3462) 
- Revert "[Sofa.Core] Linear time getRoot() method in BaseNode and Node" [#3464](https://github.com/sofa-framework/sofa/pull/3464) 
- [Sofa.Core, SofaSphFluid] Fix compilation with MSVC 2022, v17.4 [#3466](https://github.com/sofa-framework/sofa/pull/3466) 
- [Simulation.Core] Add the _API flag to Task::Status [#3543](https://github.com/sofa-framework/sofa/pull/3543) 
- [Core] Missing include in SingleStateAccessor [#3547](https://github.com/sofa-framework/sofa/pull/3547) 
- [Helper] Fix GenerateRigidMass redirection [#3560](https://github.com/sofa-framework/sofa/pull/3560) 

**Modules**
- [Topology] Fix initTopology call chains for mesh without topology [#3216](https://github.com/sofa-framework/sofa/pull/3216) 
- [Constraint.Lagrangian.Solver] fixes verbose GenericConstraintSolver [#3232](https://github.com/sofa-framework/sofa/pull/3232) 
- [SolidMechanics] Fix draw in MeshSpringForceField [#3235](https://github.com/sofa-framework/sofa/pull/3235) 
- [Constraint.Lagrangian] Make class abstract and add key function [#3240](https://github.com/sofa-framework/sofa/pull/3240) 
- [SolidMechanics.Spring] RestShapeSpringForceField: Fix addKToMatrix [#3249](https://github.com/sofa-framework/sofa/pull/3249) 
- [all] Split Tag & TagSet into separated files and fix missing includes. [#3277](https://github.com/sofa-framework/sofa/pull/3277) 
- [all] Minor warning fixes [#3306](https://github.com/sofa-framework/sofa/pull/3306) 
- [LinearSolver.Direct] Missing call to super init() [#3316](https://github.com/sofa-framework/sofa/pull/3316) 
- [LinearAlgebra] Fix BDT compilation [#3333](https://github.com/sofa-framework/sofa/pull/3333) 
- [BarycentricMappers] Fix potential division by 0 [#3383](https://github.com/sofa-framework/sofa/pull/3383) 
- [Engine.Analyze] ClusteringEngine: set correct values in load() [#3398](https://github.com/sofa-framework/sofa/pull/3398) 
- [Tests] Update tests to new modules [#3406](https://github.com/sofa-framework/sofa/pull/3406) 
- [Tests] Fix mutiple component init calls in several tests [#3447](https://github.com/sofa-framework/sofa/pull/3447) 
- [image, FEM, LinearSolver] fix Eigen3 assertion with SVD [#3452](https://github.com/sofa-framework/sofa/pull/3452) 
- [Constraint.lagrangian] Update BilateralInteractionConstraint namespace [#3468](https://github.com/sofa-framework/sofa/pull/3468) 
- [Topology.container] Fix missing TopologyElementType setting in SparseGridTopology init [#3475](https://github.com/sofa-framework/sofa/pull/3475) 
- [LinearSolver.Direct] Fix crashing unit tests [#3512](https://github.com/sofa-framework/sofa/pull/3512) 
- [LinearSolver.Direct] Segfault in SparseLDLSolver due to empty system matrix [#3529](https://github.com/sofa-framework/sofa/pull/3529) 
- [Tests] Update required plugins in tests [#3542](https://github.com/sofa-framework/sofa/pull/3542) 

**Plugins / Projects**
- [SofaCUDA] FIX namespaces [#2935](https://github.com/sofa-framework/sofa/pull/2935) 
- [Plugins] Fix Cmake configuration w/o compat [#3209](https://github.com/sofa-framework/sofa/pull/3209) 
- [GUI.Qt] Fix MSAA sampling setup [#3220](https://github.com/sofa-framework/sofa/pull/3220) 
- [DrawToolGL] Fix ill-formed drawLines with multiple colors [#3260](https://github.com/sofa-framework/sofa/pull/3260) 
- [SofaSphFluid] Fix required plugin in scenes [#3272](https://github.com/sofa-framework/sofa/pull/3272) 
- [SofaCarving] Fix bug in CarvingManager when searching for target collision model [#3276](https://github.com/sofa-framework/sofa/pull/3276) 
- [GUI/runSofa] Cmake: tweaks and fixes [#3323](https://github.com/sofa-framework/sofa/pull/3323) 
- [CImgPlugin] Replace new keyword with creating unique pointers to properly clean up allocated memory. [#3365](https://github.com/sofa-framework/sofa/pull/3365) 
- [SofaMatrix] Fix crash if init fails in FillReducingOrdering [#3366](https://github.com/sofa-framework/sofa/pull/3366) 
- [SofaDistanceGrid] Set invalid state if cannot load mesh [#3400](https://github.com/sofa-framework/sofa/pull/3400) 
- [SofaMatrix] Update plugin CMakeLists to avoid strong dependency on Qt [#3423](https://github.com/sofa-framework/sofa/pull/3423) 
- [SofaNewmat] Fix modules [#3427](https://github.com/sofa-framework/sofa/pull/3427) 
- [plugins] Fix warnings in SofaSphFluid and image [#3439](https://github.com/sofa-framework/sofa/pull/3439) 
- [GUI.Qt] Restore QGLViewer for Qt6 [#3454](https://github.com/sofa-framework/sofa/pull/3454) 
- [GUI.Qt] Restore Stats (Charts) and DocBrowser for Qt6 [#3456](https://github.com/sofa-framework/sofa/pull/3456) 
- [Modeler] rename shadow variable [#3546](https://github.com/sofa-framework/sofa/pull/3546) 
- [GUI.Common] Cmake: Fix message when searching for Sofa.GL [#3549](https://github.com/sofa-framework/sofa/pull/3549) 
- [SofaEulerianFluid] Fix: rename shadow variable in Fluid3D [#3557](https://github.com/sofa-framework/sofa/pull/3557)
- [CollisionOBBCapsule] Fix generation of configuration file for installation [#3576](https://github.com/sofa-framework/sofa/pull/3576)

**Examples / Scenes**
- [Examples] Fix & Clean ProjectToPlaneConstraint and RegularGridTopology_dimension scenes [#3453](https://github.com/sofa-framework/sofa/pull/3453) 

### Cleaning

**Architecture**
- [Core] Minor cleaning [#3176](https://github.com/sofa-framework/sofa/pull/3176) 
- [objectmodel] Remove redefinition of initData [#3190](https://github.com/sofa-framework/sofa/pull/3190) 
- [Sofa.Core] Remove shadow variables [#3212](https://github.com/sofa-framework/sofa/pull/3212) 
- [Sofa.Type] Constexpr Quaternion [#3221](https://github.com/sofa-framework/sofa/pull/3221) 
- [Helper] Reorganize accessors files and add tests [#3234](https://github.com/sofa-framework/sofa/pull/3234) 
- [Sofa.DefaultType] Constexpr VecTypes and RigidTypes [#3237](https://github.com/sofa-framework/sofa/pull/3237) 
- [Sofa.Core] Remove compilation warning because of un-used argument in BaseClass.h [#3243](https://github.com/sofa-framework/sofa/pull/3243) 
- [Topology] Remove Disabled and Deprecated method <= 22.06 [#3250](https://github.com/sofa-framework/sofa/pull/3250) 
- [Core][Type] Minor warning fixes [#3283](https://github.com/sofa-framework/sofa/pull/3283) 
- [Sofa.Defaulttype] Move Rigid{Coord, Deriv, Mass} from RigidTypes into their own headers [#3282](https://github.com/sofa-framework/sofa/pull/3282) 
- [Sofa.Core] Add missing #include [#3297](https://github.com/sofa-framework/sofa/pull/3297) 
- [Sofa.Core] Use pragma, single line namespaces and move forward declarations in fwd.h [#3303](https://github.com/sofa-framework/sofa/pull/3303) 
- [Sofa.Core] Remove the use of Link in MechanicalParams & ConstraintParams [#3304](https://github.com/sofa-framework/sofa/pull/3304) 
- [Type] Disable deprecated methods [#3346](https://github.com/sofa-framework/sofa/pull/3346) 
- [Type] Deprecate Color.h [#3347](https://github.com/sofa-framework/sofa/pull/3347) 
- [Type] equalsZero returns bool, not a real [#3371](https://github.com/sofa-framework/sofa/pull/3371) 
- [Helper] Deprecate error-prone resize and add emplace_back [#3373](https://github.com/sofa-framework/sofa/pull/3373) 
- [Simulation] Change LoadFromMemory signature to remove unused parameter Size [#3376](https://github.com/sofa-framework/sofa/pull/3376) 
- [Sofa.Core] VecId: move definition of VecTypeLabels in its own Translation Unit [#3401](https://github.com/sofa-framework/sofa/pull/3401) 
- [Sofa.Core] TagSet: move method definitions in cpp [#3403](https://github.com/sofa-framework/sofa/pull/3403) 
- [Sofa.Config] CMake: Remove "both" as a choice for SOFA_FLOATING_POINT_TYPE parameter [#3436](https://github.com/sofa-framework/sofa/pull/3436) 
- [Simulation.Core] Clean Multithreading code [#3448](https://github.com/sofa-framework/sofa/pull/3448) 
- [Core] Minor cleaning [#3492](https://github.com/sofa-framework/sofa/pull/3492) 


**Modules**
- [all] Remove code disabled in v21.06 [#3163](https://github.com/sofa-framework/sofa/pull/3163) 
- [Spring] RestSpringsForceField: Unify Vec/Rigid implementation [#3175](https://github.com/sofa-framework/sofa/pull/3175) 
- [LinearSolver.Direct] BTDLinearSolver: Clean debug informations and rename data [#3226](https://github.com/sofa-framework/sofa/pull/3226) 
- [all] Set of warning fixes [#3227](https://github.com/sofa-framework/sofa/pull/3227) 
- [All] Remove deprecation warnings while building the deprecated thing itself [#3236](https://github.com/sofa-framework/sofa/pull/3236) 
- [LinearAlgebra] Remove deprecated code [#3251](https://github.com/sofa-framework/sofa/pull/3251) 
- [tests] Fix remove SofaComponentAll plugin which doesn't exist anymore [#3266](https://github.com/sofa-framework/sofa/pull/3266) 
- [Test] Remove the output of a test from file versioning [#3285](https://github.com/sofa-framework/sofa/pull/3285) 
- [Collision] Cleaning pass on types [#3287](https://github.com/sofa-framework/sofa/pull/3287) 
- [S.C.MechanicalLoad] PlaneForcefield: Fix "potential divide by zero" warnings in msvc [#3315](https://github.com/sofa-framework/sofa/pull/3315) 
- [LinearAlgebra] Explicit instantiations and extern template for some matrices [#3334](https://github.com/sofa-framework/sofa/pull/3334) 
- [Mapping] Add checks at init in TopologicalMapping [#3339](https://github.com/sofa-framework/sofa/pull/3339) 
- [Topology.Container.Grid] trivial optimization in GridTopology [#3348](https://github.com/sofa-framework/sofa/pull/3348) 
- [All] Reduce calls to getValue [#3356](https://github.com/sofa-framework/sofa/pull/3356) 
- [all] Remove in-class typedefs pointing to common sofa::type [#3357](https://github.com/sofa-framework/sofa/pull/3357) 
- [Lagrangian.Solver] Move MechanicalGetConstraintResolutionVisitor in its own files [#3390](https://github.com/sofa-framework/sofa/pull/3390) 
- [Lagrangian.Solver] Move MechanicalGetConstraintViolationVisitor in its own files [#3391](https://github.com/sofa-framework/sofa/pull/3391) 
- [Lagrangian.Solver] Move ConstraintStoreLambdaVisitor in visitors folder [#3392](https://github.com/sofa-framework/sofa/pull/3392) 
- [Topology.Mapping] Types cleaning in Edge2QuadTopologicalMapping [#3408](https://github.com/sofa-framework/sofa/pull/3408) 
- [SolidMechanics.FEM] Clean and optimise TriangularFEMForceField draw method [#3413](https://github.com/sofa-framework/sofa/pull/3413)
- [Topology.Mapping] Replace all beginEdit by writeAccessor to access Loc2GlobMap [#3429](https://github.com/sofa-framework/sofa/pull/3429) 
- [All] Fix unnecessary copy to access topology buffer when const ref can be used [#3446](https://github.com/sofa-framework/sofa/pull/3446) 
- [Lagrangian.Solver] Merge duplicated code into small but expressive functions [#3474](https://github.com/sofa-framework/sofa/pull/3474) 
- [all] Remove compilation warnings [#3494](https://github.com/sofa-framework/sofa/pull/3494) 
- [FEM.Elastic] Clean draw method [#3508](https://github.com/sofa-framework/sofa/pull/3508) 
- [all] Lifecycle v22.12 [#3534](https://github.com/sofa-framework/sofa/pull/3534) 

**Plugins / Projects**
- [tutorials] Fix CMakeLists targets [#3200](https://github.com/sofa-framework/sofa/pull/3200) 
- [Project] Start v22.12 dev phase [#3218](https://github.com/sofa-framework/sofa/pull/3218) 
- [SofaImplicitField] Fix CMake for new NG architecture [#3223](https://github.com/sofa-framework/sofa/pull/3223)
- [SofaValidation] is a plugin, not a collection [#3296](https://github.com/sofa-framework/sofa/pull/3296) 
- [SofaSimpleGUI] Without compatibility layer [#3301](https://github.com/sofa-framework/sofa/pull/3301) 
- [ExternalBehaviorModel] Without compatibility layer [#3302](https://github.com/sofa-framework/sofa/pull/3302) 
- [SofaCarving] Some cleaning in carvingManager to use link instead of string and clean warning in scenes [#3332](https://github.com/sofa-framework/sofa/pull/3332) 
- [image] Fix CMake warning from deprecated function [#3359](https://github.com/sofa-framework/sofa/pull/3359) 
- [SofaDistanceGrid] Clean examples [#3399](https://github.com/sofa-framework/sofa/pull/3399) 
- [image] Remove example scenes requiring Flexible plugin [#3421](https://github.com/sofa-framework/sofa/pull/3421) 
- [GL.Rendering3D] OglModel: Minor cleanups [#3417](https://github.com/sofa-framework/sofa/pull/3417) 
- [MultiThreading] Clean file format [#3476](https://github.com/sofa-framework/sofa/pull/3476) 
- [applications] remove shadow variable [#3488](https://github.com/sofa-framework/sofa/pull/3488) 
- [plugins] SofaCUDA does not require Sofa.GL [#3514](https://github.com/sofa-framework/sofa/pull/3514) 
- [SofaPhysicsAPI] Some small cleaning and add access to VisualModel* [#3519](https://github.com/sofa-framework/sofa/pull/3519) 

**Examples / Scenes**
- [examples] Another pass of plugin finder [#3351](https://github.com/sofa-framework/sofa/pull/3351) 

**Scripts / Tools**
- [metis] Upgrade metis and GKlib [#3063](https://github.com/sofa-framework/sofa/pull/3063) 
- [git] Add various temporary files to .gitignore [#3344](https://github.com/sofa-framework/sofa/pull/3344) 


### Refactoring

**Architecture**
- [Sofa.Core] Move operator<<(Data&) into operator<<(BaseData). [#3300](https://github.com/sofa-framework/sofa/pull/3300) 

**Modules**
- [Lagrangian.Solver] Move GenericConstraintProblem in its own files [#3396](https://github.com/sofa-framework/sofa/pull/3396) 




## [v22.06.00]( https://github.com/sofa-framework/sofa/tree/v22.06.00 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v21.12.00...v22.06.00 )

### SOFA-NG

**Architecture**
- [SofaNG] Dispatch SofaHaptics and SofaValidation [#3041]( https://github.com/sofa-framework/sofa/pull/3041 )
- [SofaNG] Dispatch useful SofaMiscCollision components into S.C.Collision [#2896]( https://github.com/sofa-framework/sofa/pull/2896 )
- [SofaNG] Re-organize Framework [#2876]( https://github.com/sofa-framework/sofa/pull/2876 )
- [SofaNG] Re-organize unit tests  [#2873]( https://github.com/sofa-framework/sofa/pull/2873 )
- [SofaNG] Setup AnimationLoop [#2797]( https://github.com/sofa-framework/sofa/pull/2797 )
- [SofaNG] Setup Collision [#2813]( https://github.com/sofa-framework/sofa/pull/2813 )
- [SofaNG] Setup Constraint [#2790]( https://github.com/sofa-framework/sofa/pull/2790 )
- [SofaNG] Setup Diffusion [#2753]( https://github.com/sofa-framework/sofa/pull/2753 )
- [SofaNG] Setup Engine [#2812]( https://github.com/sofa-framework/sofa/pull/2812 )
- [SofaNG] Setup GUI [#2895]( https://github.com/sofa-framework/sofa/pull/2895 )
- [SofaNG] Setup IO [#2582]( https://github.com/sofa-framework/sofa/pull/2582 )
- [SofaNG] Setup LinearSolver [#2717]( https://github.com/sofa-framework/sofa/pull/2717 )
- [SofaNG] Setup Mapping  [#2635]( https://github.com/sofa-framework/sofa/pull/2635 )
- [SofaNG] Setup Mass [#2752]( https://github.com/sofa-framework/sofa/pull/2752 )
- [SofaNG] Setup MechanicalLoad [#2783]( https://github.com/sofa-framework/sofa/pull/2783 )
- [SofaNG] Setup SceneUtility [#2605]( https://github.com/sofa-framework/sofa/pull/2605 )
- [SofaNG] Setup Setting and Controller [#2843]( https://github.com/sofa-framework/sofa/pull/2843 )
- [SofaNG] Setup Sofa.GL (as a plugin) [#2709]( https://github.com/sofa-framework/sofa/pull/2709 )
- [SofaNG] Setup SolidMechanics [#2759]( https://github.com/sofa-framework/sofa/pull/2759 )
- [SofaNG] Setup StateContainer [#2766]( https://github.com/sofa-framework/sofa/pull/2766 )
- [SofaNG] Setup Topology [#2612]( https://github.com/sofa-framework/sofa/pull/2612 )
- [SofaNG] Setup Visual [#2679]( https://github.com/sofa-framework/sofa/pull/2679 )
- [SofaNG] Setup cmake and add ODESolver [#2571]( https://github.com/sofa-framework/sofa/pull/2571 )

**Modules**
- [Sofa] Compilation without Compatibility mode [#2975]( https://github.com/sofa-framework/sofa/pull/2975 )
- [Component.Compat] Remove wrong mappings in compat layer [#2705]( https://github.com/sofa-framework/sofa/pull/2705 )
- [S.C.Engine.Generate] Convert and move MeshTetraStuffing as an engine [#2917]( https://github.com/sofa-framework/sofa/pull/2917 )
- [Sofa.Component.ODESolver] Gather tests + create Sofa.Component.ODESolver.Testing [#2650]( https://github.com/sofa-framework/sofa/pull/2650 )
- [Sofa.Component] Remove empty test projects and move/clean tests from SofaBoundaryCondition [#2991]( https://github.com/sofa-framework/sofa/pull/2991 )
- [Sofa.GUI] Clean CMake variables [#2974]( https://github.com/sofa-framework/sofa/pull/2974 )
- [Sofa.Helper] Refresh ComponentChange info list [#2986]( https://github.com/sofa-framework/sofa/pull/2986 )
- [Sofa.Helper] clean ComponentChange and add new category [#2565]( https://github.com/sofa-framework/sofa/pull/2565 )
- [SofaGraphComponent] Move scene checking ability into its own library [#2960]( https://github.com/sofa-framework/sofa/pull/2960 )

**Plugins / Projects**
- [PluginExample] Move the plugin to an external repository [#2519]( https://github.com/sofa-framework/sofa/pull/2519 )
- [ManifoldTopologies] Move the plugin to an external repository [#2623]( https://github.com/sofa-framework/sofa/pull/2623 )
- [OpenCTMPlugin] Move the plugin to an external repository [#2564]( https://github.com/sofa-framework/sofa/pull/2564 )
- [OptitrackNatNet] Move the plugin to an external repository [#2548]( https://github.com/sofa-framework/sofa/pull/2548 )
- [THMPGSpatialHashing] Move the plugin to an external repository [#2609]( https://github.com/sofa-framework/sofa/pull/2609 )
- [Registration] Move the plugin to an external repository [#2552]( https://github.com/sofa-framework/sofa/pull/2552 )

**Examples / Scenes**

**Scripts / Tools**

### Breaking

**Architecture**

**Modules**
- [All] Lifecycle update before v22.06 [#3090]( https://github.com/sofa-framework/sofa/pull/3090 )
- [GL.Rendering3D] Clean up GL components [#3115]( https://github.com/sofa-framework/sofa/pull/3115 )
- [LinearSolver] Update data name in SparseLDLSolver [#2904] https://github.com/sofa-framework/sofa/pull/2904
- [Sofa.Core] A helper class for consistent component naming [#2631]( https://github.com/sofa-framework/sofa/pull/2631 )
- [SofaGeneralCollisionMesh] Move TriangleOctree utility class to Helper [#2805]( https://github.com/sofa-framework/sofa/pull/2805 )
- [SofaGeneralObjectInteraction] BoxStiffSpringForceField is now a pack of components [#2621]( https://github.com/sofa-framework/sofa/pull/2621 )
- [SofaGeneralSimpleFEM] Some cleaning in TriangularFEMForceFieldOptim [#2567]( https://github.com/sofa-framework/sofa/pull/2567 )
- [SofaSimpleFEM] Create a TriangleFEMUtils class to factorise Triangle/Triangular[FEMForceField] code [#2287]( https://github.com/sofa-framework/sofa/pull/2287 )
- [SolidMechanics] Remove unused _assembling data [#2901]( https://github.com/sofa-framework/sofa/pull/2901 )
- [Topology.Container.Dynamic] Fix duplicate Data Points in PointSetTopologyContainer [#2993]( https://github.com/sofa-framework/sofa/pull/2993 )

**Plugins / Projects**
- [runSofa] Add a button and a mechanism to activate the scenegraph updates [#3026]( https://github.com/sofa-framework/sofa/pull/3026 )
- [runSofa] Remove not working feature in QSofaListView. [#3025]( https://github.com/sofa-framework/sofa/pull/3025 )

**Examples / Scenes**

**Scripts / Tools**

### Improvements

**Project**
- [Contributing] Update info regarding GitHub Discussions [#2741]( https://github.com/sofa-framework/sofa/pull/2741 )
- [Project] Add the graph illustrating the workflow of a PR review in CONTRIBUTING [#3051]( https://github.com/sofa-framework/sofa/pull/3051 )

**Architecture**
- [collections] Install collections in their own directory [#3196]( https://github.com/sofa-framework/sofa/pull/3196 )

**Modules**
- [Constraint.Lagrangian.Correction] Searching for Direct Linear Solver in LinearSolverConstraintCorrection [#3055]( https://github.com/sofa-framework/sofa/pull/3055 ) 
- [GUI.Qt] Redirect Qt messages to the SOFA output stream [#3101]( https://github.com/sofa-framework/sofa/pull/3101 )
- [GUI.Qt] Save scene graph lock state persistently [#3119]( https://github.com/sofa-framework/sofa/pull/3119 ) 
- [GUI] Completing text description for mouse interaction with springs [#3122]( https://github.com/sofa-framework/sofa/pull/3122 )
- [HyperElastic] Add draw method to StandardTetrahedralFEMForceField [#2838]( https://github.com/sofa-framework/sofa/pull/2838 )
- [HyperElastic] Introduce example scene for StandardTetrahedralFEMForceField [#2857]( https://github.com/sofa-framework/sofa/pull/2857 )
- [LinearSolver.Direct] LU solver can be templated to CRSMat3x3 [#2862]( https://github.com/sofa-framework/sofa/pull/2862 )
- [LinearSolver.Direct] Unit tests around SparseLDLSolver [#3050]( https://github.com/sofa-framework/sofa/pull/3050 )
- [LinearSolver] Introduce Eigen solvers [#2926]( https://github.com/sofa-framework/sofa/pull/2926 )
- [Regression] Test linear solvers for regression [#2776]( https://github.com/sofa-framework/sofa/pull/2776 )
- [SceneUtility] Move RequiredPlugin to Sofa.SimulationCore [#2849]( https://github.com/sofa-framework/sofa/pull/2849 )
- [Simulation.Core] Special message if extension is Python [#2998]( https://github.com/sofa-framework/sofa/pull/2998 )
- [Sofa.Config][MSVC] Change SIMD cmake option and flags [#2652]( https://github.com/sofa-framework/sofa/pull/2652 )
- [Sofa.GL] Add method drawScaledTetrahedron [#2586]( https://github.com/sofa-framework/sofa/pull/2586 )
- [Sofa.GUI.Common] Change the current GUI by default choice [#2979]( https://github.com/sofa-framework/sofa/pull/2979 )
- [Sofa.LinearAlgebra] Test CRS product compared to Eigen [#2839]( https://github.com/sofa-framework/sofa/pull/2839 )
- [SofaCore] By default, state accessors get the bbox of their states [#2780]( https://github.com/sofa-framework/sofa/pull/2780 )
- [SofaDefaultType] Speedup MapMapSparseMatrix [#2641]( https://github.com/sofa-framework/sofa/pull/2641 )
- [SofaDeformable] Spring lengths can be a list [#2602]( https://github.com/sofa-framework/sofa/pull/2602 )
- [SofaEngine] BoxROI support for 2d and 1d types [#2600]( https://github.com/sofa-framework/sofa/pull/2600 )
- [SofaGeneralEngine] Extend features of NearestPointROI [#2595]( https://github.com/sofa-framework/sofa/pull/2595 )
- [SofaGeometry] Add geometric methods in class Triangle and Edge [#2587]( https://github.com/sofa-framework/sofa/pull/2587 )
- [SofaGuiQt][WindowProfiler] Add root tree + show overhead [#2643]( https://github.com/sofa-framework/sofa/pull/2643 )
- [SofaHelper] Load image using stb [#2551]( https://github.com/sofa-framework/sofa/pull/2551 )
- [SofaHelper] Reduce AdvancedTimer overhead [#2645]( https://github.com/sofa-framework/sofa/pull/2645 )
- [SofaSimpleFem] Improve data description [#2740]( https://github.com/sofa-framework/sofa/pull/2740 )
- [SofaSimpleFem][SofaGeneralSimpleFem][SofaMiscFem] Avoid vectors normalization [#2647]( https://github.com/sofa-framework/sofa/pull/2647 )
- [SofaSimulationCore] More generic DefaultVisualManagerLoop [#2549]( https://github.com/sofa-framework/sofa/pull/2549 )
- [SofaSimulationGraph] Add the ability to control where the object is added in a node.  [#2396]( https://github.com/sofa-framework/sofa/pull/2396 )
- [SofaSparseSolver] Apply fill in permutation in SparseLDLSolver [#2762]( https://github.com/sofa-framework/sofa/pull/2762 )
- [SofaSparseSolver] Introduction of an asynchronous LDL solver [#2661]( https://github.com/sofa-framework/sofa/pull/2661 )
- [SofaSparseSolver] Added the choice to compute the decomposition at each step [#2756]( https://github.com/sofa-framework/sofa/pull/2756 )

**Plugins / Projects**
- [plugins] Add BeamAdapter as a new plugin [#2890]( https://github.com/sofa-framework/sofa/pull/2890 )
- [ArticulatedSystemPlugin] ArticulatedSystemMapping new features [#2803]( https://github.com/sofa-framework/sofa/pull/2803 )
- [ArticulatedSystemPlugin] New example [#2804]( https://github.com/sofa-framework/sofa/pull/2804 )
- [SofaCUDA] Add ConstantForceField, MultiMapping and SubsetMultiMapping [#2557]( https://github.com/sofa-framework/sofa/pull/2557 )
- [SofaCUDA] Add benchmark scenes to check CudaMeshSpringForceField [#2556]( https://github.com/sofa-framework/sofa/pull/2556 )
- [SofaCUDA] Add benchmarks logs in .csv file [#2588]( https://github.com/sofa-framework/sofa/pull/2588 )
- [SofaCUDA] Add support for NearestPointROI [#2620]( https://github.com/sofa-framework/sofa/pull/2620 )
- [SofaCUDA] Add support of CudaVector for qt gui dataWidget [#2558]( https://github.com/sofa-framework/sofa/pull/2558 )
- [SofaMatrix] A new plugin adding tools for linear system matrix [#2517]( https://github.com/sofa-framework/sofa/pull/2517 )
- [SofaMatrix] Eigen can be used to reorder a mesh in order to reduce fill-in [#2875]( https://github.com/sofa-framework/sofa/pull/2875 )
- [SofaMatrix] Export the compliance matrix of a constraint solver [#2782]( https://github.com/sofa-framework/sofa/pull/2782 )
- [SofaMatrix] More details in the readme file [#2580]( https://github.com/sofa-framework/sofa/pull/2580 )
- [SofaMatrix] Setup arithmetic precision for matrix export [#2724]( https://github.com/sofa-framework/sofa/pull/2724 )

**Examples / Scenes**
- [examples] An example showing a skybox [#2678]( https://github.com/sofa-framework/sofa/pull/2678 )
- [examples] Minor fix in file for topological change process [#3121]( https://github.com/sofa-framework/sofa/pull/3121 )
- [examples] Speed up the falling cubes [#2646]( https://github.com/sofa-framework/sofa/pull/2646 )

**Scripts / Tools**
- [tools] Update sofa launcher to use SofaPython3 [#2968]( https://github.com/sofa-framework/sofa/pull/2968 )

### Bug Fixes

**Architecture**
- [CMake] FIX out-of-tree build [#2940]( https://github.com/sofa-framework/sofa/pull/2940 )
- [CMake] FIX out-of-tree build (2) [#2953]( https://github.com/sofa-framework/sofa/pull/2953 )
- [CMake] FIX out-of-tree configure [#2891]( https://github.com/sofa-framework/sofa/pull/2891 )
- [CMake] Fix install config files [#3031]( https://github.com/sofa-framework/sofa/pull/3031 )
- [Config][CMake] Fix include directories [#3023]( https://github.com/sofa-framework/sofa/pull/3023 )
- [project] Remove duplicate entry in clang-format [#3140]( https://github.com/sofa-framework/sofa/pull/3140 )

**Modules**
- [All] Fix compilation when SOFA_FLOATING_POINT_TYPE is set to float [#2560]( https://github.com/sofa-framework/sofa/pull/2560 )
- [All] Move addInput/addOutput from init() to constructor [#2825]( https://github.com/sofa-framework/sofa/pull/2825 )
- [All] Fix missing #include config.h or invalid ordering that disturb the factory's getTarget() to operate properly. [#2693]( https://github.com/sofa-framework/sofa/pull/2693 )
- [Component.IO.Mesh] Fix substring comparison [#2837]( https://github.com/sofa-framework/sofa/pull/2837 )
- [Container.Dynamic] Compute the bounding box of *GeometryAlgorithms [#3034]( https://github.com/sofa-framework/sofa/pull/3034 )
- [Controller] Fix specialization issue in inline file [#3182]( https://github.com/sofa-framework/sofa/pull/3182 )
- [Core] Dynamic control of the computation of the bounding box [#3080]( https://github.com/sofa-framework/sofa/pull/3080 )
- [Core] Fix ObjectFactory::getEntriesFromTarget that returns duplicated names [#2544]( https://github.com/sofa-framework/sofa/pull/2544 )
- [Controller] Fix specialization issue in inline file [#3182]( https://github.com/sofa-framework/sofa/pull/3182 )
- [FEM.elastic] Some optimisation to make FastTetrahedralCorotational even faster [#2877]( https://github.com/sofa-framework/sofa/pull/2877 )
- [GKlib] Portable random number generation [#3072]( https://github.com/sofa-framework/sofa/pull/3072 )
- [GUI.Common] FIX unused variable [#3158]( https://github.com/sofa-framework/sofa/pull/3158 )
- [GUI.Qt] Move default format setup before application creation [#3105]( https://github.com/sofa-framework/sofa/pull/3105 )
- [Helper] Convert path back slashes to slash [#2970]( https://github.com/sofa-framework/sofa/pull/2970 )
- [Helper] Deduce plugin name from path based on known extension [#2961]( https://github.com/sofa-framework/sofa/pull/2961 )
- [Helper] Fix crash when module does not provide a version [#2949]( https://github.com/sofa-framework/sofa/pull/2949 )
- [Helper] Make TagFactory thread safe [#2942]( https://github.com/sofa-framework/sofa/pull/2942 )
- [IO.Mesh] Fix binary loading in case of degenerated triangles [#3084]( https://github.com/sofa-framework/sofa/pull/3084 )
- [Lagrangian.Correction] LinearSolverConstraintCorrection: Trivial optimisations for MSVC [#3135]( https://github.com/sofa-framework/sofa/pull/3135 )
- [Mass] Fix point topological changes for UniformMass [#2853]( https://github.com/sofa-framework/sofa/pull/2853 )
- [MeshGmsh]Fixed false error detection in MeshGmsh.cpp file [#3030]( https://github.com/sofa-framework/sofa/pull/3030 )
- [Preconditioner] Fix missing find_package() in the cmake.in [#2841]( https://github.com/sofa-framework/sofa/pull/2841 )
- [S.C.Engine] Remove previous engine files and fix some cmake.in [#2909]( https://github.com/sofa-framework/sofa/pull/2909 )
- [S.C.LinearSolver]fix bug in linearsolvers and preconditioners [#2883]( https://github.com/sofa-framework/sofa/pull/2883 )
- [S.C.Mapping] Fix assertion in BarycentricMapper [#2989]( https://github.com/sofa-framework/sofa/pull/2989 )
- [S.C.Topology.Container.Dynamic] Fix assert error in QuadSetTopologyContainer [#2990]( https://github.com/sofa-framework/sofa/pull/2990 )
- [SceneUtility] MakeDataAliasComponent: Remove alias in destructor [#2832]( https://github.com/sofa-framework/sofa/pull/2832 )
- [Sofa.Component.Diffusion][Sofa.Component.Mass] Fix diffusion [#2798]( https://github.com/sofa-framework/sofa/pull/2798 )
- [Sofa.Component.Engine] Change the default drawSize from 0.0 to 1.0 for ROIs. [#3045]( https://github.com/sofa-framework/sofa/pull/3045 )
- [Sofa.Component] Fix compilation when FLOATING_POINT is set to float [#2907]( https://github.com/sofa-framework/sofa/pull/2907 )
- [Sofa.Component] Put the initilization code in init() instead of the entrypoint initExternalPlugin() [#3112]( https://github.com/sofa-framework/sofa/pull/3112 )
- [Sofa.Core] Remove annoying 'unused' warning in StateAccessor [#2835]( https://github.com/sofa-framework/sofa/pull/2835 )
- [Sofa.DefaultType] Removes definition of GLdouble in SolidTypes [#3060]( https://github.com/sofa-framework/sofa/pull/3060 )
- [Sofa.Defaulttype] Revert #2641 (changing unordered_map from map in MapMapSparseMatrix) [#2699]( https://github.com/sofa-framework/sofa/pull/2699 )
- [Sofa.GL.Component] Fix tangents/bitangents type set for OpenGL [#2855]( https://github.com/sofa-framework/sofa/pull/2855 )
- [Sofa.GL] Fix cmake config file for out-of-tree when trying to find glew on Windows [#3098]( https://github.com/sofa-framework/sofa/pull/3098 )
- [Sofa.GL] Fix double->GLFloat conversion [#2628]( https://github.com/sofa-framework/sofa/pull/2628 )
- [Sofa.GL] Fix draw function from *SetGeometryAlgorithms [#3070]( https://github.com/sofa-framework/sofa/pull/3070 )
- [Sofa.GUI.Common] Fix module reference in CMake file [#2994]( https://github.com/sofa-framework/sofa/pull/2994 )
- [Sofa.GUI.Qt] Fix GUI registration [#3007]( https://github.com/sofa-framework/sofa/pull/3007 )
- [Sofa.GUI.Qt] Fix toolTip for button reload [#3199]( https://github.com/sofa-framework/sofa/pull/3199 )
- [Sofa.LinearAlgebra] Fix SOFA_OPENMP [#2669]( https://github.com/sofa-framework/sofa/pull/2669 )
- [Sofa.LinearAlgebra] Fix compilation warnings [#2627]( https://github.com/sofa-framework/sofa/pull/2627 )
- [Sofa.Simulation.Graph] Fix broken context management [#2964]( https://github.com/sofa-framework/sofa/pull/2964 )
- [Sofa.Type] Change (?) default constructor for fixed_array [#2764]( https://github.com/sofa-framework/sofa/pull/2764 )
- [Sofa.Type] Mat: Correct tensorProduct [#2787]( https://github.com/sofa-framework/sofa/pull/2787 )
- [Sofa.Type] Fix cmake typo, which was disabling tests [#3129]( https://github.com/sofa-framework/sofa/pull/3129 ) 
- [Sofa.Visual] Revert PR #2856 [#3073]( https://github.com/sofa-framework/sofa/pull/3073 )
- [SofaBaseCollision] Fixed invalid vector bug in ContactListener [#2676]( https://github.com/sofa-framework/sofa/pull/2676 )
- [SofaBaseLinearSolver] Fix message in MatrixLinearSolver [#2781]( https://github.com/sofa-framework/sofa/pull/2781 )
- [SofaBaseTopology][SofaDeformable][SofaMiscTopology] Proper topological changes in SpringForceField [#2653]( https://github.com/sofa-framework/sofa/pull/2653 )
- [SofaBoundaryCondition] Remove duplicated code [#2830]( https://github.com/sofa-framework/sofa/pull/2830 )
- [SofaCore.Topology] Depreciate method linked to topologyHandler create/destroy function [#2869]( https://github.com/sofa-framework/sofa/pull/2869 )
- [SofaCore] Fix invalid include in MappingHelper [#2632]( https://github.com/sofa-framework/sofa/pull/2632 )
- [SofaCore] Inf fix topo subset indices [#2870]( https://github.com/sofa-framework/sofa/pull/2870 )
- [SofaCore] Make inheritance virtual [#2594]( https://github.com/sofa-framework/sofa/pull/2594 )
- [SofaCore] Restore default component naming for python [#2801]( https://github.com/sofa-framework/sofa/pull/2801 )
- [SofaCore] Restore xml and python different naming conventions [#2773]( https://github.com/sofa-framework/sofa/pull/2773 )
- [SofaCore] Virtual inheritance of BaseObject [#2799]( https://github.com/sofa-framework/sofa/pull/2799 )
- [SofaDeformable] Handle division by zero in MeshSpringForceField [#2596]( https://github.com/sofa-framework/sofa/pull/2596 )
- [SofaDeformable] Springs are able to compute their bounding box [#2599]( https://github.com/sofa-framework/sofa/pull/2599 )
- [SofaEngine] Fix BoxROI undefined behavior [#2604]( https://github.com/sofa-framework/sofa/pull/2604 )
- [SofaFramework] Add tests on aliases for "multiple projects" out-of-tree build [#2566]( https://github.com/sofa-framework/sofa/pull/2566 )
- [SofaGeneralImplicitOdeSolver] Propagate position inside Newton loop [#2584]( https://github.com/sofa-framework/sofa/pull/2584 )
- [SofaGeneralObjectInteraction] Fix dependencies in cmake [#2659]( https://github.com/sofa-framework/sofa/pull/2659 )
- [SofaGuiCommon] Restore argv  [#2802]( https://github.com/sofa-framework/sofa/pull/2802 )
- [SofaGuiQt] Fix some dll export macro missing [#2555]( https://github.com/sofa-framework/sofa/pull/2555 )
- [SofaGui] Restore and update CMake variables and modules [#3011]( https://github.com/sofa-framework/sofa/pull/3011 )
- [SofaHelper] Speedup ReadAccessor conversion operator [#2583]( https://github.com/sofa-framework/sofa/pull/2583 )
- [SofaMiscFEM] Fix TriangleFEMForceField and TriangularFEMForceField to have the same behavior [#2275]( https://github.com/sofa-framework/sofa/pull/2275 )
- [SofaMiscFem] FIX missing Strain Displacement matrix update in TriangularFEMForceField [#2706]( https://github.com/sofa-framework/sofa/pull/2706 )
- [SofaMiscForceField] Fix error while trying to compile a plugin depending on SofaGuiQt [#2707]( https://github.com/sofa-framework/sofa/pull/2707 )
- [SofaMiscForceField][SofaBaseMechanics] Add compile-time conditions to avoid compiling unrealistic cases [#2514]( https://github.com/sofa-framework/sofa/pull/2514 )
- [SofaMiscSolver] Accumulate mapped forces in NewmarkImplicitSolver [#2578]( https://github.com/sofa-framework/sofa/pull/2578 )
- [SofaSparseSolver] Add cmake configuration for Threads [#2739]( https://github.com/sofa-framework/sofa/pull/2739 )
- [SofaSparseSolver] Fix compilation error due to namespace change [#2543]( https://github.com/sofa-framework/sofa/pull/2543 )
- [SofaUserInteraction] Fix compat alias defined in the same scope [#3032]( https://github.com/sofa-framework/sofa/pull/3032 )
- [SolidMechanics] TopologySubsetIndices for RestShapeSpringsForceField [#3037]( https://github.com/sofa-framework/sofa/pull/3037 )
- [StateContainer] Fix VecId names for != V_COORD [#2872]( https://github.com/sofa-framework/sofa/pull/2872 )
- [Topology.Container] Fix save/restoreLastState in draw method were missing [#3143]( https://github.com/sofa-framework/sofa/pull/3143 )

**Plugins / Projects**
- [ArticulatedSystemMapping] Fixed draw method [#3095]( https://github.com/sofa-framework/sofa/pull/3095 )
- [CollisionOBBCapsule] Fix compat for mappers [#2903]( https://github.com/sofa-framework/sofa/pull/2903 )
- [CollisionOBBCapsule] Fix module name [#3107]( https://github.com/sofa-framework/sofa/pull/3107 )
- [CollisionOBBCapsule] Fix return type to support rigid types [#3075]( https://github.com/sofa-framework/sofa/pull/3075 )
- [SofaCUDA] Add matrix3 transpose method on device [#2675]( https://github.com/sofa-framework/sofa/pull/2675 )
- [SofaCUDA] Fail gracefully if no cuda device found [#3087]( https://github.com/sofa-framework/sofa/pull/3087 )
- [SofaCUDA] Fix compilation for SOFA_GPU_CUDA_DOUBLE [#2863]( https://github.com/sofa-framework/sofa/pull/2863 )
- [SofaCUDA] Fix some namespace, headers includes and decl exports [#3003]( https://github.com/sofa-framework/sofa/pull/3003 )
- [SofaCUDA] Fix symbol definition at run time and compilation error [#2634]( https://github.com/sofa-framework/sofa/pull/2634 )
- [SofaCUDA] Ignore CudaTLED-related scene on the CI [#2893]( https://github.com/sofa-framework/sofa/pull/2893 )
- [SofaCUDA] Restore CudaHexahedronFEMForceField [#2535]( https://github.com/sofa-framework/sofa/pull/2535 )
- [SofaCUDA] Rework and move benchmark scenes for HexahedronFEMForceField and TetrahedronFEMForceField [#2561]( https://github.com/sofa-framework/sofa/pull/2561 )
- [SofaCUDA] fix HexahedronFEMForceField double compilation [#3081]( https://github.com/sofa-framework/sofa/pull/3081 )
- [SofaCUDA] update .scene-tests files to fix tests on dashboard [#2616]( https://github.com/sofa-framework/sofa/pull/2616 )
- [SofaCUDA] FIX linking error : needed the code of the destructor [#2708]( https://github.com/sofa-framework/sofa/pull/2708 )
- [SofaCUDA] Forward declaration in wrong namespace [#2923]( https://github.com/sofa-framework/sofa/pull/2923 )
- [SofaMatrix] Move CI setting to the plugin folder [#2579]( https://github.com/sofa-framework/sofa/pull/2579 )
- [SofaNewMat] Fix compilation [#2829]( https://github.com/sofa-framework/sofa/pull/2829 )
- [SofaSphFluid] Fix ParticleSource topologicalChanges use new callbacks mechanism [#2868]( https://github.com/sofa-framework/sofa/pull/2868 )
- [image] fix shadow variable [#2606]( https://github.com/sofa-framework/sofa/pull/2606 )
- [image] remove shadow variable [#2910]( https://github.com/sofa-framework/sofa/pull/2910 )
- [runSofa] Clean up in case GUI init fails [#3106]( https://github.com/sofa-framework/sofa/pull/3106 )
- [runSofa] Fix CMake error if all plugins are disabled [#3069]( https://github.com/sofa-framework/sofa/pull/3069 )
- [runSofa] Fix bug which makes sofa crash when changing a data in a node (issue #2919). [#3020]( https://github.com/sofa-framework/sofa/pull/3020 )
- [runSofa] Fix crash in runSofa when showing object with materials [#3018]( https://github.com/sofa-framework/sofa/pull/3018 )
- [sofaProjectExample] Resuscitate application with gui and simulation (cpp) [#2792]( https://github.com/sofa-framework/sofa/pull/2792 )

**Examples / Scenes**
- [examples] FIX benchmark_cubes.scn needs CollisionOBBCapsule [#2898]( https://github.com/sofa-framework/sofa/pull/2898 )
- [examples] Fix ProjectToPointconstraint scene by using MeshSpringForceField [#2827]( https://github.com/sofa-framework/sofa/pull/2827 )
- [examples] Fix falling cubes scene [#2831]( https://github.com/sofa-framework/sofa/pull/2831 )
- [examples] Fix plugins in a couple of scenes [#2810]( https://github.com/sofa-framework/sofa/pull/2810 )
- [examples] Move PointSplatModel and SpatialGridPointModel scenes into Sph plugin [#2847]( https://github.com/sofa-framework/sofa/pull/2847 )
- [examples] FIX scenes with old CollisionGroupManager component [#3198]( https://github.com/sofa-framework/sofa/pull/3198 )

**Scripts / Tools**
- [tools] Correction of the import of the queue library [#2572]( https://github.com/sofa-framework/sofa/pull/2572 )

### Cleanings

**Project**
- [GitHub] Update Changelog and version [#2546]( https://github.com/sofa-framework/sofa/pull/2546 )
- [Git] Clean project files [#2884]( https://github.com/sofa-framework/sofa/pull/2884 )
- [Project] Update README [#2905]( https://github.com/sofa-framework/sofa/pull/2905 )
- [Readme] Delete references to removed/moved directories in licence paragraph [#2547]( https://github.com/sofa-framework/sofa/pull/2547 )
- [doc] Remove old doc materials  [#2889]( https://github.com/sofa-framework/sofa/pull/2889 )

**Architecture**
- [CMake] Clean modules post-NG [#2980]( https://github.com/sofa-framework/sofa/pull/2980 )
- [CMake] Replace references of SofaNG codename to SOFA [#3102]( https://github.com/sofa-framework/sofa/pull/3102 )
- [CMake] Update all deps to SofaFramework modules [#2958]( https://github.com/sofa-framework/sofa/pull/2958 )
- [CMake] Use standard macros + fix out-of-tree builds [#3120]( https://github.com/sofa-framework/sofa/pull/3120 )
- [CMake] Clean packaging for v22.06 [#3197]( https://github.com/sofa-framework/sofa/pull/3197 )

**Modules**
- [All] Fix minor warnings [#3177]( https://github.com/sofa-framework/sofa/pull/3177 )
- [All] Minor changes to use new SOFA-NG includes [#3160]( https://github.com/sofa-framework/sofa/pull/3160 )
- [All] Remove warnings [#3118]( https://github.com/sofa-framework/sofa/pull/3118 )
- [All] Uniformize includes [#3145]( https://github.com/sofa-framework/sofa/pull/3145 )
- [All] Few fixes to allow compilation with MSVC/Clang-cl [#2563]( https://github.com/sofa-framework/sofa/pull/2563 )
- [All] Minor improvements [#2667]( https://github.com/sofa-framework/sofa/pull/2667 )
- [All] Update Data comments [#2719]( https://github.com/sofa-framework/sofa/pull/2719 )
- [All] Remove few trivial compilation warning & commented code. [#3044]( https://github.com/sofa-framework/sofa/pull/3044 )
- [All] Update the code to remove some deprecation warnings [#2529]( https://github.com/sofa-framework/sofa/pull/2529 )
- [Collision.Geometry] Remove unnecessary *_API in template class definition [#3022]( https://github.com/sofa-framework/sofa/pull/3022 )
- [Collision.Response.Mapper] Missing override keyword [#3083]( https://github.com/sofa-framework/sofa/pull/3083 )
- [Component] All modules have a version [#2948]( https://github.com/sofa-framework/sofa/pull/2948 )
- [Components.Engine.Transform] remove shadow variable in ROIValueMapper [#2987]( https://github.com/sofa-framework/sofa/pull/2987 )
- [Config] Remove unused option SOFA_WITH_DEPRECATED_COMPONENTS [#3172]( https://github.com/sofa-framework/sofa/pull/3172 )
- [Constraint.Lagrangian.Solver] Convert double to SReal [#2922]( https://github.com/sofa-framework/sofa/pull/2922 )
- [Core.Topology] Fix info message when Topology given to topologyHandler is not dynamic [#3142]( https://github.com/sofa-framework/sofa/pull/3142 )
- [Core.Topology] Remove array quadsInHexahedronArray, QuadsOrientationInHexahedronArray should be used [#2995]( https://github.com/sofa-framework/sofa/pull/2995 )
- [Core] Extract BaseLinearSolver class into its own files [#2938]( https://github.com/sofa-framework/sofa/pull/2938 )
- [DefaultType] Prevent division by zero [#2929]( https://github.com/sofa-framework/sofa/pull/2929 )
- [FEM.Elastic] Fix typo on Poisson's ratio [#2911]( https://github.com/sofa-framework/sofa/pull/2911 )
- [FEM.Elastic] Minor changes in TetraXX FEM at init and to be able to access Data  [#2845]( https://github.com/sofa-framework/sofa/pull/2845 )
- [FEM.Elastic] Simplify addkToMatrix in TriangularFEMForceFieldOptim [#2861]( https://github.com/sofa-framework/sofa/pull/2861 )
- [FEM.HyperElastic] Remove optimization based on type of matrix in StandardTetrahedralFEMForceField [#2858]( https://github.com/sofa-framework/sofa/pull/2858 )
- [LMConstraint] Remove LMConstraint folder as it is an external plugin [#2828]( https://github.com/sofa-framework/sofa/pull/2828 )
- [LinearSolver.Direct] Minor clean up in SparseCholeskySolver [#2881]( https://github.com/sofa-framework/sofa/pull/2881 )
- [LinearSolver.Direct] Move advice message into parse method [#3029]( https://github.com/sofa-framework/sofa/pull/3029 )
- [LinearSolver.Iterative] Clean floating point types in CG [#2808]( https://github.com/sofa-framework/sofa/pull/2808 )
- [LinearSolver] Remove explicit template instantiation for floating point types different from SReal [#2939]( https://github.com/sofa-framework/sofa/pull/2939 )
- [LinearSolver] SparseLDLSolver: template warning to info message [#2969]( https://github.com/sofa-framework/sofa/pull/2969 )
- [Mapping.Linear] Fix some warnings [#2933]( https://github.com/sofa-framework/sofa/pull/2933 )
- [Mass] fix some warnings [#2931]( https://github.com/sofa-framework/sofa/pull/2931 )
- [Mass] Make warnings become info_msg in masses [#3116]( https://github.com/sofa-framework/sofa/pull/3116 ) 
- [Metis] Add header files to the project [#2581]( https://github.com/sofa-framework/sofa/pull/2581 )
- [Metis] Disable compilation warnings [#2874]( https://github.com/sofa-framework/sofa/pull/2874 )
- [S.C.LinearSolver] Restore specialized functions in SSORPreconditioner [#2885]( https://github.com/sofa-framework/sofa/pull/2885 )
- [Sofa.Component.Collision] Remove un-needed msg_warning in LocalMinDistance.cpp [#2976]( https://github.com/sofa-framework/sofa/pull/2976 )
- [Sofa.Component.ODESolver] Rewrite tests without SceneCreator [#2733]( https://github.com/sofa-framework/sofa/pull/2733 )
- [Sofa.Core] Use forward declaration for BoundingBox in Base [#2728]( https://github.com/sofa-framework/sofa/pull/2728 )
- [Sofa.DefaultType] Name() should be compile-time evaluable [#3174]( https://github.com/sofa-framework/sofa/pull/3174 )
- [Sofa.GL.Component] OglModel: Uniformize floating point type [#2856]( https://github.com/sofa-framework/sofa/pull/2856 )
- [Sofa.GUI.Component] Add module version [#3038]( https://github.com/sofa-framework/sofa/pull/3038 )
- [Sofa.Geometry][SofaMeshCollision] Deprecate RayTriangleIntersection class [#2763]( https://github.com/sofa-framework/sofa/pull/2763 )
- [Sofa.Helper] Introduce narrow_cast [#2590]( https://github.com/sofa-framework/sofa/pull/2590 )
- [Sofa.Helper] Remove some compilation warnings [#2619]( https://github.com/sofa-framework/sofa/pull/2619 )
- [Sofa.LinearAlgebra] Deprecate unused EigenMatrixManipulator [#2793]( https://github.com/sofa-framework/sofa/pull/2793 )
- [Sofa.Type] Quat: Optimize rotate() (and inverseRotate()) [#3138]( https://github.com/sofa-framework/sofa/pull/3138 )
- [Sofa.Type] Fix some compilation warnings [#2589]( https://github.com/sofa-framework/sofa/pull/2589 )
- [Sofa.Type] Make RGBAColor constexpr [#2630]( https://github.com/sofa-framework/sofa/pull/2630 )
- [Sofa.Type] Simplify template rebinding [#2570]( https://github.com/sofa-framework/sofa/pull/2570 )
- [SofaBaseMechanics] Clean AddMToMatrixFunctor [#2755]( https://github.com/sofa-framework/sofa/pull/2755 )
- [SofaBaseMechanics][SofaMiscForcefield] Auto-detect MassType [#2644]( https://github.com/sofa-framework/sofa/pull/2644 )
- [SofaBaseTopology] Clearer error message [#2638]( https://github.com/sofa-framework/sofa/pull/2638 )
- [SofaBaseVisual] Remove unused background setting in BaseCamera [#2637]( https://github.com/sofa-framework/sofa/pull/2637 )
- [SofaBoundaryCondition] Remove optimization based on type of matrix [#2859]( https://github.com/sofa-framework/sofa/pull/2859 )
- [SofaBoundaryCondition] Replace deprecated headers [#2574]( https://github.com/sofa-framework/sofa/pull/2574 )
- [SofaBoundaryCondition] Sanitize [#2559]( https://github.com/sofa-framework/sofa/pull/2559 )
- [SofaConstraint] Clean timers in GenericConstraintSolver [#2769]( https://github.com/sofa-framework/sofa/pull/2769 )
- [SofaConstraint] Clean timers in LCPConstraintSolver [#2640]( https://github.com/sofa-framework/sofa/pull/2640 )
- [SofaConstraint] Replace raw pointer to Link [#2770]( https://github.com/sofa-framework/sofa/pull/2770 )
- [SofaConstraint] Sanitize [#2577]( https://github.com/sofa-framework/sofa/pull/2577 )
- [SofaCore] Add ObjectFactory_test.cpp [#2696]( https://github.com/sofa-framework/sofa/pull/2696 )
- [SofaCore] Check link to mstate if specified [#2553]( https://github.com/sofa-framework/sofa/pull/2553 )
- [SofaCore] Fix type conversion [#2639]( https://github.com/sofa-framework/sofa/pull/2639 )
- [SofaCore] Name dynamically allocated vec ids [#2811]( https://github.com/sofa-framework/sofa/pull/2811 )
- [SofaCore_test] Minor clean of few unit tests [#2573]( https://github.com/sofa-framework/sofa/pull/2573 )
- [SofaEulerianFluid] Move examples into plugin [#2833]( https://github.com/sofa-framework/sofa/pull/2833 )
- [SofaGeneralEngine] Fix some warnings [#2723]( https://github.com/sofa-framework/sofa/pull/2723 )
- [SofaGuiQt] Error handling when exporting graph [#2771]( https://github.com/sofa-framework/sofa/pull/2771 )
- [SofaGuiQt] Fix english spelling: AdvancedTimer instead of AdvanceTimer [#2816]( https://github.com/sofa-framework/sofa/pull/2816 )
- [SofaGuiQt] Modernize the 'qt::connect' to use the c++ syntax [#2943]( https://github.com/sofa-framework/sofa/pull/2943 )
- [SofaGuiQt] Remove sec unit from GUI [#2742]( https://github.com/sofa-framework/sofa/pull/2742 )
- [SofaGuiQt] Rename attributes to follow SOFA guidelines. [#2944]( https://github.com/sofa-framework/sofa/pull/2944 )
- [SofaKernel] fix shadow variable [#2791]( https://github.com/sofa-framework/sofa/pull/2791 )
- [SofaMiscCollision] Extract OBB/Capsule code into its own plugin [#2820]( https://github.com/sofa-framework/sofa/pull/2820 )
- [SofaMiscFEM] Clean FastTetrahedralCorotationalForceField and add extraData [#2569]( https://github.com/sofa-framework/sofa/pull/2569 )
- [SofaMiscForceField] Fix type conversion in tests [#2714]( https://github.com/sofa-framework/sofa/pull/2714 )
- [SofaRigid][SofaGeneralSimpleFem] Fix typos [#2947]( https://github.com/sofa-framework/sofa/pull/2947 )
- [SofaSimpleFEM_test] Add tests on Tetrahedron, CorotationalTetrahedral and FastTetrahedral FEM ForceField [#2842]( https://github.com/sofa-framework/sofa/pull/2842 )
- [SofaSimpleFem] Prefer usage of accessor in HexahedronFEMForceField [#2778]( https://github.com/sofa-framework/sofa/pull/2778 )
- [SofaSimulation] Remove DAGNodeMultiMappingElement [#2694]( https://github.com/sofa-framework/sofa/pull/2694 )
- [SofaSparseSolver] Disable matrix export in SparseLDLSolver [#2725]( https://github.com/sofa-framework/sofa/pull/2725 )
- [SofaSparseSolver] SparseLUSolver and SparseCholeskySolver support fill reducing permutation [#2788]( https://github.com/sofa-framework/sofa/pull/2788 )
- [SofaTest] Move last tests to Sofa.Component [#2996]( https://github.com/sofa-framework/sofa/pull/2996 )
- [SofaTest] Remove last usages and make it optional [#3000]( https://github.com/sofa-framework/sofa/pull/3000 )
- [SofaUserInteraction] remove shadow variable [#2795]( https://github.com/sofa-framework/sofa/pull/2795 )
- [SolidMechanics.FEM.HyperElastic] Reformat and clean TetrahedronHyperelasticityFEMForceField [#3141]( https://github.com/sofa-framework/sofa/pull/3141 )
- [SolidMechanics] TetrahedronFEMForceField: adds error message for Poisson's ratio  [#2908]( https://github.com/sofa-framework/sofa/pull/2908 )
- [StateContainer] Remove deprecated code for topologicalChanges in MechanicalObject. [#2867]( https://github.com/sofa-framework/sofa/pull/2867 )
- [Topology.Dynamic] Remove implicit conversion warnings [#2973]( https://github.com/sofa-framework/sofa/pull/2973 )

**Plugins / Projects**
- [Plugins] Create ArticulatedSystem plugin (originally from SofaGeneralRigid) [#2684]( https://github.com/sofa-framework/sofa/pull/2684 )
- [Geomagic] Clean scene and code warnings + minor changes [#2846]( https://github.com/sofa-framework/sofa/pull/2846 )
- [Geomagic] Fix CMake and replace includes from legacy header to new architecture [#3076]( https://github.com/sofa-framework/sofa/pull/3076 )
- [SensableEmulation][CMake] Clean Boost dep [#3067]( https://github.com/sofa-framework/sofa/pull/3067 )
- [SofaAssimp] fix shadow variable [#2844]( https://github.com/sofa-framework/sofa/pull/2844 )
- [SofaCUDA] Fix CMake to compile plugin without legacy headers option [#3077]( https://github.com/sofa-framework/sofa/pull/3077 )
- [SofaCUDA][SofaSphFluid] Remove some getTemplateName deprecated methods [#3144]( https://github.com/sofa-framework/sofa/pull/3144 )
- [SofaCUDA] Better readme [#2597]( https://github.com/sofa-framework/sofa/pull/2597 )
- [SofaCUDA] Clean Cuda Collision models [#2673]( https://github.com/sofa-framework/sofa/pull/2673 )
- [SofaCUDA] Clean and factorize CudaTriangularFEMForceFieldOptim [#2568]( https://github.com/sofa-framework/sofa/pull/2568 )
- [SofaCUDA] Convert standard stream to msg_* API [#2864]( https://github.com/sofa-framework/sofa/pull/2864 )
- [SofaCUDA] Modernize CMake for CUDA [#2878]( https://github.com/sofa-framework/sofa/pull/2878 )
- [SofaCUDA] QuadSpringsSphere scenes [#2598]( https://github.com/sofa-framework/sofa/pull/2598 )
- [SofaCUDA] Remove calls to __umul24 on device [#2715]( https://github.com/sofa-framework/sofa/pull/2715 )
- [SofaCUDA] Resurrect CudaTetrahedronTLEDForceField [#2865]( https://github.com/sofa-framework/sofa/pull/2865 )
- [SofaCUDA] harmless cleaning of namespace declaration and header inclusion [#2674]( https://github.com/sofa-framework/sofa/pull/2674 )
- [SofaMatrix] Reduce dependency [#2768]( https://github.com/sofa-framework/sofa/pull/2768 )
- [SofaMatrix][SofaBaseLinearSolver] Move GlobalSystemMatrixExporter [#2545]( https://github.com/sofa-framework/sofa/pull/2545 )
- [SofaPhysicsAPI] Clean CMake file and compile without compat [#3091]( https://github.com/sofa-framework/sofa/pull/3091 )
- [image] remove shadow variables [#2936]( https://github.com/sofa-framework/sofa/pull/2936 )
- [image] Remove qt4 usage [#2663]( https://github.com/sofa-framework/sofa/pull/2663 )
- [image] remove shadow variable [#2928]( https://github.com/sofa-framework/sofa/pull/2928 )
- [image] remove shadow variable [#2722]( https://github.com/sofa-framework/sofa/pull/2722 )
- [image] remove shadow variables [#2921]( https://github.com/sofa-framework/sofa/pull/2921 )

**Examples / Scenes**
- [examples] Optimal RequiredPlugin [#2836]( https://github.com/sofa-framework/sofa/pull/2836 )
- [examples] Remove misplaced example file [#2726]( https://github.com/sofa-framework/sofa/pull/2726 )
- [examples] Remove missing files from the scene list [#2809]( https://github.com/sofa-framework/sofa/pull/2809 )
- [examples] Remove scene which should be in LMConstraint [#2834]( https://github.com/sofa-framework/sofa/pull/2834 )
- [examples] Run PluginFinder [#2950]( https://github.com/sofa-framework/sofa/pull/2950 )
- [examples] Run PluginFinder after some module changes [#2860]( https://github.com/sofa-framework/sofa/pull/2860 )
- [examples] Remove all uses of DefaultCollisionGroupManager [#3104]( https://github.com/sofa-framework/sofa/pull/3104 )

**Scripts / Tools**


____________________________________________________________



## [v21.12.00]( https://github.com/sofa-framework/sofa/tree/v21.12.00 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v21.06.03...v21.12.00 )

### SOFA-NG

**Modules**
- [All] Remove more GeometryAlgorithms usage [#2465]( https://github.com/sofa-framework/sofa/pull/2465 )
- [All] Remove some trivial dependencies on SofaBaseTopology [#2449]( https://github.com/sofa-framework/sofa/pull/2449 )
- [Sofa.Core][SofaBaseTopology] Move TopologyData* from SofaBaseTopology to Sofa.Core [#2423]( https://github.com/sofa-framework/sofa/pull/2423 )
- [Sofa.Core][SofaBaseTopology] Refactor TopologyContainer [#2387]( https://github.com/sofa-framework/sofa/pull/2387 )
- [Sofa.LinearAlgebra] Dispatch *Matrix from BTDLinearSolver, and BlockDiagonalMatrix [#2334]( https://github.com/sofa-framework/sofa/pull/2334 )
- [Sofa.LinearAlgebra] Dispatch EigenMatrix and EigenVector from Eigen2Solver [#2339]( https://github.com/sofa-framework/sofa/pull/2339 )
- [Sofa.LinearAlgebra] Gather tests [#2383]( https://github.com/sofa-framework/sofa/pull/2383 )
- [SofaBaseMechanics] BarycentricMapping: Remove avoidable Sofa.BaseTopology dependencies [#2403]( https://github.com/sofa-framework/sofa/pull/2403 )
- [SofaBaseTopology][SofaMiscForcefield] Remove GeometryAlgorithms from DiagonalMass & MeshMatrixMass [#2436]( https://github.com/sofa-framework/sofa/pull/2436 )
- [SofaFramework] Create Sofa.LinearAlgebra (BaseVector/BaseMatrix & co) [#2314]( https://github.com/sofa-framework/sofa/pull/2314 )

**Plugins / Projects**
- [Plugins] Move CGALPlugin to an external repository [#2500]( https://github.com/sofa-framework/sofa/pull/2500 )
- [Plugins] Move Flexible/Compliant/RigidScale to external repositories [#1873]( https://github.com/sofa-framework/sofa/pull/1873 )
- [Plugins] Move SofaPython to an external repository [#2224]( https://github.com/sofa-framework/sofa/pull/2224 )
- [Plugins] Remove LMConstraint from SOFA repository [#2508]( https://github.com/sofa-framework/sofa/pull/2508 )

### Breaking

**Architecture**

**Modules**
- [All] Unused include directive [#2475]( https://github.com/sofa-framework/sofa/pull/2475 )
- [All] Merge TopologySubsetData and TopologySparseData [#2114]( https://github.com/sofa-framework/sofa/pull/2114 )
- [All] Merge branch topologyChanges_PoC  introducing topological changes callbacks [#2228]( https://github.com/sofa-framework/sofa/pull/2228 )
- [All] Remove ForceMask [#2316]( https://github.com/sofa-framework/sofa/pull/2316 )
- [All] Update collision response names [#2522]( https://github.com/sofa-framework/sofa/pull/2522 )
- [Sofa.BaseTopology] Remove 1D template for 2D/3D primitives in topology algorithms [#2291]( https://github.com/sofa-framework/sofa/pull/2291 )
- [Sofa.Core] Deprecate officially the usage of SofaOStream (sout, serr, sendl in Base) [#2292]( https://github.com/sofa-framework/sofa/pull/2292 )
- [Sofa.Core] Factorize mstate access [#2438]( https://github.com/sofa-framework/sofa/pull/2438 )
- [Sofa.Core] Use links instead of private std::list [#2364]( https://github.com/sofa-framework/sofa/pull/2364 )
- [Sofa.LinearAlgebra] Fix typo 'bloc' -> 'block' [#2404]( https://github.com/sofa-framework/sofa/pull/2404 )
- [SofaBaseLinearSolver] Clean MatrixLinearSolver [#2195]( https://github.com/sofa-framework/sofa/pull/2195 )
- [SofaBaseTopology] Disable method createTopologyHandler with a TopologyHandler* as parameter [#2393]( https://github.com/sofa-framework/sofa/pull/2393 )
- [SofaBaseTopology] Fix Last element index update in TopologyData [#2359]( https://github.com/sofa-framework/sofa/pull/2359 )
- [SofaCore] Clean RotationMatrix [#1995]( https://github.com/sofa-framework/sofa/pull/1995 )
- [SofaGUI] Replace boost's program_options with cxxopts [#2263]( https://github.com/sofa-framework/sofa/pull/2263 )
- [SofaGuiQt] FIX screenshot mechanism [#2507]( https://github.com/sofa-framework/sofa/pull/2507 )
- [SofaGeneralDeformable] Clean class TriangularBendingSpring and add tests [#2363]( https://github.com/sofa-framework/sofa/pull/2363 )
- [SofaKernel] Split the content of DataTracker.h in different headers files. [#2286]( https://github.com/sofa-framework/sofa/pull/2286 )
- [SofaLoader] Rename MeshObjLoader to MeshOBJLoader [#2428]( https://github.com/sofa-framework/sofa/pull/2428 )
- [SofaMiscFem] Quick clean unused parameters in Triangular and TriangleFEMForceField [#2283]( https://github.com/sofa-framework/sofa/pull/2283 )
- [SofaMiscForceField] Fix MeshMatrixMass duplicate Data parameters [#2192]( https://github.com/sofa-framework/sofa/pull/2192 )
- [SofaSimulationCore] Remove unused class VisitorAsync [#1994]( https://github.com/sofa-framework/sofa/pull/1994 )

**Plugins / Projects**
- [CImgPlugin/image] Move file and clean headers [#2307]( https://github.com/sofa-framework/sofa/pull/2307 )
- [MultiThreading] Parallel hexa fem [#2177]( https://github.com/sofa-framework/sofa/pull/2177 )

**Examples / Scenes**

**Scripts / Tools**

### Improvements

**Architecture**
- [CMake] CLEAN and reenable old macro for git infos [#2309]( https://github.com/sofa-framework/sofa/pull/2309 )
- [GitHub] Create bug-report issue template [#2365]( https://github.com/sofa-framework/sofa/pull/2365 )
- [SofaMacros] Improve sofa_install_git_infos [#2373]( https://github.com/sofa-framework/sofa/pull/2373 )
- [SofaScenes] Introduce option to add scenes as a project of the build [#2327]( https://github.com/sofa-framework/sofa/pull/2327 )

**Modules**
- [All] Add warnings if components are missing to solve a specific response [#2523]( https://github.com/sofa-framework/sofa/pull/2523 )
- [SofaHelper] Handle recent MSH format in MeshGmshLoader [#2155]( https://github.com/sofa-framework/sofa/pull/2155 )
- [Sofa.Geometry][Sofa.Topology] Add some functions + create unittests [#2434]( https://github.com/sofa-framework/sofa/pull/2434 )
- [Sofa.Helper] PluginManager: Add optional way to check if a plugin is init at the load stage [#2425]( https://github.com/sofa-framework/sofa/pull/2425 )
- [Sofa.LinearAlgebra] Introduce fast sparse matrix product and use it in MechanicalMatrixMapper [#2394]( https://github.com/sofa-framework/sofa/pull/2394 )
- [SofaBaseLinearSolver] Introduce GlobalSystemMatrixExporter [#2303]( https://github.com/sofa-framework/sofa/pull/2303 )
- [SofaBaseTopology] Add method to register callbacks directly using the topologyData [#2375]( https://github.com/sofa-framework/sofa/pull/2375 )
- [SofaBaseTopology] Ensure to add a topology EndingEvent before propagating to all topologyData [#2376]( https://github.com/sofa-framework/sofa/pull/2376 )
- [SofaBaseVisual] Fix VisualModelImpl to use topologyData and callback to handle topological changes [#2300]( https://github.com/sofa-framework/sofa/pull/2300 )
- [SofaCore] Add const version of getMSState to Mass [#2293]( https://github.com/sofa-framework/sofa/pull/2293 )
- [SofaCore] Clearer message when template parameter is not compatible with current context [#2262]( https://github.com/sofa-framework/sofa/pull/2262 )
- [SofaCore] Required data msg now depends on existing default value [#2527]( https://github.com/sofa-framework/sofa/pull/2527 )
- [SofaDefaultType] Introduce test interface for BaseMatrix [#2280]( https://github.com/sofa-framework/sofa/pull/2280 )
- [SofaGeneralAnimationLoop] More optimization on MechanicalMatrixMapper [#2411]( https://github.com/sofa-framework/sofa/pull/2411 )
- [SofaGeneralAnimationLoop] Remove matrix compression [#2367]( https://github.com/sofa-framework/sofa/pull/2367 )
- [SofaGeneralDeformable] Add option to enable/disable spring display in TriangularBendingSprings [#2297]( https://github.com/sofa-framework/sofa/pull/2297 )
- [SofaGeneralSimpleFem] Update BeamFEMForceField topologyHander and add tests [#2382]( https://github.com/sofa-framework/sofa/pull/2382 )
- [SofaGraphComponent] Restore check of deprecated components  [#2267]( https://github.com/sofa-framework/sofa/pull/2267 )
- [SofaGuiQt] Introduce expand/collapse buttons [#2322]( https://github.com/sofa-framework/sofa/pull/2322 )
- [SofaGuiQt] Introduce expand/collapse buttons in profiler [#2351]( https://github.com/sofa-framework/sofa/pull/2351 )
- [SofaGuiQt] Multiple selection of timers in the profiler [#2141]( https://github.com/sofa-framework/sofa/pull/2141 )
- [SofaGuiQt] UX: graph is easier to read [#2356]( https://github.com/sofa-framework/sofa/pull/2356 )
- [SofaHelper] Add a method getTrace() to BackTrace class. [#2341]( https://github.com/sofa-framework/sofa/pull/2341 )
- [SofaHelper] Factory key type can be other than std::string [#2259]( https://github.com/sofa-framework/sofa/pull/2259 )
- [SofaMiscFEM_test] Add class to test and compare TriangleFEM and TriangularFEMForceField [#2249]( https://github.com/sofa-framework/sofa/pull/2249 )
- [SofaMiscForceField_test] Add TopologicalChanges unit tests for MeshMatrixMass [#2215]( https://github.com/sofa-framework/sofa/pull/2215 )
- [SofaMiscTopology] Add a component TopologyBoundingTrasher to remove mesh going outside from scene bounding box [#2288]( https://github.com/sofa-framework/sofa/pull/2288 )
- [SofaSimpleFem] Add callback on VonMises in TetrahedronFEMForceField [#2407]( https://github.com/sofa-framework/sofa/pull/2407 )
- [SofaSimpleFem] Hexa FEM optimization when assembling by bloc [#2240]( https://github.com/sofa-framework/sofa/pull/2240 )
- [SofaSimpleFem] Simplify bloc-based optimization [#2281]( https://github.com/sofa-framework/sofa/pull/2281 )
- [SofaSimulationCore] Clearer message when a linear solver is missing [#2221]( https://github.com/sofa-framework/sofa/pull/2221 )
- [SofaSparseSolver] Fix msg readibility in SparseLDL [#2294]( https://github.com/sofa-framework/sofa/pull/2294 )
- [SofaSparseSolver] Introduce FillReducingOrdering [#2501]( https://github.com/sofa-framework/sofa/pull/2501 )

**Plugins / Projects**
- [image] Restore ability to select a subset of instanciations  [#2340]( https://github.com/sofa-framework/sofa/pull/2340 )

**Examples / Scenes**
- [examples] Add a new example how to create heterogeneous tet FEM [#2289]( https://github.com/sofa-framework/sofa/pull/2289 )
- [examples] Speedup the caduceus [#2471]( https://github.com/sofa-framework/sofa/pull/2471 )
- [Scenes] Update TriangleSurfaceCutting scene to use a bigger mesh with texture [#2381]( https://github.com/sofa-framework/sofa/pull/2381 )

**Scripts / Tools**

### Bug Fixes

**Architecture**
- [CMake] Clean Sofa.Core CMakeLists + Add missing headers [#2444]( https://github.com/sofa-framework/sofa/pull/2444 )
- [CMake] FIX Threads dependency [#2511]( https://github.com/sofa-framework/sofa/pull/2511 )
- [CMake] FIX libs copy on Windows [#2261]( https://github.com/sofa-framework/sofa/pull/2261 )
- [CMake][modules] FIX out-of-tree builds [#2453]( https://github.com/sofa-framework/sofa/pull/2453 )
- [CMake] Fix return values of Qt5/Qt6 find_packages [#2346]( https://github.com/sofa-framework/sofa/pull/2346 )

**Modules**
- [All] Fix potential bugs/crashes (from warnings) [#2379]( https://github.com/sofa-framework/sofa/pull/2379 )
- [All] Update IDE directory for targets [#2239]( https://github.com/sofa-framework/sofa/pull/2239 )
- [SofaMiscForceField] Fix massDensity vector update when adding new elements in MeshMatrixMass [#2257]( https://github.com/sofa-framework/sofa/pull/2257 )
- [Sofa.Compat] Fix Matrixexpr alias [#2369]( https://github.com/sofa-framework/sofa/pull/2369 )
- [Sofa.Compat] Fix install [#2360]( https://github.com/sofa-framework/sofa/pull/2360 )
- [Sofa.Core] Fix diamond inheritance in PairInteractionConstraint [#2488]( https://github.com/sofa-framework/sofa/pull/2488 )
- [Sofa.DefaultType] Fix declaration of global variables [#2317]( https://github.com/sofa-framework/sofa/pull/2317 )
- [Sofa.GL] Set glew as mandatory [#2358]( https://github.com/sofa-framework/sofa/pull/2358 )
- [Sofa.Helper] Fix and micro-optimize AdvancedTimer [#2349]( https://github.com/sofa-framework/sofa/pull/2349 )
- [Sofa.Helper] More tests for NameDecoder + fix them [#2380]( https://github.com/sofa-framework/sofa/pull/2380 )
- [Sofa.Helper] PluginManager: check if a plugin is already loaded with a different path [#2352]( https://github.com/sofa-framework/sofa/pull/2352 )
- [Sofa.Helper_test] "fix" wstring unittest [#2320]( https://github.com/sofa-framework/sofa/pull/2320 )
- [Sofa.LinearAlgebra] Fix assert in CompressedRowSparseMatrix [#2347]( https://github.com/sofa-framework/sofa/pull/2347 )
- [Sofa.LinearAlgebra] Fix installation (fwd.h) [#2337]( https://github.com/sofa-framework/sofa/pull/2337 )
- [Sofa.Type, Sofa.Topology] Fix testEdgeBuffer with clang (and add constexpr default constructors) [#2357]( https://github.com/sofa-framework/sofa/pull/2357 )
- [Sofa.Type] FIX createFromRotationVEctor, the inequality was wrong [#2332]( https://github.com/sofa-framework/sofa/pull/2332 )
- [Sofa.Type] Optimize constructor with params for sofa::type::vec [#2439]( https://github.com/sofa-framework/sofa/pull/2439 )
- [SofaBaseLinearSolver] CGLinearSolver must call super init() to check required Data [#2419]( https://github.com/sofa-framework/sofa/pull/2419 )
- [SofaBaseLinearSolver] CRS explicit instantiation [#2306]( https://github.com/sofa-framework/sofa/pull/2306 )
- [SofaBaseLinearSolver] Fix compilation when enabling CRSMultiMatrixAccessor [#2456]( https://github.com/sofa-framework/sofa/pull/2456 )
- [SofaBaseLinearSolver] Fix mulTranspose for scalar types [#2331]( https://github.com/sofa-framework/sofa/pull/2331 )
- [SofaBaseLinearSolver] Link is not overwritten [#2497]( https://github.com/sofa-framework/sofa/pull/2497 )
- [SofaBaseMechanics] Fix UniformMass topology changes handling mode. [#2377]( https://github.com/sofa-framework/sofa/pull/2377 )
- [SofaBaseMechanics] Fix compilation in BarycentricMapping [#2443]( https://github.com/sofa-framework/sofa/pull/2443 )
- [SofaBaseMechanics] Update MechanicalObject resize method to exit if not necessary [#1977]( https://github.com/sofa-framework/sofa/pull/1977 )
- [SofaBaseTopology] Add missing export symbol keyword for TopologySubsetData [#2247]( https://github.com/sofa-framework/sofa/pull/2247 )
- [SofaBaseTopology] Fix TopologySubsetData: call to creation/desctruction callback was missing [#2374]( https://github.com/sofa-framework/sofa/pull/2374 )
- [SofaBaseTopology] use WriteOnlyAccessor in TopologyData [#2414]( https://github.com/sofa-framework/sofa/pull/2414 )
- [SofaBaseVisual] ADD Update callbacks in VisualModelImpl (#1483) [#2245]( https://github.com/sofa-framework/sofa/pull/2245 )
- [SofaBaseVisual] Fix VisualModelImpl updateTextures callback to not call init method [#2298]( https://github.com/sofa-framework/sofa/pull/2298 )
- [SofaBaseVisual] Fix assert [#2417]( https://github.com/sofa-framework/sofa/pull/2417 )
- [SofaBoundaryCondition] Fix ProjectToLineConstraint_RemovingMeshTest.scn [#2241]( https://github.com/sofa-framework/sofa/pull/2241 )
- [SofaBoundaryCondition] Fix constraints in assembled systems [#2269]( https://github.com/sofa-framework/sofa/pull/2269 )
- [SofaConstraint] Fix BilateralInteractionConstraint's draw() [#2530]( https://github.com/sofa-framework/sofa/pull/2530 )
- [SofaConstraint] Fix crash when constraint correction is missing [#2222]( https://github.com/sofa-framework/sofa/pull/2222 )
- [SofaConstraint] fix segfault in GenericConstraintSolver  [#2265]( https://github.com/sofa-framework/sofa/pull/2265 )
- [SofaCore] Fix potential memory failure on TopologyData Add method  [#2459]( https://github.com/sofa-framework/sofa/pull/2459 )
- [SofaCore] FIX BaseData::getLinkPath() [#2354]( https://github.com/sofa-framework/sofa/pull/2354 )
- [SofaCore] Link: case where Data parent is invalid [#2211]( https://github.com/sofa-framework/sofa/pull/2211 )
- [SofaCore_simutest] Fix getobjects and testcomponentstate unit tests [#2326]( https://github.com/sofa-framework/sofa/pull/2326 )
- [SofaDeformable] Forgot minus sign in #2429 [#2448]( https://github.com/sofa-framework/sofa/pull/2448 )
- [SofaGUICommon] Fix configuration file for install with cxxopts [#2279]( https://github.com/sofa-framework/sofa/pull/2279 )
- [SofaGeneralEngine] Fix a typo in the PairBoxRoi.inl [#2324]( https://github.com/sofa-framework/sofa/pull/2324 )
- [SofaGeneralLoader] Make GridMeshCreator work again [#2473]( https://github.com/sofa-framework/sofa/pull/2473 )
- [SofaGuiQt] Fix graph update on startup [#2458]( https://github.com/sofa-framework/sofa/pull/2458 )
- [SofaGuiQt] Revert #2458 [#2479]( https://github.com/sofa-framework/sofa/pull/2479 )
- [SofaGuiQt] Some fixes for Qt6 [#2490]( https://github.com/sofa-framework/sofa/pull/2490 )
- [SofaGuiQt] Switch name and class name for slaves [#2371]( https://github.com/sofa-framework/sofa/pull/2371 )
- [SofaKernel] Remove ambiguous constructors from type::vector [#2270]( https://github.com/sofa-framework/sofa/pull/2270 )
- [SofaKernel] Remove the getXXXAccessor in accessor.h to keep the one in Data.h [#2278]( https://github.com/sofa-framework/sofa/pull/2278 )
- [SofaMeshCollision, SofaConstraint] Remove LMDNewProximityIntersection [#2272]( https://github.com/sofa-framework/sofa/pull/2272 )
- [SofaMeshCollision] Fix cmake generated config file [#2400]( https://github.com/sofa-framework/sofa/pull/2400 )
- [SofaMeshCollision] Fix compilation without Compatiblity layer [#2227]( https://github.com/sofa-framework/sofa/pull/2227 )
- [SofaMiscCollision] Fix the BarycentricStickContact response [#2509]( https://github.com/sofa-framework/sofa/pull/2509 )
- [SofaMiscForceField] Fix MeshMatrixMass init and topologicalChanges [#2193]( https://github.com/sofa-framework/sofa/pull/2193 )
- [SofaMiscForceField] Fix addForce function in MeshMatrixMass [#2305]( https://github.com/sofa-framework/sofa/pull/2305 )
- [SofaSimpleFem] Check topology in draw [#2478]( https://github.com/sofa-framework/sofa/pull/2478 )
- [SofaSimpleFem] Fix row/col indices in hexa fem for bloc-based matrices [#2277]( https://github.com/sofa-framework/sofa/pull/2277 )
- [SofaSimpleFem] Set valid component state [#2499]( https://github.com/sofa-framework/sofa/pull/2499 )
- [SofaTopologyMapping] Fix Tetra2TriangleTopologicalMapping lost ancestor info [#2460]( https://github.com/sofa-framework/sofa/pull/2460 )
- [SofaUserInteraction] Fix InteractionPerformerFactory symbol export [#2495]( https://github.com/sofa-framework/sofa/pull/2495 )

**Plugins / Projects**
- [CGALPlugin] FIX sofa::type [#2246]( https://github.com/sofa-framework/sofa/pull/2246 )
- [CGALPlugin] Fix CGAL compilation with CImgData include [#2345]( https://github.com/sofa-framework/sofa/pull/2345 )
- [CImgPlugin] Add Threads dependency in Cmake [#2302]( https://github.com/sofa-framework/sofa/pull/2302 )
- [Geomagic] Fix compilation of code with openHaptics due to sofa::type namespace missing. [#2229]( https://github.com/sofa-framework/sofa/pull/2229 )
- [image] Ignore python2 scenes in CI [#2526]( https://github.com/sofa-framework/sofa/pull/2526 )
- [image] fix shadow variable [#2515]( https://github.com/sofa-framework/sofa/pull/2515 )
- [image] fix shadow variables [#2528]( https://github.com/sofa-framework/sofa/pull/2528 )
- [SofaCUDA] Fix broken visuals in examples [#2447]( https://github.com/sofa-framework/sofa/pull/2447 )
- [SofaCUDA] Fix compilation [#2446]( https://github.com/sofa-framework/sofa/pull/2446 )
- [SofaCUDA] Fix static variable definition for double-precision [#2440]( https://github.com/sofa-framework/sofa/pull/2440 )
- [SofaCUDA] Redundant namespace [#2538]( https://github.com/sofa-framework/sofa/pull/2538 )
- [SofaGuiGlut] FIX compilation for v21.06 [#2274]( https://github.com/sofa-framework/sofa/pull/2274 )
- [SofaGuiGlut] Fix compilation for v21.12 [#2539]( https://github.com/sofa-framework/sofa/pull/2539 )

**Examples / Scenes**
- [examples] FIX duplicated scene in regression-tests [#2251]( https://github.com/sofa-framework/sofa/pull/2251 )
- [examples] FIX duplicated scene in regression-tests (2) [#2252]( https://github.com/sofa-framework/sofa/pull/2252 )

**Scripts / Tools**

### Cleanings

**Architecture**
- [CMake] Remove option SOFA_ENABLE_SOFT_DEPS_TO_SOFAPYTHON [#2533]( https://github.com/sofa-framework/sofa/pull/2533 )
- [GitHub] Improve "bug report" issue template [#2454]( https://github.com/sofa-framework/sofa/pull/2454 )

**Modules**
- [All] Clean warnings [#1549]( https://github.com/sofa-framework/sofa/pull/1549 )
- [All] Remove several TopologyDataHandler and headers inclusion in components [#2162]( https://github.com/sofa-framework/sofa/pull/2162 )
- [All] Remove warnings [#2378]( https://github.com/sofa-framework/sofa/pull/2378 )
- [All] Update codebase to compile w/o Sofa.Compat for v21.12 [#2525]( https://github.com/sofa-framework/sofa/pull/2525 )
- [All] Update lifecycle (macros, etc) for v21.12 [#2524]( https://github.com/sofa-framework/sofa/pull/2524 )
- [All] Clean unreferenced files [#2333]( https://github.com/sofa-framework/sofa/pull/2333 )
- [All] Declaration hides previous local declaration [#2463]( https://github.com/sofa-framework/sofa/pull/2463 )
- [All] Expression is converted to bool and can be replaced [#2464]( https://github.com/sofa-framework/sofa/pull/2464 )
- [All] Minor cleaning [#2461]( https://github.com/sofa-framework/sofa/pull/2461 )
- [All] Minor code cleaning [#2225]( https://github.com/sofa-framework/sofa/pull/2225 )
- [All] Variable can be made constexpr [#2472]( https://github.com/sofa-framework/sofa/pull/2472 )
- [All] fix some warnings and remove unnecessary includes [#2312]( https://github.com/sofa-framework/sofa/pull/2312 )
- [All] Fix minor compilation warnings [#2233]( https://github.com/sofa-framework/sofa/pull/2233 )
- [Sofa.Core] More information when the same Data name is used multiple times [#2489]( https://github.com/sofa-framework/sofa/pull/2489 )
- [Sofa.Core] Remove unnecessary functions because they are in base class [#2430]( https://github.com/sofa-framework/sofa/pull/2430 )
- [Sofa.Helper] Remove duplicated code [#2482]( https://github.com/sofa-framework/sofa/pull/2482 )
- [Sofa.Helper] Remove usage of boost::filesystem [#2342]( https://github.com/sofa-framework/sofa/pull/2342 )
- [Sofa.SimulationCore] Wrong doxygen symbol [#2467]( https://github.com/sofa-framework/sofa/pull/2467 )
- [Sofa.Type] Clean and modernize Vec and Mat [#2282]( https://github.com/sofa-framework/sofa/pull/2282 )
- [Sofa.Type] Move LCPsolver class into a utility function [#2187]( https://github.com/sofa-framework/sofa/pull/2187 )
- [Sofa.Type] Speed up fixed_array constructors with perfect forwarding [#2450]( https://github.com/sofa-framework/sofa/pull/2450 )
- [Sofa2EigenSolver] move SVDLinearSolver to SofaDenseSolver and deprecate the module  [#2368]( https://github.com/sofa-framework/sofa/pull/2368 )
- [SofaBaseLinearSolver] Extract and uncomment CRSMultiMatrixAccessor [#2220]( https://github.com/sofa-framework/sofa/pull/2220 )
- [SofaBaseLinearSolver] Fix typo [#2256]( https://github.com/sofa-framework/sofa/pull/2256 )
- [SofaBaseLinearSolver] Make CRSMultiMatrixAccessor optional (compilation and usage) [#2372]( https://github.com/sofa-framework/sofa/pull/2372 )
- [SofaBaseLinearSolver] Remove FullMatrix<bool> and FullVector<bool> [#2313]( https://github.com/sofa-framework/sofa/pull/2313 )
- [SofaBaseMechanics] Remove TopologyHandler in masses to use TopologyData callbacks (part 5) [#2391]( https://github.com/sofa-framework/sofa/pull/2391 )
- [SofaBaseMechanics] Remove unused handleEvent in UniformMass [#2521]( https://github.com/sofa-framework/sofa/pull/2521 )
- [SofaBaseMechanics] Simplify expression [#2468]( https://github.com/sofa-framework/sofa/pull/2468 )
- [SofaBaseMechanics] Use directly clear() when resetting force in MechanicalObject [#2518]( https://github.com/sofa-framework/sofa/pull/2518 )
- [SofaBaseTopology] 'createTopologyHandler' overrides a member function but is not marked 'override' [#2260]( https://github.com/sofa-framework/sofa/pull/2260 )
- [SofaBaseVisual] Split VisualModelImpl init method in several methods for more clarity [#2299]( https://github.com/sofa-framework/sofa/pull/2299 )
- [SofaConstraint] Better includes [#2266]( https://github.com/sofa-framework/sofa/pull/2266 )
- [SofaConstraint] Divide a timer in 2 [#2469]( https://github.com/sofa-framework/sofa/pull/2469 )
- [SofaConstraint] Remove dependency on TetrahedronFEMForcefield [#2250]( https://github.com/sofa-framework/sofa/pull/2250 )
- [SofaCore] Clean force fields [#2243]( https://github.com/sofa-framework/sofa/pull/2243 )
- [SofaCore] Fix two determiners in a row [#2271]( https://github.com/sofa-framework/sofa/pull/2271 )
- [SofaCore] Remove unused addSubMBKToMatrix in force fields [#2244]( https://github.com/sofa-framework/sofa/pull/2244 )
- [SofaCore] Remove unwanted logs in TopologyHandler. [#2401]( https://github.com/sofa-framework/sofa/pull/2401 )
- [SofaDeformable] clean and optimize addKToMatrix [#2429]( https://github.com/sofa-framework/sofa/pull/2429 )
- [SofaDenseSolver][SofaBaseLinearSolver] CLEAN macros *_CHECK and *_VERBOSE [#2310]( https://github.com/sofa-framework/sofa/pull/2310 )
- [SofaEigen2Solver] Description + timers + support of any BaseMatrix [#2336]( https://github.com/sofa-framework/sofa/pull/2336 )
- [SofaExporter] Rename OBJExporter into VisualModelOBJExporter [#2505]( https://github.com/sofa-framework/sofa/pull/2505 )
- [SofaGeneralAnimationLoop, Sofa.SimulationCore] Extract MechanicalAccumulateJacobian [#2481]( https://github.com/sofa-framework/sofa/pull/2481 )
- [SofaGeneralAnimationLoop] MechanicalMatrixMapper timers [#2362]( https://github.com/sofa-framework/sofa/pull/2362 )
- [SofaGeneralDeformable] Remove TopologyHandler in FEM to use TopologyData callbacks (part 2) [#2388]( https://github.com/sofa-framework/sofa/pull/2388 )
- [SofaGeneralDeformable] Remove TopologyHandler in FEM to use TopologyData callbacks (part 3) [#2389]( https://github.com/sofa-framework/sofa/pull/2389 )
- [SofaGeneralEngine] Cleaning of MeshBoundaryROI [#2319]( https://github.com/sofa-framework/sofa/pull/2319 )
- [SofaGeneralSimpleFem] Add comments and tests for TriangularFEMForceFieldOptim [#2284]( https://github.com/sofa-framework/sofa/pull/2284 )
- [SofaGeneralSimpleFem] Remove TopologyHandler in FEM to use TopologyData callbacks (part 1) [#2384]( https://github.com/sofa-framework/sofa/pull/2384 )
- [SofaGuiGlut] Unreachable break [#2405]( https://github.com/sofa-framework/sofa/pull/2405 )
- [SofaGuiQt] Get rid of magic numbers when centering the window [#2466]( https://github.com/sofa-framework/sofa/pull/2466 )
- [SofaHelper] replace infinite loop Update PipeProcess.cpp [#2477]( https://github.com/sofa-framework/sofa/pull/2477 )
- [SofaHelper] Remove boost::thread dependency [#2264]( https://github.com/sofa-framework/sofa/pull/2264 )
- [SofaHelper_test] Remove temporary file in FileMonitor_test [#2537]( https://github.com/sofa-framework/sofa/pull/2537 )
- [SofaKernel] change data content copy on write condition [#2285]( https://github.com/sofa-framework/sofa/pull/2285 )
- [SofaMiscFEM] Small optimizations on TriangularFEMForceField (speedup ~x1.6) [#2273]( https://github.com/sofa-framework/sofa/pull/2273 )
- [SofaMiscFem] Minor homogeneization in TriangleFEMFF [#2408]( https://github.com/sofa-framework/sofa/pull/2408 )
- [SofaMiscFem] Remove TopologyHandler in FEM to use TopologyData callbacks (part 4) [#2390]( https://github.com/sofa-framework/sofa/pull/2390 )
- [SofaMiscFem] Remove debug code [#2420]( https://github.com/sofa-framework/sofa/pull/2420 )
- [SofaMiscFem] Remove unnecessary copy/pasted code [#2421]( https://github.com/sofa-framework/sofa/pull/2421 )
- [SofaMiscForcefield] Small optimization in addMdx in MeshMatrixMass [#2516]( https://github.com/sofa-framework/sofa/pull/2516 )
- [SofaMiscFem][SofaNonUniformFem] Remove newmat usage [#2531]( https://github.com/sofa-framework/sofa/pull/2531 )
- [SofaPreconditioner] Cleaning [#2493]( https://github.com/sofa-framework/sofa/pull/2493 )
- [SofaSimpleFem] Move duplicated code into a function [#2231]( https://github.com/sofa-framework/sofa/pull/2231 )
- [SofaSimpleFem] Remove branches based on type of matrix [#2323]( https://github.com/sofa-framework/sofa/pull/2323 )
- [SofaSparseSolver] Clean examples scenes of sparse linear solvers [#2422]( https://github.com/sofa-framework/sofa/pull/2422 )
- [SofaTopologyMapping] Fix shadowed variable #2158 [#2413]( https://github.com/sofa-framework/sofa/pull/2413 )
- [Tests] Update SceneCreator_test to inherit from BaseSimulationTest and remove some warnings [#2406]( https://github.com/sofa-framework/sofa/pull/2406 )

**Plugins / Projects**
- [Plugins] Move Newmat matters into the new SofaNewmat plugin [#2532]( https://github.com/sofa-framework/sofa/pull/2532 )
- [CImgPlugin] Minimize plugin dependencies [#2318]( https://github.com/sofa-framework/sofa/pull/2318 )
- [image] fix shadow variable #2432 [#2437]( https://github.com/sofa-framework/sofa/pull/2437 )
- [image] Remove useless DiffusionSolver dependency [#2308]( https://github.com/sofa-framework/sofa/pull/2308 )
- [image] Clean init members warning [#2536]( https://github.com/sofa-framework/sofa/pull/2536 )
- [SofaCUDA] reorder CMakeLists just for more clarity [#2534]( https://github.com/sofa-framework/sofa/pull/2534 )

**Examples / Scenes**
- [examples] Add handleDynamicTopology in OglModel with dynamic texcoords [#2445]( https://github.com/sofa-framework/sofa/pull/2445 )
- [examples] Clean linear solver scenes [#2494]( https://github.com/sofa-framework/sofa/pull/2494 )
- [examples] Limit GlobalSystemMatrixExporter.scn to 1 iteration [#2328]( https://github.com/sofa-framework/sofa/pull/2328 )
- [examples] Make scene resolution independent + doc [#2361]( https://github.com/sofa-framework/sofa/pull/2361 )
- [examples] add a warning comment in the scene header [#2470]( https://github.com/sofa-framework/sofa/pull/2470 )
- [scenes] Replace BruteForceDetection by BruteForceNarrowPhase+BVHNarrowPhase [#2232]( https://github.com/sofa-framework/sofa/pull/2232 )

**Scripts / Tools**


____________________________________________________________



## [v21.06.03]( https://github.com/sofa-framework/sofa/tree/v21.06.03 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v21.06.02...v21.06.03 )

### Bug Fixes
**Architecture**
-  [CMake][modules] FIX out-of-tree builds [#2453]( https://github.com/sofa-framework/sofa/pull/2453 )
-  [All] Update IDE directory for targets [#2239]( https://github.com/sofa-framework/sofa/pull/2239 )

**Modules**
- [SofaGeneralLoader] Make GridMeshCreator work again [#2473]( https://github.com/sofa-framework/sofa/pull/2473 )


____________________________________________________________



## [v21.06.02]( https://github.com/sofa-framework/sofa/tree/v21.06.02 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v21.06.01...v21.06.02 )

### Bug Fixes
**Architecture**
-  [CMake] Clean Sofa.Core CMakeLists + Add missing headers [#2444]( https://github.com/sofa-framework/sofa/pull/2444 )
-  [SofaMeshCollision] Fix cmake generated config file [#2400]( https://github.com/sofa-framework/sofa/pull/2400 )

**Modules**
- [Sofa.Compat] Fix install [#2360]( https://github.com/sofa-framework/sofa/pull/2360 )
- [Sofa.Helper] PluginManager: check if a plugin is already loaded with a different path [#2352]( https://github.com/sofa-framework/sofa/pull/2352 )


____________________________________________________________



## [v21.06.01]( https://github.com/sofa-framework/sofa/tree/v21.06.01 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v21.06.00...v21.06.01 )

### Improvements
**Architecture**
- [CMake] CLEAN and reenable old macro for git infos [#2309]( https://github.com/sofa-framework/sofa/pull/2309 )
- [SofaMacros] Improve sofa_install_git_infos [#2373]( https://github.com/sofa-framework/sofa/pull/2373 )

### Bug Fixes
**Architecture**
- [CMake] Fix return values of Qt5/Qt6 find_packages [#2346]( https://github.com/sofa-framework/sofa/pull/2346 )

**Modules**
- [Sofa.Type] FIX createFromRotationVEctor, the inequality was wrong [#2332]( https://github.com/sofa-framework/sofa/pull/2332 )
- [SofaBoundaryCondition] Fix ProjectToLineConstraint_RemovingMeshTest.scn [#2241]( https://github.com/sofa-framework/sofa/pull/2241 )
- [SofaConstraint] fix segfault in GenericConstraintSolver [#2265]( https://github.com/sofa-framework/sofa/pull/2265 )
- [SofaGeneralEngine] Fix a typo in the PairBoxRoi.inl [#2324]( https://github.com/sofa-framework/sofa/pull/2324 )

**Plugins / Projects**
- [Geomagic] Fix compilation of code with openHaptics due to sofa::type namespace missing. [#2229]( https://github.com/sofa-framework/sofa/pull/2229 )
- [SofaGuiGlut] FIX compilation for v21.06 [#2274]( https://github.com/sofa-framework/sofa/pull/2274 )

### Cleanings
**Modules**
- [All] Minor code cleaning [#2225]( https://github.com/sofa-framework/sofa/pull/2225 )


____________________________________________________________



## [v21.06.00]( https://github.com/sofa-framework/sofa/tree/v21.06.00 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v20.12.03...v21.06.00 )

### SOFA-NG
**Architecture**
- [CMake] Macro replace dot with underscore for preprocessor defines [#1701]( https://github.com/sofa-framework/sofa/pull/1701 )

**Modules**
- **[Sofa.Type]** Remove stdtype subdirectory [#1946]( https://github.com/sofa-framework/sofa/pull/1946 )
- **[SofaBaseCollision]** Move OBB/Capsule-related intersections/detections/contacts to SofaMiscCollision [#2073]( https://github.com/sofa-framework/sofa/pull/2073 )
- **[SofaBaseCollision]** Remove dependencies on BaseIntTool [#2081]( https://github.com/sofa-framework/sofa/pull/2081 )
- **[SofaCore]** Clean dependencies to SimulationCore [#1928]( https://github.com/sofa-framework/sofa/pull/1928 )
- **[SofaFramework]** Create Sofa.Config module [#1674]( https://github.com/sofa-framework/sofa/pull/1674 )
- **[SofaFramework]** Create Sofa.Testing module [#1834]( https://github.com/sofa-framework/sofa/pull/1834 )
- **[SofaFramework]** Isolate OpenGL code into a single module (Sofa.GL) [#1649]( https://github.com/sofa-framework/sofa/pull/1649 )
- **[SofaFramework]** Rename modules (cmake) [#2001]( https://github.com/sofa-framework/sofa/pull/2001 )
- [SofaGui] Package SofaGui [#1719]( https://github.com/sofa-framework/sofa/pull/1719 )
- **[SofaHelper]** Clean dependencies to defaulttype [#1915]( https://github.com/sofa-framework/sofa/pull/1915 )
- **[SofaHelper]** Remove some SofaCore dependencies [#1810]( https://github.com/sofa-framework/sofa/pull/1810 )
- **[SofaKernel]** Create Sofa.Geometry and Sofa.Topology modules [#1912]( https://github.com/sofa-framework/sofa/pull/1912 )
- **[SofaKernel]** Create Sofa.Type module [#1555]( https://github.com/sofa-framework/sofa/pull/1555 )
- **[SofaKernel]** Package all modules of SofaFramework [#1990]( https://github.com/sofa-framework/sofa/pull/1990 )
- **[SofaKernel]** Port Mat from Defaulttype to Sofa.Type [#1775]( https://github.com/sofa-framework/sofa/pull/1775 )
- **[SofaKernel]** Port Quat from Defaulttype to Sofa.Type [#1790]( https://github.com/sofa-framework/sofa/pull/1790 )
- **[SofaKernel]** Port Vec from Defaulttype to Sofa.Type (bis) [#1734]( https://github.com/sofa-framework/sofa/pull/1734 )
- **[SofaKernel]** Port a bunch of types from Helper and Defaulttype to Sofa.Type [#1818]( https://github.com/sofa-framework/sofa/pull/1818 )
- **[SofaKernel]** Port vector (and its siblings) from Helper to Sofa.Type [#1893]( https://github.com/sofa-framework/sofa/pull/1893 )
- **[SofaKernel]** Regroup all compatibility files in a Sofa.Compat module [#1944]( https://github.com/sofa-framework/sofa/pull/1944 )
- **[SofaSimulation]** Package SofaSimulation [#1694]( https://github.com/sofa-framework/sofa/pull/1694 )
- [SofaUserInteraction] Refactor PickParticlePerformer [#2084]( https://github.com/sofa-framework/sofa/pull/2084 )

**Plugins / Projects**
- [SofaPython] Move python-related files to SofaPython [#1887]( https://github.com/sofa-framework/sofa/pull/1887 )
- [SofaPython] Regroup dependencies [#1875]( https://github.com/sofa-framework/sofa/pull/1875 )


### Breaking
**Modules**
- [All] Add forward declaration and opaque API for ExecParams, MechanicalParams, VisualParams and ConstraintParams [#1794]( https://github.com/sofa-framework/sofa/pull/1794 )
- [All] Remove SOFA_NO_OPENGL (not the cmake option) [#1888]( https://github.com/sofa-framework/sofa/pull/1888 )
- [All] Removes search & searchAll from BaseObject [#1842]( https://github.com/sofa-framework/sofa/pull/1842 )
- [All] ADD forward declaration headers + remove unneeded includes [#1764]( https://github.com/sofa-framework/sofa/pull/1764 )
- [All] Adds forward declaration and opaque api for the base types (the one used in Node.h) [#1765]( https://github.com/sofa-framework/sofa/pull/1765 )
- [All] Clean MechanicalMatrixVisitor [#1992]( https://github.com/sofa-framework/sofa/pull/1992 )
- [All] Data filename in BaseLoader is now using d_ [#2095]( https://github.com/sofa-framework/sofa/pull/2095 )
- [All] Move BaseMechanicalVisitor in its own files  [#1989]( https://github.com/sofa-framework/sofa/pull/1989 )
- [All] Move ConstraintResolution in its own file instead of BaseConstraint.h [#1847]( https://github.com/sofa-framework/sofa/pull/1847 )
- [All] Move ScopedAdvancedTimer in its own files (.h & .cpp) [#1853]( https://github.com/sofa-framework/sofa/pull/1853 )
- [All] Remove un-needed includes. [#1730]( https://github.com/sofa-framework/sofa/pull/1730 )
- [All] Clean lifecycle for v21.06 [#2198]( https://github.com/sofa-framework/sofa/pull/2198 )
- **[Sofa.Helper]** Remove atomic.h (and fix compilation on Silicon M1) [#2160]( https://github.com/sofa-framework/sofa/pull/2160 )
- **[Sofa.Type]** Clean Quat [#1878]( https://github.com/sofa-framework/sofa/pull/1878 )
- **[SofaBaseCollision]** REFACTOR BruteForceDetection [#1999]( https://github.com/sofa-framework/sofa/pull/1999 )
- **[SofaBaseLinearSolver]** Cleaning in FullMatrix & FullVector & MechanicalState [#1792]( https://github.com/sofa-framework/sofa/pull/1792 )
- **[SofaBaseLinearSolver]** Document and clean CGLinearSolver  [#2098]( https://github.com/sofa-framework/sofa/pull/2098 )
- **[SofaBaseMechanics]** Restore useRestPos Data initialization for BarycentricMapping [#1939]( https://github.com/sofa-framework/sofa/pull/1939 )
- **[SofaBaseTopology]** Clean RenumberPoints methods [#1840]( https://github.com/sofa-framework/sofa/pull/1840 )
- **[SofaBaseTopology]** Remove getter to some Data in Topology container and put them public [#1947]( https://github.com/sofa-framework/sofa/pull/1947 )
- **[SofaBaseTopology]** Remove public access to propagateTopologyChanges [#1860]( https://github.com/sofa-framework/sofa/pull/1860 )
- **[SofaBaseTopology]** Rename TopologyEngine and TopologyData classes to match file names [#1872]( https://github.com/sofa-framework/sofa/pull/1872 )
- **[SofaBaseTopology]** Secure All Topology modifiers API [#1859]( https://github.com/sofa-framework/sofa/pull/1859 )
- **[SofaBaseTopology]** Secure PointSetTopologyModifier API [#1858]( https://github.com/sofa-framework/sofa/pull/1858 )
- **[SofaBaseTopology]** Totally remove topologyEngine and move mechanism only in TopologyData and TopologyHandler [#1898]( https://github.com/sofa-framework/sofa/pull/1898 )
- **[SofaCore]** Base::findLinkDest returns Base* instead of void* [#1700]( https://github.com/sofa-framework/sofa/pull/1700 )
- **[SofaCore]** FIX SingleLink clear/set behavior [#1749]( https://github.com/sofa-framework/sofa/pull/1749 )
- **[SofaCore]** Move definition of Link::updateLinks to BaseLink.h [#1735]( https://github.com/sofa-framework/sofa/pull/1735 )
- **[SofaCore]** Moves MechanicaMatrix out of MultiMatrix.h [#1870]( https://github.com/sofa-framework/sofa/pull/1870 )
- **[SofaCore]** Remove "depend" from Node [#1763]( https://github.com/sofa-framework/sofa/pull/1763 )
- **[SofaCore]** isDiagonal is const [#1903]( https://github.com/sofa-framework/sofa/pull/1903 )
- [SofaExplicitOdeSolver] Introduce visitor to know the number of non-diagonal mass matrices [#2165]( https://github.com/sofa-framework/sofa/pull/2165 )
- [SofaGeneralSimpleFem] Remove \*Containers [#2099]( https://github.com/sofa-framework/sofa/pull/2099 )
- **[SofaHelper]** Add two search paths for every prefixes of the plugin manager [#1824]( https://github.com/sofa-framework/sofa/pull/1824 )
- **[SofaHelper]** Move polygon cube intersection ad-hoc algorithm to SofaBaseTopology [#1772]( https://github.com/sofa-framework/sofa/pull/1772 )
- **[SofaHelper]** Remove SofaSimulationCore dependency from AdvancedTimer [#1770]( https://github.com/sofa-framework/sofa/pull/1770 )
- **[SofaHelper]** Replace boost::shared_ptr for std::shared_ptr [#1901]( https://github.com/sofa-framework/sofa/pull/1901 )
- **[SofaHelper]** remove stream operator<< in accessor [#1808]( https://github.com/sofa-framework/sofa/pull/1808 )
- [SofaMiscCollision] Clean options [#2170]( https://github.com/sofa-framework/sofa/pull/2170 )
- [SofaMiscCollision] Fix multiple bugs in group manager [#2076]( https://github.com/sofa-framework/sofa/pull/2076 )
- [SofaMiscCollision] Move OBB/Capsule (and related Intr* Code) [#2168]( https://github.com/sofa-framework/sofa/pull/2168 )
- [SofaMiscMapping] Factorize code to remove duplicated struct [#1957]( https://github.com/sofa-framework/sofa/pull/1957 )
- **[SofaSimulationCore]** Move CpuTask class into its own file [#1993]( https://github.com/sofa-framework/sofa/pull/1993 )
- **[SofaSimulationCore]** Simplify repetitive code in BaseMechanicalVisitor [#2125]( https://github.com/sofa-framework/sofa/pull/2125 )
- **[SofaSimulationGraph]** Remove dead-end experiment in SimpleApi.h [#1845]( https://github.com/sofa-framework/sofa/pull/1845 )
- [SofaTopologyMapping] adds ellipse feature to Edge2QuadTopologyMapping [#1861]( https://github.com/sofa-framework/sofa/pull/1861 )

**Plugins / Projects**
- [CGALPlugin] new features [#2124]( https://github.com/sofa-framework/sofa/pull/2124 )


### Improvements
**Architecture**
- [CMake] Speedup configure step [#1837]( https://github.com/sofa-framework/sofa/pull/1837 )
- [CMake] ADD option to fetch pull-request commits in ExternalProject [#1961]( https://github.com/sofa-framework/sofa/pull/1961 )
- [CMake] ADD option to enable/disable compatibility layer [#2216]( https://github.com/sofa-framework/sofa/pull/2216 )

**Modules**
- **[SofaBaseCollision]** BVH narrow phase [#2043]( https://github.com/sofa-framework/sofa/pull/2043 )
- **[SofaBaseCollision]** Introduce brute force broad phase [#2010]( https://github.com/sofa-framework/sofa/pull/2010 )
- **[SofaBaseCollision]** Speedup direct SAP [#1917]( https://github.com/sofa-framework/sofa/pull/1917 )
- **[SofaBaseMechanics_test]** Add TopologicalChanges unit tests for DiagonalMass [#2176]( https://github.com/sofa-framework/sofa/pull/2176 )
- **[SofaBaseTopology]** add intersection methods [#2131]( https://github.com/sofa-framework/sofa/pull/2131 )
- [SofaBoundaryCondition] Add callback in Partial/FixedConstraint [#1806]( https://github.com/sofa-framework/sofa/pull/1806 )
- [SofaBoundaryCondition] Add projectVelocity in FixedConstraint and PartialFixedConstraint [#1558]( https://github.com/sofa-framework/sofa/pull/1558 )
- [SofaBoundaryCondition_test] Add TopologicalChanges unit test for FixedConstraint [#2136]( https://github.com/sofa-framework/sofa/pull/2136 )
- [SofaConstraint] GenericConstraintSolver: compute compliances concurrently [#1862]( https://github.com/sofa-framework/sofa/pull/1862 )
- [SofaConstraint] Parallel free motion and collision detection [#2040]( https://github.com/sofa-framework/sofa/pull/2040 )
- **[SofaCore]** Add PCH support in CMakeLists.txt [#1727]( https://github.com/sofa-framework/sofa/pull/1727 )
- [SofaExporter] Add option for obj export [#1789]( https://github.com/sofa-framework/sofa/pull/1789 )
- [SofaGeneralAnimationLoop] MechanicalMatrixMapper: adds template [#1703]( https://github.com/sofa-framework/sofa/pull/1703 )
- [SofaGeneralLoader] adds translation and rotation to ReadState [#1733]( https://github.com/sofa-framework/sofa/pull/1733 )
- [SofaGeneralMeshCollision] Direct SAP is written as a narrow phase [#2030]( https://github.com/sofa-framework/sofa/pull/2030 )
- [SofaGuiQt] Add new about window UI and url redirect [#1801]( https://github.com/sofa-framework/sofa/pull/1801 )
- [SofaGuiQt] Qt6 support [#1756]( https://github.com/sofa-framework/sofa/pull/1756 )
- [SofaGuiQt] Tooltips [#2139]( https://github.com/sofa-framework/sofa/pull/2139 )
- [SofaGui] Improve background v20.12 [#1758]( https://github.com/sofa-framework/sofa/pull/1758 )
- **[SofaHelper]** Optimize use of map_ptr_stable_compare [#2105]( https://github.com/sofa-framework/sofa/pull/2105 )
- **[SofaHelper][SofaPython]** ADD PluginManager callback and use it in SofaPython [#1777]( https://github.com/sofa-framework/sofa/pull/1777 )
- [SofaImplicitOdeSolver] Rewrite of the static Newton-Raphson ODE solver [#2050]( https://github.com/sofa-framework/sofa/pull/2050 )
- **[SofaKernel]** Write template list in error message [#2207]( https://github.com/sofa-framework/sofa/pull/2207 )
- [SofaMiscFem] Proposal of FEM force field for Reissner-Mindlin Flat Shell Element  [#1745]( https://github.com/sofa-framework/sofa/pull/1745 )
- [SofaMiscTopology] Add component TopologyChecker [#1594]( https://github.com/sofa-framework/sofa/pull/1594 )
- [SofaOpenglVisual]  Add transparency when we draw triangles (this allows to see inside the volume). [#1742]( https://github.com/sofa-framework/sofa/pull/1742 )
- **[SofaSimulationCore]** Add option to call ODE::solve in parallel in SolveVisitor [#2135]( https://github.com/sofa-framework/sofa/pull/2135 )
- **[SofaSimulationCore]** Launch a new event when textures have been initialized. [#1832]( https://github.com/sofa-framework/sofa/pull/1832 )

**Plugins / Projects**
- [MultiThreading] Parallel BVH narrow phase [#2053]( https://github.com/sofa-framework/sofa/pull/2053 )
- [MultiThreading] Parallel brute force broad phase [#2038]( https://github.com/sofa-framework/sofa/pull/2038 )
- [Plugins] Add SofaGLFW GUI [#2062]( https://github.com/sofa-framework/sofa/pull/2062 )
- [SofaCUDA] Add more examples for liver scene and squareTissue [#2016]( https://github.com/sofa-framework/sofa/pull/2016 )

**Examples / Scenes**
- [Benchmark] Introduce benchmark on matrix assembly [#2208]( https://github.com/sofa-framework/sofa/pull/2208 )
- [examples] Improve BeamFEMForceField example [#2092]( https://github.com/sofa-framework/sofa/pull/2092 )
- [scenes] Add scenes to test several components during mesh removal [#2161]( https://github.com/sofa-framework/sofa/pull/2161 )


### Bug Fixes
**Architecture**
- [CMake] FIX Debug build + more cleaning [#1891]( https://github.com/sofa-framework/sofa/pull/1891 )
- [CMake] FIX SofaFramework aliases [#2175]( https://github.com/sofa-framework/sofa/pull/2175 )
- [CMake] FIX header include tree [#1863]( https://github.com/sofa-framework/sofa/pull/1863 )
- [CMake] FIX resources and translations install on Windows [#1949]( https://github.com/sofa-framework/sofa/pull/1949 )
- [CMake] Set CMake install default component [#2037]( https://github.com/sofa-framework/sofa/pull/2037 )
- [CMake][tools] v20.12.00 changes [#1804]( https://github.com/sofa-framework/sofa/pull/1804 )

**Extlibs**
- [GTest] Fix with GCC11 [#2181]( https://github.com/sofa-framework/sofa/pull/2181 )

**Modules**
- [All] FIX build without SofaPython soft dependencies [#1829]( https://github.com/sofa-framework/sofa/pull/1829 )
- [All] FIX warnings during STC#11 [#2140]( https://github.com/sofa-framework/sofa/pull/2140 )
- [All] Fix compilation warning. [#1699]( https://github.com/sofa-framework/sofa/pull/1699 )
- [All] Fix invalid Memory access in MechanicalObject and Compliant_test [#1849]( https://github.com/sofa-framework/sofa/pull/1849 )
- [All] Fix debug compilation [#2223]( https://github.com/sofa-framework/sofa/pull/2223 )
- [All] Changes needed for v21.06 [#2226]( https://github.com/sofa-framework/sofa/pull/2226 )
- **[Sofa.Type]** Add forgotten function declaration, used in Debug [#1937]( https://github.com/sofa-framework/sofa/pull/1937 )
- **[Sofa.Type]** Add missing header in fixed_array [#2006]( https://github.com/sofa-framework/sofa/pull/2006 )
- **[SofaBaseCollision]** Fix pipeline warning test [#2132]( https://github.com/sofa-framework/sofa/pull/2132 )
- **[SofaBaseCollision]** Clean code related to intersection methods [#2178]( https://github.com/sofa-framework/sofa/pull/2178 )
- **[SofaBaseMechanics]** Clean DiagonalMass init  [#2186]( https://github.com/sofa-framework/sofa/pull/2186 )
- **[SofaBaseMechanics]** Restore tests commented by mistake [#2104]( https://github.com/sofa-framework/sofa/pull/2104 )
- **[SofaBaseMechanics]** fixes reinit of BarycentricMapping [#1815]( https://github.com/sofa-framework/sofa/pull/1815 )
- **[SofaBaseMechanics]** Fix bug when deleting point. Mass vector was not well recomputed. [#2164]( https://github.com/sofa-framework/sofa/pull/2164 )
- **[SofaBaseMechanics_test]** Add more checks in DiagonalMass_test [#2183]( https://github.com/sofa-framework/sofa/pull/2183 )
- **[SofaBaseTopology]** Add security in TopologyData to check input Topology pointer [#2027]( https://github.com/sofa-framework/sofa/pull/2027 )
- **[SofaBaseTopology]** Missing override keyword [#2035]( https://github.com/sofa-framework/sofa/pull/2035 )
- **[SofaBaseTopology]** correct the logic issue in computeEdgeSegmentIntersection [#2184]( https://github.com/sofa-framework/sofa/pull/2184 )
- [SofaConstraint] Fix symbol export of BilateralInteractionConstraint on RigidTypes [#2031]( https://github.com/sofa-framework/sofa/pull/2031 )
- **[SofaCore]** FIX Issue #1865 [#1866]( https://github.com/sofa-framework/sofa/pull/1866 )
- **[SofaCore]** FIX buffer overflow when using AddressSanitizer [#2121]( https://github.com/sofa-framework/sofa/pull/2121 )
- **[SofaCore]** Fix explicit instantiations with MultiMapping [#1982]( https://github.com/sofa-framework/sofa/pull/1982 )
- **[SofaCore]** Remove duplicated explicit instanciation [#1981]( https://github.com/sofa-framework/sofa/pull/1981 )
- **[SofaDefaultType][SofaCUDA]** FIX compilation errors [#1761]( https://github.com/sofa-framework/sofa/pull/1761 )
- [SofaDeformable] Fix test in StiffSpringForceField doUpdateInternal [#1979]( https://github.com/sofa-framework/sofa/pull/1979 )
- **[SofaFramework]** Fix config files for Sofa.GL, for out-of-tree [#1911]( https://github.com/sofa-framework/sofa/pull/1911 )
- **[SofaFramework]** Put back Sofa.GL in Framework [#1920]( https://github.com/sofa-framework/sofa/pull/1920 )
- [SofaGeneralEngine] fixes ExtrudeQuadsAndGenerateHexas [#1673]( https://github.com/sofa-framework/sofa/pull/1673 )
- [SofaGeneralExplicitOdeSolver] Fix typo in CentralDifferenceSolver description [#1894]( https://github.com/sofa-framework/sofa/pull/1894 )
- [SofaGraphComponent] Clean format of RequiredPlugin message [#2111]( https://github.com/sofa-framework/sofa/pull/2111 )
- [SofaGraphComponent] Fix a typo in the warning emited by the APIVersion component and add missing allowed versions. [#2103]( https://github.com/sofa-framework/sofa/pull/2103 )
- [SofaGraphComponent] Fix message for RequiredPlugin [#2093]( https://github.com/sofa-framework/sofa/pull/2093 )
- [SofaGuiCommon] FIX build with SOFA_NO_OPENGL [#1724]( https://github.com/sofa-framework/sofa/pull/1724 )
- [SofaGuiCommon] Fix computationTimeSampling [#1698]( https://github.com/sofa-framework/sofa/pull/1698 )
- [SofaGuiCommon] Fix include path for compat files when installing [#1892]( https://github.com/sofa-framework/sofa/pull/1892 )
- [SofaGuiCommon] Remove unneeded include in PickHandler [#1707]( https://github.com/sofa-framework/sofa/pull/1707 )
- [SofaGuiCommon] Write json file [#2018]( https://github.com/sofa-framework/sofa/pull/2018 )
- [SofaGuiCommon] fix BackgroundSetting [#1826]( https://github.com/sofa-framework/sofa/pull/1826 )
- [SofaGuiQt] ADD qt.conf.h for custom qt.conf loading at runtime [#1820]( https://github.com/sofa-framework/sofa/pull/1820 )
- [SofaGuiQt] Clean QtGLViewer with key events [#1850]( https://github.com/sofa-framework/sofa/pull/1850 )
- [SofaGuiQt] FIX "show grid" (G) shortcut for QGLViewer [#2151]( https://github.com/sofa-framework/sofa/pull/2151 )
- [SofaGuiQt] FIX segfault due to qFatal in GenGraphForm [#1948]( https://github.com/sofa-framework/sofa/pull/1948 )
- [SofaGuiQt] Fix Expand node [#2069]( https://github.com/sofa-framework/sofa/pull/2069 )
- [SofaGuiQt] Fix Export Graph option [#1785]( https://github.com/sofa-framework/sofa/pull/1785 )
- [SofaGuiQt] Fix RealGUI: QDesktopWidget header missing for Qt < 5.11 [#1904]( https://github.com/sofa-framework/sofa/pull/1904 )
- [SofaGuiQt] Fix compat file SofaGuiQt.h [#2044]( https://github.com/sofa-framework/sofa/pull/2044 )
- [SofaGuiQt] Fix compilation [#1889]( https://github.com/sofa-framework/sofa/pull/1889 )
- [SofaGuiQt] Fix compilation when SOFA_DUMP_VISITOR is enabled [#1914]( https://github.com/sofa-framework/sofa/pull/1914 )
- [SofaGuiQt] Use opaque API instead [#1918]( https://github.com/sofa-framework/sofa/pull/1918 )
- **[SofaHelper]** FIX "name vs path" issue in PluginManager + FIX PluginManager_test [#1823]( https://github.com/sofa-framework/sofa/pull/1823 )
- **[SofaHelper]** FIX "name vs path" issue in PluginManager + FIX PluginManager_test (2) [#1825]( https://github.com/sofa-framework/sofa/pull/1825 )
- **[SofaHelper]** Fix AdvandedTimer test with end() [#1781]( https://github.com/sofa-framework/sofa/pull/1781 )
- **[SofaHelper]** Fix compilation in kdtree [#1942]( https://github.com/sofa-framework/sofa/pull/1942 )
- **[SofaHelper]** Fix out of bounds error [#1868]( https://github.com/sofa-framework/sofa/pull/1868 )
- **[SofaHelper]** Fix the use of Read/WriteAccessorVector that is too permisive (in accessor.h) [#1900]( https://github.com/sofa-framework/sofa/pull/1900 )
- **[SofaHelper][SofaBoundaryCondition]** Fix export keywords [#1984]( https://github.com/sofa-framework/sofa/pull/1984 )
- **[SofaKernel]** Fix SimpleApi forward declaration of BaseObject and relocatable of Sofa.GL [#1760]( https://github.com/sofa-framework/sofa/pull/1760 )
- **[SofaKernel]** Removing double load() in Loaders [#2094]( https://github.com/sofa-framework/sofa/pull/2094 )
- [SofaLoader] Fix MeshObjLoader material parsing by using the proper locale. [#2200]( https://github.com/sofa-framework/sofa/pull/2200 )
- [SofaLoader] FIX Circular dependency in the update of MeshObjLoader [#2201]( https://github.com/sofa-framework/sofa/pull/2201 )
- [SofaMacros] Shorter name for relocatable targets [#1769]( https://github.com/sofa-framework/sofa/pull/1769 )
- [SofaMeshCollision] Fix: windows debug linkage of class EmptyFilter [#1839]( https://github.com/sofa-framework/sofa/pull/1839 )
- [SofaMiscCollision] Contact response order [#2134]( https://github.com/sofa-framework/sofa/pull/2134 )
- [SofaMiscCollision] Fix config.in cmake file for export [#2052]( https://github.com/sofa-framework/sofa/pull/2052 )
- [SofaMiscFEM] Minor changes in TrianglePressureFF and TriangularFEMFF [#1779]( https://github.com/sofa-framework/sofa/pull/1779 )
- [SofaMiscForceField] Clean MeshMatrixMass_test and add more tests [#2191]( https://github.com/sofa-framework/sofa/pull/2191 )
- [SofaMiscTopology_test] Replace inheritence of SofaTest by BaseSimulation_test and fix test [#1909]( https://github.com/sofa-framework/sofa/pull/1909 )
- [SofaOpenGLVisual] Fix unreachable code in VisualManagerSecondaryPass [#2068]( https://github.com/sofa-framework/sofa/pull/2068 )
- [SofaOpenglVisual] Fix typo [#2029]( https://github.com/sofa-framework/sofa/pull/2029 )
- [SofaRigid] fixes applyJT of RigidMapping [#1813]( https://github.com/sofa-framework/sofa/pull/1813 )
- **[SofaSimpleFem]** Fix TetrahedronFEMForceField Von Mises stress drawing [#1854]( https://github.com/sofa-framework/sofa/pull/1854 )
- **[SofaSimpleFem]** Fix nasty bug in HexaFEMForceField' s draw() [#1766]( https://github.com/sofa-framework/sofa/pull/1766 )
- **[SofaSimpleFem]** Fix rendering options in TetrahedronFEMFF [#2156]( https://github.com/sofa-framework/sofa/pull/2156 )
- **[SofaSimulation/Tests]** Fix init/cleanup in SofaSimulation modules [#1987]( https://github.com/sofa-framework/sofa/pull/1987 )
- **[SofaSimulationCore]**  Set default  TaskScheduler worker threads to idle state [#1997]( https://github.com/sofa-framework/sofa/pull/1997 )
- **[SofaSimulationCore]** Broken URL [#2011]( https://github.com/sofa-framework/sofa/pull/2011 )
- **[SofaSimulationCore]** FIX Task scheduler memory leak [#1927]( https://github.com/sofa-framework/sofa/pull/1927 )
- **[SofaSimulationCore]** Reorder according to definition [#2034]( https://github.com/sofa-framework/sofa/pull/2034 )
- **[SofaSimulationCore]** Revert changes in #1927 [#2017]( https://github.com/sofa-framework/sofa/pull/2017 )
- [SofaSimulationGraph] Restore tests [#1988]( https://github.com/sofa-framework/sofa/pull/1988 )
- [SofaSimulationGraph] Reading links fail when owner is invalid [#2166]( https://github.com/sofa-framework/sofa/pull/2166 )
- [SofaSparseSolver] SparseLDLSolver optimizations [#1996]( https://github.com/sofa-framework/sofa/pull/1996 )
- [SofaTopologyMapping] Fix Edge2Quad condition [#2126]( https://github.com/sofa-framework/sofa/pull/2126 )
- [SofaUserInteraction] Fix Bug of removing topological element when a Hexa2TetraTopologicalMapping is in the scene [#1973]( https://github.com/sofa-framework/sofa/pull/1973 )

**Plugins / Projects**
- [CGALPlugin] Fix cgal drawings [#2206]( https://github.com/sofa-framework/sofa/pull/2206 )
- [CGALPlugin] Fix compilation errors related to SOFA.GL and types converted to DataTypes [#2065]( https://github.com/sofa-framework/sofa/pull/2065 )
- [CImgPlugin] Fix unit test [#2147]( https://github.com/sofa-framework/sofa/pull/2147 )
- [Geomagic] Fix Geomagic plugin compilation  [#2033]( https://github.com/sofa-framework/sofa/pull/2033 )
- [Geomagic] Fix GeomagicVisualModel compilation [#1776]( https://github.com/sofa-framework/sofa/pull/1776 )
- [Geomagic] Fix compilation of GeomagicVisualModel due to change in MechanicalVisitor [#2113]( https://github.com/sofa-framework/sofa/pull/2113 )
- [LMConstraint] Fix collision model [#1819]( https://github.com/sofa-framework/sofa/pull/1819 )
- [LMConstraint] Fix unload [#1831]( https://github.com/sofa-framework/sofa/pull/1831 )
- [PreassembledMass] FIX typo in PreassembledMass.inl [#1833]( https://github.com/sofa-framework/sofa/pull/1833 )
- [SofaCUDA] Fix CudaFixedConstraint when indices are not contiguous [#1780]( https://github.com/sofa-framework/sofa/pull/1780 )
- [SofaCUDA] Ignore example scenes using SofaCUDA on CI [#2042]( https://github.com/sofa-framework/sofa/pull/2042 )
- [SofaCUDA] Quick fix for SofaCUDA NVCC flags include not found during CMake setup [#2022]( https://github.com/sofa-framework/sofa/pull/2022 )
- [SofaCUDA] change the setTopology method signature [#1843]( https://github.com/sofa-framework/sofa/pull/1843 )
- [SofaComponentAll] FIX build with disabled dependencies [#1940]( https://github.com/sofa-framework/sofa/pull/1940 )
- [SofaOpenCL] Fix compilation [#1795]( https://github.com/sofa-framework/sofa/pull/1795 )
- [SofaOpenCL] Remove use of deleted functions [#1880]( https://github.com/sofa-framework/sofa/pull/1880 )
- [SofaPardisoSolver] Fix plugin and add example [#1830]( https://github.com/sofa-framework/sofa/pull/1830 )
- [SofaSphFluid] Fix ParticleSink init and refresh all examples. [#2026]( https://github.com/sofa-framework/sofa/pull/2026 )
- [Tutorials] Resurrect tutorials projects [#2024]( https://github.com/sofa-framework/sofa/pull/2024 )
- [VolumetricRendering] Fix compilation due to removal of params [#1869]( https://github.com/sofa-framework/sofa/pull/1869 )
- [examples] Add missing RequiredPlugin [#1895]( https://github.com/sofa-framework/sofa/pull/1895 )
- [examples] FIX scene name for Regression_test [#1881]( https://github.com/sofa-framework/sofa/pull/1881 )
- [examples] Fix or disable scene tests [#1919]( https://github.com/sofa-framework/sofa/pull/1919 )
- [examples] Fix warnings in Pendulum tutorial [#2055]( https://github.com/sofa-framework/sofa/pull/2055 )
- [examples] Scenes with build_lcp [#2110]( https://github.com/sofa-framework/sofa/pull/2110 )
- [image] Fix wrong condition [#2083]( https://github.com/sofa-framework/sofa/pull/2083 )
- [plugins] Disable old python2 examples [#2025]( https://github.com/sofa-framework/sofa/pull/2025 )
- [runSofa/Modules] Fix various problems with Sofa.GL [#1743]( https://github.com/sofa-framework/sofa/pull/1743 )

**Examples / Scenes**
- [scenes] Fix (new?) failing scenes on the CI [#1798]( https://github.com/sofa-framework/sofa/pull/1798 )


### Cleanings
**Architecture**
- [CMake] CLEAN old metapackage usage [#1883]( https://github.com/sofa-framework/sofa/pull/1883 )
- [CMake] CLEAN option SOFA_BUILD_WITH_PCH_ENABLED [#1755]( https://github.com/sofa-framework/sofa/pull/1755 )
- [CMake] CLEAN/FIX deprecated things (MSVC mainly) [#2217]( https://github.com/sofa-framework/sofa/pull/2217 )

**Modules**
- [All] Remove commented code & fix trivial issues [#1693]( https://github.com/sofa-framework/sofa/pull/1693 )
- [All] Remove useless pragma guards in cpp files [#1929]( https://github.com/sofa-framework/sofa/pull/1929 )
- [All] CLEAN tests, use CMake weak dependencies [#1886]( https://github.com/sofa-framework/sofa/pull/1886 )
- [All] Disable Drawtool functions using Vec4f [#2197]( https://github.com/sofa-framework/sofa/pull/2197 )
- [All] Rename SOFA_NO_OPENGL + Relocate Sofa.GL + Improve sofa_add_* macros [#1913]( https://github.com/sofa-framework/sofa/pull/1913 )
- [All] Avoid redundant printLog check [#2102]( https://github.com/sofa-framework/sofa/pull/2102 )
- [All] Depreciate one of the missing use of Aspect in Link and update the code base [#1712]( https://github.com/sofa-framework/sofa/pull/1712 )
- [All] Fix warnings [#2196]( https://github.com/sofa-framework/sofa/pull/2196 )
- [All] Fix warnings [#2097]( https://github.com/sofa-framework/sofa/pull/2097 )
- [All] Refactor Read/Write Accessor.  [#1807]( https://github.com/sofa-framework/sofa/pull/1807 )
- [All] Refactor vector & vector_device [#1799]( https://github.com/sofa-framework/sofa/pull/1799 )
- [All] Remove last includes of SofaSimulationTree [#1812]( https://github.com/sofa-framework/sofa/pull/1812 )
- [All] Remove un-needed includes. [#1750]( https://github.com/sofa-framework/sofa/pull/1750 )
- [All] Remove unused includes [#1960]( https://github.com/sofa-framework/sofa/pull/1960 )
- [All] Remove warning: unused variable [#1787]( https://github.com/sofa-framework/sofa/pull/1787 )
- [All] Update code to use sofa::InvalidID instead of topology::InvalidID [#2116]( https://github.com/sofa-framework/sofa/pull/2116 )
- [All] use =deleted for deprecation [#1793]( https://github.com/sofa-framework/sofa/pull/1793 )
- [All] Convert tests to Sofa.Testing [#2188]( https://github.com/sofa-framework/sofa/pull/2188 )
- [All] Remove useless inclusions of MechanicalObject.h [#2015]( https://github.com/sofa-framework/sofa/pull/2015 )
- [All] Remove deprecated calls and warnings [#2210]( https://github.com/sofa-framework/sofa/pull/2210 )
- [All] Remove use of compatibility layer [#2179]( https://github.com/sofa-framework/sofa/pull/2179 )
- **[Sofa.Core]** Move eq,peq utilities functions to a standalone file [#2137]( https://github.com/sofa-framework/sofa/pull/2137 )
- **[Sofa.GL]** Relocate the module (again) + clarify module vs plugin definitions [#1941]( https://github.com/sofa-framework/sofa/pull/1941 )
- **[Sofa.GL]** Remove warnings from deprecated headers [#2045]( https://github.com/sofa-framework/sofa/pull/2045 )
- **[Sofa.GL]** static variable belongs to the class [#1951]( https://github.com/sofa-framework/sofa/pull/1951 )
- **[Sofa.Topology]** remove unnecessary pragma [#1969]( https://github.com/sofa-framework/sofa/pull/1969 )
- **[Sofa.Type & DefaultType]** Adds forward declaration for Vec and StdRigidTypes. [#1907]( https://github.com/sofa-framework/sofa/pull/1907 )
- **[Sofa.Type]** Modernize fixed_array [#1985]( https://github.com/sofa-framework/sofa/pull/1985 )
- **[Sofa.Type]** Remove MIN_DETERMINANT preprocessor value [#1932]( https://github.com/sofa-framework/sofa/pull/1932 )
- **[Sofa.Type]** add operator* for RGBAColor [#1952]( https://github.com/sofa-framework/sofa/pull/1952 )
- **[SofaBaseCollision]** Broad phase and narrow phase separation [#2118]( https://github.com/sofa-framework/sofa/pull/2118 )
- **[SofaBaseCollision]** Fix simulation dependency [#1768]( https://github.com/sofa-framework/sofa/pull/1768 )
- **[SofaBaseMechanics_test]** Update tests to use Sofa.Testing instead of Sofa_Test [#2144]( https://github.com/sofa-framework/sofa/pull/2144 )
- **[SofaBaseTopology]** Improve TopologyDataHandler message [#2152]( https://github.com/sofa-framework/sofa/pull/2152 )
- **[SofaBaseTopology]** Remove definition of real [#1955]( https://github.com/sofa-framework/sofa/pull/1955 )
- **[SofaBaseTopology]** Remove warning when a Data is directly linked to a topoogy Data container [#1971]( https://github.com/sofa-framework/sofa/pull/1971 )
- **[SofaBaseTopology]** Remove warnings [#2046]( https://github.com/sofa-framework/sofa/pull/2046 )
- **[SofaBaseTopology]** Single definition of global variables [#1950]( https://github.com/sofa-framework/sofa/pull/1950 )
- **[SofaBaseUtils]** Clean RequiredPlugin [#1899]( https://github.com/sofa-framework/sofa/pull/1899 )
- **[SofaBaseVisual]** Move #include<Mat.h> from BaseCamera.h to BaseCamera.cpp [#1846]( https://github.com/sofa-framework/sofa/pull/1846 )
- **[SofaBase]** Convert tests to Sofa.Testing [#2146]( https://github.com/sofa-framework/sofa/pull/2146 )
- [SofaBoundaryCondition] Avoid ambiguity [#1958]( https://github.com/sofa-framework/sofa/pull/1958 )
- [SofaCommon] Convert tests to Sofa.Testing [#2153]( https://github.com/sofa-framework/sofa/pull/2153 )
- [SofaConstraint] Small cleaning [#2174]( https://github.com/sofa-framework/sofa/pull/2174 )
- **[SofaCore]** Deprecate operator= [#2167]( https://github.com/sofa-framework/sofa/pull/2167 )
- **[SofaCore]** Factoring code in Link.h [#1704]( https://github.com/sofa-framework/sofa/pull/1704 )
- **[SofaCore]** Merge TData in Data.  [#1753]( https://github.com/sofa-framework/sofa/pull/1753 )
- **[SofaCore]** Minor cleaning in constraints [#2138]( https://github.com/sofa-framework/sofa/pull/2138 )
- **[SofaCore]** Move Link::CheckPath() method to PathResolver::CheckPath() [#1717]( https://github.com/sofa-framework/sofa/pull/1717 )
- **[SofaCore]** Move definition Link::read() method to BaseLink::read() [#1736]( https://github.com/sofa-framework/sofa/pull/1736 )
- **[SofaCore]** Move function definition in cpp files [#2041]( https://github.com/sofa-framework/sofa/pull/2041 )
- **[SofaCore]** Move the streaming operator from .h to .cpp [#1751]( https://github.com/sofa-framework/sofa/pull/1751 )
- **[SofaCore]** Reduce the needs for include in Node.h [#1744]( https://github.com/sofa-framework/sofa/pull/1744 )
- **[SofaCore]** Remove empty file [#2204]( https://github.com/sofa-framework/sofa/pull/2204 )
- **[SofaCore_simutest]** Add tests for Link::CheckPath [#1714]( https://github.com/sofa-framework/sofa/pull/1714 )
- **[SofaCore_test]** Update the two failling tests so they match the new convention for TypeInfo [#1709]( https://github.com/sofa-framework/sofa/pull/1709 )
- **[SofaExplicitOdeSolver]** Mark Data as disabled [#2218]( https://github.com/sofa-framework/sofa/pull/2218 )
- [SofaExporter] Remove SofaBaseVisual dependency [#2039]( https://github.com/sofa-framework/sofa/pull/2039 )
- [SofaExporter] Removed duplicated header guard [#1954]( https://github.com/sofa-framework/sofa/pull/1954 )
- [SofaExporter] Update OBJExporter example [#1782]( https://github.com/sofa-framework/sofa/pull/1782 )
- **[SofaFramework/Sofa.Testing]** remove macro and move testing resources from SofaFramework  [#2000]( https://github.com/sofa-framework/sofa/pull/2000 )
- **[SofaFramework]** Remove warnings (a lot) [#1876]( https://github.com/sofa-framework/sofa/pull/1876 )
- [SofaGeneralEngine] Add option in NearestPointROI to use restPosition or position [#1978]( https://github.com/sofa-framework/sofa/pull/1978 )
- [SofaGeneralEngine] Remove dep on IdentityMapping (because of helper::eq) [#2090]( https://github.com/sofa-framework/sofa/pull/2090 )
- [SofaGeneralMeshCollision] Introduce RayTraceNarrowPhase [#2145]( https://github.com/sofa-framework/sofa/pull/2145 )
- [SofaGuiCommon] Clean includes [#2064]( https://github.com/sofa-framework/sofa/pull/2064 )
- [SofaGuiQt] FIX warning in GraphListenerQListView [#2091]( https://github.com/sofa-framework/sofa/pull/2091 )
- **[SofaHelper]** Clarify with global namespace [#1953]( https://github.com/sofa-framework/sofa/pull/1953 )
- **[SofaHelper]** Moves operator>> specialisation for int from set.h to set.cpp [#1902]( https://github.com/sofa-framework/sofa/pull/1902 )
- **[SofaHelper]** Remove unused UnitTest class + clean FnDispatcher includes [#1983]( https://github.com/sofa-framework/sofa/pull/1983 )
- [SofaImplicitOdeSolver] Fix Latex format for doxygen [#2205]( https://github.com/sofa-framework/sofa/pull/2205 )
- **[SofaKernel]** Remove last template parameter in Visitor::for_each/for_each_r [#1689]( https://github.com/sofa-framework/sofa/pull/1689 )
- **[SofaKernel]** Remove some dependencies from SofaHelper to SofaCore [#1686]( https://github.com/sofa-framework/sofa/pull/1686 )
- **[SofaKernel]** Removes the method BaseData::getOwnerClass()  [#1890]( https://github.com/sofa-framework/sofa/pull/1890 )
- **[SofaKernel][SofaGui]** Move Boost::program_option and remove Boost::system dependencies [#1848]( https://github.com/sofa-framework/sofa/pull/1848 )
- [SofaMiscCollision] Move back to modules (instead of applications/plugins) [#2127]( https://github.com/sofa-framework/sofa/pull/2127 )
- [SofaMiscMapping] Remove unused global variable [#1956]( https://github.com/sofa-framework/sofa/pull/1956 )
- [SofaNonUniformFem] Remove shadow variable [#1874]( https://github.com/sofa-framework/sofa/pull/1874 )
- **[SofaSimpleFem]** FIX wrong initialization order [#1938]( https://github.com/sofa-framework/sofa/pull/1938 )
- **[SofaSimulation\*]** Fix confusion between namespaces [#1959]( https://github.com/sofa-framework/sofa/pull/1959 )
- **[SofaSimulationCore]** Clean free motion animation loop [#1930]( https://github.com/sofa-framework/sofa/pull/1930 )
- **[SofaSimulationCore]** Move WorkerThread class into its own file [#2002]( https://github.com/sofa-framework/sofa/pull/2002 )
- **[SofaSimulationCore]** Moves code from TopologyChangeVisitor.h into .cpp [#1905]( https://github.com/sofa-framework/sofa/pull/1905 )
- **[SofaSimulationCore]** Remove ClassSystem.h [#1844]( https://github.com/sofa-framework/sofa/pull/1844 )
- **[SofaSimulationCore]** Add tons of details in Euler solver [#2163]( https://github.com/sofa-framework/sofa/pull/2163 )
- **[SofaSimulationCore]** Clean AnimateVisitor [#2194]( https://github.com/sofa-framework/sofa/pull/2194 )
- **[SofaSimulation]** Convert tests to Sofa.Testing [#2154]( https://github.com/sofa-framework/sofa/pull/2154 )
- [SofaSparseSolver] Clean useless dependencies [#2012]( https://github.com/sofa-framework/sofa/pull/2012 )
- [SofaUserInteraction] Minor cleaning of RayTraceDetection [#2009]( https://github.com/sofa-framework/sofa/pull/2009 )
- [SofaUserInteraction] Remove AddFramePerformer as it is not compiled [#1970]( https://github.com/sofa-framework/sofa/pull/1970 )
- [SofaUserInteraction] Deprecate RayTraceDetection [#2212]( https://github.com/sofa-framework/sofa/pull/2212 )
- [Tests] Move tests in their (new) correct locations [#1998]( https://github.com/sofa-framework/sofa/pull/1998 )
- [Tests] Replace sofa::helper::testing by sofa::testing [#2143]( https://github.com/sofa-framework/sofa/pull/2143 )

**Plugins / Projects**
- [CGAL] Remove #pragma once in .cpp files [#2202]( https://github.com/sofa-framework/sofa/pull/2202 )
- [Geomagic] Add macro HAS_OPENHAPTICS to know if library is present [#2115]( https://github.com/sofa-framework/sofa/pull/2115 )
- [Geomagic] Duplicated includes #2070 [#2072]( https://github.com/sofa-framework/sofa/pull/2072 )
- [HeadlessRecorder] CLEAN Headless recorder [#2058]( https://github.com/sofa-framework/sofa/pull/2058 )
- [LMConstraint] Move LMConstraint components in a new plugin [#1659]( https://github.com/sofa-framework/sofa/pull/1659 )
- [LMConstraint] Move examples in LMConstraint plugin [#1778]( https://github.com/sofa-framework/sofa/pull/1778 )
- [MultiThreading] Removes useless classid in AnimationLoopParalleleScheduler.cpp [#1906]( https://github.com/sofa-framework/sofa/pull/1906 )
- [SofaCUDA] Renaming cudaMatrix methods to use rowSize and colSize  [#1788]( https://github.com/sofa-framework/sofa/pull/1788 )
- [SofaDistanceGrid] Regroup miniFlowVR-related things [#1616]( https://github.com/sofa-framework/sofa/pull/1616 )
- [SofaPython] CLEAN PythonEnvironment info messages [#1835]( https://github.com/sofa-framework/sofa/pull/1835 )
- [THMPGSpatialHashing] Add a readme file and basic information [#2007]( https://github.com/sofa-framework/sofa/pull/2007 )
- [projects] remove unused variable [#1747]( https://github.com/sofa-framework/sofa/pull/1747 )

**Examples / Scenes**
- [examples] Clean warning scene examples [#1802]( https://github.com/sofa-framework/sofa/pull/1802 )
- [scenes] Update tetrahedron and tetrahedralCorotational FEM scenes [#2172]( https://github.com/sofa-framework/sofa/pull/2172 )

**Scripts / Tools**
- [tools] CLEAN logs for macos-postinstall-fixup [#2120]( https://github.com/sofa-framework/sofa/pull/2120 )


____________________________________________________________



## [v20.12.03]( https://github.com/sofa-framework/sofa/tree/v20.12.03 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v20.12.02...v20.12.03 )


### Bug Fixes
**Architecture**
- [CMake] Set CMake install default component [#2037]( https://github.com/sofa-framework/sofa/pull/2037 )

**Modules**
- [SofaBoundaryCondition] Fix export keywords [#1984]( https://github.com/sofa-framework/sofa/pull/1984 )
- [SofaGuiQt] Fix Expand node [#2069]( https://github.com/sofa-framework/sofa/pull/2069 )
- **[SofaHelper]** Fix export keywords [#1984]( https://github.com/sofa-framework/sofa/pull/1984 )
- [SofaMiscCollision] Fix config.in cmake file for export [#2052]( https://github.com/sofa-framework/sofa/pull/2052 )
- **[SofaSimulationCore]** Broken URL [#2011]( https://github.com/sofa-framework/sofa/pull/2011 )

**Plugins / Projects**
- [Geomagic] Fix Geomagic plugin compilation  [#2033]( https://github.com/sofa-framework/sofa/pull/2033 )


____________________________________________________________


## [v20.12.02]( https://github.com/sofa-framework/sofa/tree/v20.12.02 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v20.12.01...v20.12.02 )


### Bug Fixes
**Architecture**
- [CMake] FIX resources and translations install on Windows [#1949]( https://github.com/sofa-framework/sofa/pull/1949 )

**Modules**
- [SofaGeneralExplicitOdeSolver] Fix typo in CentralDifferenceSolver description [#1894]( https://github.com/sofa-framework/sofa/pull/1894 )
- [SofaGuiQt] Fix RealGUI: QDesktopWidget header missing for Qt < 5.11 [#1904]( https://github.com/sofa-framework/sofa/pull/1904 )
- [SofaGuiQt] FIX segfault due to qFatal in GenGraphForm [#1948]( https://github.com/sofa-framework/sofa/pull/1948 )
- **[SofaHelper]** Fix out of bounds error [#1868]( https://github.com/sofa-framework/sofa/pull/1868 )

**Plugins / Projects**
- [SofaCUDA] change the setTopology method signature [#1843]( https://github.com/sofa-framework/sofa/pull/1843 )

**Examples / Scenes**
- [examples] Add missing RequiredPlugin [#1895]( https://github.com/sofa-framework/sofa/pull/1895 )


____________________________________________________________



## [v20.12.01]( https://github.com/sofa-framework/sofa/tree/v20.12.01 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v20.12.00...v20.12.01 )


### Improvements
**Modules**
- [SofaGui] Improve background v20.12 [#1758]( https://github.com/sofa-framework/sofa/pull/1758 )
- **[SofaHelper]** ADD PluginManager callback and use it in SofaPython [#1777]( https://github.com/sofa-framework/sofa/pull/1777 )

**Plugins / Projects**
- [SofaPython] ADD PluginManager callback and use it in SofaPython [#1777]( https://github.com/sofa-framework/sofa/pull/1777 )

### Bug Fixes
**Architecture**
- [SofaMacros] Shorter name for relocatable targets [#1769]( https://github.com/sofa-framework/sofa/pull/1769 )

**Modules**
- [All] FIX build without SofaPython soft dependencies [#1829]( https://github.com/sofa-framework/sofa/pull/1829 )
- [All] Fix invalid Memory access in MechanicalObject and Compliant_test [#1849]( https://github.com/sofa-framework/sofa/pull/1849 )
- [All] Almost green dashboard [#1669]( https://github.com/sofa-framework/sofa/pull/1669 )
- **[SofaCore_test]** Update the two failling tests so they match the new convention for TypeInfo [#1709]( https://github.com/sofa-framework/sofa/pull/1709 )
- **[SofaDefaultType]** FIX compilation errors [#1761]( https://github.com/sofa-framework/sofa/pull/1761 )
- **[SofaBaseMechanics]** fixes reinit of BarycentricMapping [#1815]( https://github.com/sofa-framework/sofa/pull/1815 )
- [SofaGuiCommon] Fix computationTimeSampling [#1698]( https://github.com/sofa-framework/sofa/pull/1698 )
- [SofaGuiCommon] fix BackgroundSetting [#1826]( https://github.com/sofa-framework/sofa/pull/1826 )
- [SofaGuiQt] ADD qt.conf.h for custom qt.conf loading at runtime [#1820]( https://github.com/sofa-framework/sofa/pull/1820 )
- [SofaGuiQt] Fix Export Graph option [#1785]( https://github.com/sofa-framework/sofa/pull/1785 )
- [SofaGuiQt] Clean QtGLViewer with key events [#1850]( https://github.com/sofa-framework/sofa/pull/1850 )
- **[SofaHelper]** FIX "name vs path" issue in PluginManager + FIX PluginManager_test [#1823]( https://github.com/sofa-framework/sofa/pull/1823 )
- [SofaMiscFEM] Minor changes in TrianglePressureFF and TriangularFEMFF [#1779]( https://github.com/sofa-framework/sofa/pull/1779 )
- **[SofaSimpleFem]** Fix nasty bug in HexaFEMForceField' s draw() [#1766]( https://github.com/sofa-framework/sofa/pull/1766 )

**Plugins / Projects**
- [Geomagic] Fix GeomagicVisualModel compilation [#1776]( https://github.com/sofa-framework/sofa/pull/1776 )
- [PreassembledMass] FIX typo in PreassembledMass.inl [#1833]( https://github.com/sofa-framework/sofa/pull/1833 )
- [SofaCUDA] FIX compilation errors [#1761]( https://github.com/sofa-framework/sofa/pull/1761 )
- [SofaCUDA] Fix CudaFixedConstraint when indices are not contiguous [#1780]( https://github.com/sofa-framework/sofa/pull/1780 )
- [SofaOpenCL] Fix compilation [#1795]( https://github.com/sofa-framework/sofa/pull/1795 )
- [SofaPardisoSolver] Fix plugin and add example [#1830]( https://github.com/sofa-framework/sofa/pull/1830 )
- [SofaRigid] fixes applyJT of RigidMapping [#1813]( https://github.com/sofa-framework/sofa/pull/1813 )
- [SofaSphFluid] Remove std::execution usage [#1684]( https://github.com/sofa-framework/sofa/pull/1684 )

**Examples / Scenes**
- [Scenes] Fix (new?) failing scenes on the CI [#1798]( https://github.com/sofa-framework/sofa/pull/1798 )

### Cleanings
**Plugins / Projects**
- [SofaPython] CLEAN PythonEnvironment info messages [#1835]( https://github.com/sofa-framework/sofa/pull/1835 )
- [SofaCUDA] Renaming cudaMatrix methods to use rowSize and colSize  [#1788]( https://github.com/sofa-framework/sofa/pull/1788 )


____________________________________________________________



## [v20.12.00]( https://github.com/sofa-framework/sofa/tree/v20.12.00 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v20.06.01...v20.12.00 )


### SOFA-NG
Follow the SOFA-NG project on [its board]( https://github.com/sofa-framework/sofa/projects/9) and [its main issue](https://github.com/sofa-framework/sofa/issues/1527 ).
- [SofaMisc] Pluginize all modules [#1306]( https://github.com/sofa-framework/sofa/issues/1306 )
- [SofaGeneral] Pluginize all modules [#1529]( https://github.com/sofa-framework/sofa/issues/1529 )
- [SofaCommon] Pluginize all modules [#1597]( https://github.com/sofa-framework/sofa/issues/1597 )
- [SofaBase] Package all modules [#1633]( https://github.com/sofa-framework/sofa/issues/1633 )
- [modules] Set relocatable flags to pluginized modules [#1604]( https://github.com/sofa-framework/sofa/pull/1604 )
- [SofaCommon] Make SofaCommon a deprecated collection [#1609]( https://github.com/sofa-framework/sofa/pull/1609 )
- [SofaGeneral] Make SofaGeneral a deprecated collection [#1596]( https://github.com/sofa-framework/sofa/pull/1596 )
- [SofaGeneral] Move BVH-format feature from Helper to SofaGeneralRigid [#1644]( https://github.com/sofa-framework/sofa/pull/1644 )


### Breaking
**Architecture**
- [SofaMacros] Refactor for better target and package management [#1433]( https://github.com/sofa-framework/sofa/pull/1433 )

**Modules**
- [All] Change index_type from size_t to uint [#1514]( https://github.com/sofa-framework/sofa/pull/1514 )
- [All] Deprecate m_componentstate and rename to d_componentState [#1358]( https://github.com/sofa-framework/sofa/pull/1358 )
- [All] Factorize and rename TopologyObjectType into TopologyElementType [#1593]( https://github.com/sofa-framework/sofa/pull/1593 )
- [All] Remove topologyAlgorithms classes [#1546]( https://github.com/sofa-framework/sofa/pull/1546 )
- [All] Standardize index type for Vector/Matrix templates [#1453]( https://github.com/sofa-framework/sofa/pull/1453 )
- [All] Uniform size type  [#1515]( https://github.com/sofa-framework/sofa/pull/1515 )
- **[SofaKernel]** Refactor BaseData to use DataLink [#1491]( https://github.com/sofa-framework/sofa/pull/1491 )
- **[SofaKernel]** ♻️ FIX & CLEANUP BoxROI [#1482]( https://github.com/sofa-framework/sofa/pull/1482 )
- **[SofaKernel]**[SofaCore][SofaLoader][SofaGeneralLoader] SOFA with callbacks [#1408]( https://github.com/sofa-framework/sofa/pull/1408 )

**Plugins / Projects**
- [ColladaSceneLoader][SofaAssimp] Move ColladaSceneLoader plugin content into SofaAssimp plugin [#1360]( https://github.com/sofa-framework/sofa/pull/1360 )
- [plugins] Remove plugins to be deleted [#1439]( https://github.com/sofa-framework/sofa/pull/1439 )


### Improvements
**Architecture**
- [All] Accelerating CMake generation [#1464]( https://github.com/sofa-framework/sofa/pull/1464 )
- [SofaMacros] Handle COMPONENTS (needed by SofaPython3) [#1671]( https://github.com/sofa-framework/sofa/pull/1671 )

**Modules**
- [All] Replace last use of Qwt by QtCharts and remove internal library [#1512]( https://github.com/sofa-framework/sofa/pull/1512 )
- [SofaBaseCollision] Add option to use of normal orientation in triangle and self-colliding cube [#1559]( https://github.com/sofa-framework/sofa/pull/1559 )
- **[SofaCore]** Add virtual getPathName function in Base [#1455]( https://github.com/sofa-framework/sofa/pull/1455 )
- [SofaGeneralLoader] Add option for transform in SphereLoader to match other loaders API [#1495]( https://github.com/sofa-framework/sofa/pull/1495 )
- [SofaGeneralLoader] allow ReadState at init [#1654]( https://github.com/sofa-framework/sofa/pull/1654 )
- [SofaHaptics] Add multithread test on LCPForceFeedback component [#1581]( https://github.com/sofa-framework/sofa/pull/1581 )
- [SofaHaptics] Add simple tests on LCPForceFeedback component [#1576]( https://github.com/sofa-framework/sofa/pull/1576 )
- [SofaImplicitField] Add new ImplicitFields and getHessian to ScalarField [#1510]( https://github.com/sofa-framework/sofa/pull/1510 )
- **[SofaKernel]** ADD: add polynomial springs force fields [#1342]( https://github.com/sofa-framework/sofa/pull/1342 )
- **[SofaKernel]** Add DataLink object & PathResolver. [#1485]( https://github.com/sofa-framework/sofa/pull/1485 )
- **[SofaKernel]** Add setLinkedBase method in BaseLink [#1436]( https://github.com/sofa-framework/sofa/pull/1436 )
- **[SofaKernel]** Add whole program optimization (aka link-time optimization) for msvc [#1468]( https://github.com/sofa-framework/sofa/pull/1468 )
- **[SofaKernel]** Exposing Data in ContactListener. [#1678]( https://github.com/sofa-framework/sofa/pull/1678 )
- **[SofaKernel]** Filerepository gettemppath [#1383]( https://github.com/sofa-framework/sofa/pull/1383 )
- **[SofaKernel]** Set read-only all data defined by the file loaded [#1660]( https://github.com/sofa-framework/sofa/pull/1660 )
- [SofaQtGui] Restore GraphWidget for Momentum and Energy using QtCharts instead of Qwt [#1508]( https://github.com/sofa-framework/sofa/pull/1508 )

**Plugins / Projects**
- [Compliant] Add WinchMultiMapping and ContactMultiMapping [#1557]( https://github.com/sofa-framework/sofa/pull/1557 )


### Bug Fixes
**Architecture**
- [CMake] FIX non-existent target with sofa_add_plugin [#1584]( https://github.com/sofa-framework/sofa/pull/1584 )
- [CMake] Fix Cmake configure step with SOFA_WITH_DEPRECATED_COMPONENTS [#1452]( https://github.com/sofa-framework/sofa/pull/1452 )

**Extlibs**
- [extlibs/gtest] Fix the broken sofa_create_package_with_targets in gtest [#1457]( https://github.com/sofa-framework/sofa/pull/1457 )

**Modules**
- [All] issofa_bugfix: cleans and fixes [#218]( https://github.com/sofa-framework/sofa/pull/218 )
- [SofaBaseLinearSolver] Fix logging info with SPARSEMATRIX_VERBOSE  [#1715]( https://github.com/sofa-framework/sofa/pull/1715 )
- [SofaBaseMechanics] Use d_showColor for indices instead of arbitrary white [#1511]( https://github.com/sofa-framework/sofa/pull/1511 )
- [SofaBaseMechanics] 🐛 FIX draw function in UniformMass [#1480]( https://github.com/sofa-framework/sofa/pull/1480 )
- [SofaCarving] Fix method doCarve should be called at AnimateEndEvent [#1532]( https://github.com/sofa-framework/sofa/pull/1532 )
- **[SofaCore]** FIX const correctness in DataTracker [#1488]( https://github.com/sofa-framework/sofa/pull/1488 )
- **[SofaCore]** FIX simu unload crash caused by missing checks on slaves ptrs [#1445]( https://github.com/sofa-framework/sofa/pull/1445 )
- **[SofaFramework]** Fix deprecated_as_error macro for MSVC [#1658]( https://github.com/sofa-framework/sofa/pull/1658 )
- [SofaGUI] Fix Cmake files for out-of-tree compilation [#1570]( https://github.com/sofa-framework/sofa/pull/1570 )
- [SofaGeneralAnimationLoop] Fix mechanical matrix mapper [#1587]( https://github.com/sofa-framework/sofa/pull/1587 )
- [SofaGeneralEngine] Fix BarycentricMapperEngine parse() function [#1516]( https://github.com/sofa-framework/sofa/pull/1516 )
- [SofaGeneralLoader] fix GIDMeshLoader and add example [#1554]( https://github.com/sofa-framework/sofa/pull/1554 )
- [SofaHelper/image] Fix unit tests [#1585]( https://github.com/sofa-framework/sofa/pull/1585 )
- **[SofaHelper]** Add SOFA/bin to SOFA_PLUGIN_PATH [#1718]( https://github.com/sofa-framework/sofa/pull/1718 )
- **[SofaHelper]** Link necessary Boost macro with SofaHelper (for Windows) [#1578]( https://github.com/sofa-framework/sofa/pull/1578 )
- **[SofaKernel]**[SofaGuiQt] Qt viewer with intel drivers [#1690]( https://github.com/sofa-framework/sofa/pull/1690 )
- **[SofaKernel]** Add updateOnTransformChange update callback on MeshLoader. [#1459]( https://github.com/sofa-framework/sofa/pull/1459 )
- **[SofaKernel]** Data file repository now looks into the SOFA install directory [#1656]( https://github.com/sofa-framework/sofa/pull/1656 )
- **[SofaKernel]** Improve check for already registered plugins [#1472]( https://github.com/sofa-framework/sofa/pull/1472 )
- **[SofaKernel]** In DataFileName, the name FILE used in the PathType enum could be ambigous  [#1465]( https://github.com/sofa-framework/sofa/pull/1465 )
- **[SofaKernel]** 🐛 Break link when passing a nullptr to setLinkedBase [#1479]( https://github.com/sofa-framework/sofa/pull/1479 )
- **[SofaKernel]**[SofaGeneralRigid] Minor fixes in ArticulatedSystemMapping and JointSpringForceField [#1448]( https://github.com/sofa-framework/sofa/pull/1448 )
- **[SofaKernel]** Implement an update mechanism on component: RequiredPlugin [#1458]( https://github.com/sofa-framework/sofa/pull/1458 )
- **[SofaKernel]** Switch to include_guard() instead of include_guard(GLOBAL) [#1467]( https://github.com/sofa-framework/sofa/pull/1467 )
- [SofaMacros] FIX RELOCATABLE_INSTALL_DIR target property [#1631]( https://github.com/sofa-framework/sofa/pull/1631 )
- [SofaMacros] FIX deprecated macro sofa_generate_package [#1446]( https://github.com/sofa-framework/sofa/pull/1446 )
- [SofaMacros] FIX libs copy and plugin deps finding [#1708]( https://github.com/sofa-framework/sofa/pull/1708 )
- [SofaMacros] FIX missing lib copy on Windows [#1711]( https://github.com/sofa-framework/sofa/pull/1711 )
- [SofaMacros] FIX plugins RPATH [#1619]( https://github.com/sofa-framework/sofa/pull/1619 )
- [SofaMacros] Improve RPATH coverage on MacOS [#1713]( https://github.com/sofa-framework/sofa/pull/1713 )
- [SofaMacros] Recursive deps search for RPATH [#1710]( https://github.com/sofa-framework/sofa/pull/1710 )
- [SofaOpenglVisual] OglViewport: a fix for compatibility with modern OpenGL [#1500]( https://github.com/sofa-framework/sofa/pull/1500 )
- [SofaSimulationGraph] No reason to have moveChild in private [#1470]( https://github.com/sofa-framework/sofa/pull/1470 )

**Plugins / Projects**
- [CGALPlugin] Fix compilation for CGal version > 5 [#1522]( https://github.com/sofa-framework/sofa/pull/1522 )
- [CImgPlugin] CLEAN dependencies in CMakeLists [#1651]( https://github.com/sofa-framework/sofa/pull/1651 )
- [Flexible] FIX deps to pluginized modules [#1590]( https://github.com/sofa-framework/sofa/pull/1590 )
- [Geomagic] Fix scenes ForceFeedBack behavior due to a change in UncoupledConstraintCorrection [#1435]( https://github.com/sofa-framework/sofa/pull/1435 )
- [OmniDriverEmul] Fix thread behavior and replace boost/pthread by std::thread [#1665]( https://github.com/sofa-framework/sofa/pull/1665 )
- [SofaOpenCL] Fix Cmake configuration [#1647]( https://github.com/sofa-framework/sofa/pull/1647 )
- [SofaPython] Small fixes to import plugin and remove scene warnings [#1466]( https://github.com/sofa-framework/sofa/pull/1466 )
- [runSofa] CLEAN unused dep to SofaGeneralLoader [#1648]( https://github.com/sofa-framework/sofa/pull/1648 )
- [SofaSPHFluid] Fix compilation with std::execution [#1542]( https://github.com/sofa-framework/sofa/pull/1542 )

**Examples / Scenes**
- [examples] Fix HexahedronForceFieldTopologyChangeHandling scene [#1573]( https://github.com/sofa-framework/sofa/pull/1573 )

**Scripts / Tools**
- [scripts] Update licenseUpdater [#1498]( https://github.com/sofa-framework/sofa/pull/1498 )


### Cleanings
**Architecture**
- [SofaMacros] Split SofaMacros.cmake into multiple files [#1477]( https://github.com/sofa-framework/sofa/pull/1477 )
- [SofaMacros] Use IN_LIST (CMake >= 3.3) [#1484]( https://github.com/sofa-framework/sofa/pull/1484 )

**Modules**
- [All] Bunch of removal of sout/serr in the whole code base [#1513]( https://github.com/sofa-framework/sofa/pull/1513 )
- [All] Fix compilation with flag NO_OPENGL [#1478]( https://github.com/sofa-framework/sofa/pull/1478 )
- [All] Fix many warnings [#1682]( https://github.com/sofa-framework/sofa/pull/1682 )
- [All] Remove SMP-related Code [#1613]( https://github.com/sofa-framework/sofa/pull/1613 )
- [All] Replace all sofa::defaulttypes::RGBAColor to sofa::helper::types::RGBAColor [#1463]( https://github.com/sofa-framework/sofa/pull/1463 )
- [Doc] Remove Inria Foundation mention from CONTRIBUTING [#1451]( https://github.com/sofa-framework/sofa/pull/1451 )
- [SofaBaseTopology] Fix ambiguity causing compilation error with MSVC [#1577]( https://github.com/sofa-framework/sofa/pull/1577 )
- [SofaBaseTopology] Rework method getIntersectionPointWithPlane [#1545]( https://github.com/sofa-framework/sofa/pull/1545 )
- [SofaBaseVisual][SofaDeformable] Clean some codes [#1449]( https://github.com/sofa-framework/sofa/pull/1449 )
- [SofaDeformable] Update RestShapeSpringsForceField [#1431]( https://github.com/sofa-framework/sofa/pull/1431 )
- [SofaGeneralEngine] Improve mesh barycentric mapper engine [#1487]( https://github.com/sofa-framework/sofa/pull/1487 )
- [SofaGeneralEngine] Remove useless create() function in some components [#1622]( https://github.com/sofa-framework/sofa/pull/1622 )
- [SofaGuiQt] Move libQGLViewer lib into its module [#1617]( https://github.com/sofa-framework/sofa/pull/1617 )
- [SofaHaptics] Small fix on LCPForceFeedback haptic rendering [#1537]( https://github.com/sofa-framework/sofa/pull/1537 )
- **[SofaHelper]** DrawTool uses RGBAColor now (instead of Vec4f) [#1626]( https://github.com/sofa-framework/sofa/pull/1626 )
- **[SofaHelper]** Remove OpenGL dependency in vector_device [#1646]( https://github.com/sofa-framework/sofa/pull/1646 )
- **[SofaKernel]** Clean namespace BarycentricMapper [#1571]( https://github.com/sofa-framework/sofa/pull/1571 )
- **[SofaKernel]** Factorize some code for maintenance [#1569]( https://github.com/sofa-framework/sofa/pull/1569 )
- **[SofaKernel]** Refactor the FileRepository constructors [#1610]( https://github.com/sofa-framework/sofa/pull/1610 )
- **[SofaKernel]** Remove core::Plugin/core::PluginManager [#1612]( https://github.com/sofa-framework/sofa/pull/1612 )
- **[SofaKernel]** Remove parentBaseData in  BaseData.h [#1490]( https://github.com/sofa-framework/sofa/pull/1490 )
- **[SofaKernel]** Remove support for BaseData in Link.h [#1503]( https://github.com/sofa-framework/sofa/pull/1503 )
- **[SofaKernel]** Remove un-needed StringUtil.h in Base.h [#1519]( https://github.com/sofa-framework/sofa/pull/1519 )
- **[SofaKernel]** Remove un-needed class reflection system. [#1541]( https://github.com/sofa-framework/sofa/pull/1541 )
- [SofaLoader] Use references in Meshloader [#1627]( https://github.com/sofa-framework/sofa/pull/1627 )
- [modules] Minor fixes [#1441]( https://github.com/sofa-framework/sofa/pull/1441 )

**Plugins / Projects**
- [plugins] Reactivate SofaPython3 [#1574]( https://github.com/sofa-framework/sofa/pull/1574 )
- [Geomagic] Update demo scenes to use direct solvers. [#1533]( https://github.com/sofa-framework/sofa/pull/1533 )
- [InvertibleFVM] Externalize the plugin [#1443]( https://github.com/sofa-framework/sofa/pull/1443 )


____________________________________________________________



## [v20.06]( https://github.com/sofa-framework/sofa/tree/v20.06 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v19.12...v20.06 )


### Breaking

**Architecture**
- [All] CMake and includes massive clean [#1397]( https://github.com/sofa-framework/sofa/pull/1397 )
- [CMake] Remove the use of an internal Eigen3 version and instead use the one installed on the system. [#1281]( https://github.com/sofa-framework/sofa/pull/1281 )
- [CMake] Remove Collections [#1314]( https://github.com/sofa-framework/sofa/pull/1314 )
- [Modularization] SofaNonUniformFem + SofaAdvanced removal [#1344]( https://github.com/sofa-framework/sofa/pull/1344 )
- [Modularization] SofaValidation [#1302]( https://github.com/sofa-framework/sofa/pull/1302 )

**Modules**
- [All] BaseClass reflection refactoring [#1283]( https://github.com/sofa-framework/sofa/pull/1283 )
- [All] Remove Aspects from Sofa [#1269]( https://github.com/sofa-framework/sofa/pull/1269 )
- [All] Remove compilation warnings related to collision models [#1301]( https://github.com/sofa-framework/sofa/pull/1301 )
- [All] Update code base according to refactoring done in PR1283. [#1330]( https://github.com/sofa-framework/sofa/pull/1330 )
- [All] Remove all deprecation warnings after v1912 [#1241]( https://github.com/sofa-framework/sofa/pull/1241 )
- [All] v19.12 changes + DocBrowser by Defrost [#1275]( https://github.com/sofa-framework/sofa/pull/1275 )
- **[SofaBaseMechanics]** Change data name: handleTopologicalChanges in UniformM [#1291]( https://github.com/sofa-framework/sofa/pull/1291 )
- **[SofaCore/Visual]** Add new functions in drawTool (BREAKING) [#1252]( https://github.com/sofa-framework/sofa/pull/1252 )
- [SofaGeneralEngine] Refresh MeshBarycentricMapperEngine [#1404]( https://github.com/sofa-framework/sofa/pull/1404 )
- **[SofaCore]** ADD directory DataFileNames [#1407]( https://github.com/sofa-framework/sofa/pull/1407 )
- **[SofaKernel]** Refactor DDGNode [#1372]( https://github.com/sofa-framework/sofa/pull/1372 )
- **[SofaKernel]** Totally remove the macro CHECK_TOPOLOGY from BaseMeshTopology [#1399]( https://github.com/sofa-framework/sofa/pull/1399 )
- **[SofaKernel]** Update EulerExplicitSolver [#1260]( https://github.com/sofa-framework/sofa/pull/1260 )
- **[SofaKernel]** implement activer's code at CollisionModel. [#1259]( https://github.com/sofa-framework/sofa/pull/1259 )

**Plugins**
- [SofaCUDA] Arch auto-detection for nvcc [#1336]( https://github.com/sofa-framework/sofa/pull/1336 )


### Improvements

**Architecture**
- [CMake] Create an IDE folder for all relocatable_install targets [#1405]( https://github.com/sofa-framework/sofa/pull/1405 )

**Modules**
- **[SofaBaseVisual]** Add camera gizmo in base camera [#1253]( https://github.com/sofa-framework/sofa/pull/1253 )
- **[SofaCore]** Remove warning in ExecParam [#1325]( https://github.com/sofa-framework/sofa/pull/1325 )
- **[SofaCore]** ADD: DataCallback system in Base [#1406]( https://github.com/sofa-framework/sofa/pull/1406 )
- **[SofaDefaultType]** Add a Ray type. [#1251]( https://github.com/sofa-framework/sofa/pull/1251 )
- **[SofaHelper]** Add the obj id to labels when available [#1256]( https://github.com/sofa-framework/sofa/pull/1256 )
- **[SofaHelper]** Add auto-friendly getWriteAccessors/getReadAcessor... [#1254]( https://github.com/sofa-framework/sofa/pull/1254 )
- **[SofaKernel]** Set the default compilation mode to c++17. [#1249]( https://github.com/sofa-framework/sofa/pull/1249 )
- **[SofaKernel]** Add DataTypeInfo for BoundingBox [#1373]( https://github.com/sofa-framework/sofa/pull/1373 )
- **[SofaKernel]** Cleaner output when the creation of an object fails [#1266]( https://github.com/sofa-framework/sofa/pull/1266 )
- **[SofaSimulationCore]** Add a way to fill a multi vector from a base vector with a MechanicalOperations (mops). [#1248]( https://github.com/sofa-framework/sofa/pull/1248 )

**Plugins / Projects**
- [plugins] Fix warnings for option compilation [#1316]( https://github.com/sofa-framework/sofa/pull/1316 )
- [sofa-launcher] Change doc on sofa-launcher to markdown [#1311]( https://github.com/sofa-framework/sofa/pull/1311 )
- [SofaCUDA] Compilation without OpenGL [#1242]( https://github.com/sofa-framework/sofa/pull/1242 )

**Examples / Scenes**
- [examples] Add a scene illustrating ShewchukPCG [#1420]( https://github.com/sofa-framework/sofa/pull/1420 )


### Bug Fixes

**Architecture**
- [CMake] Add SofaNonUniformFem to the dependencies of SofaMisc [#1384]( https://github.com/sofa-framework/sofa/pull/1384 )
- [SofaFramework/CMake] Create configuration type dir before copying lib [#1347]( https://github.com/sofa-framework/sofa/pull/1347 )
- [extlibs/gtest] Add character test in gtest paramName to allow dash character [#1265]( https://github.com/sofa-framework/sofa/pull/1265 )

**Modules**
- [All] Clean warnings for option config, again [#1339]( https://github.com/sofa-framework/sofa/pull/1339 )
- [All] Fix SOFA_LIBSUFFIX used in Debug by PluginManager [#1313]( https://github.com/sofa-framework/sofa/pull/1313 )
- [All] Overridden 'canCreate' methods should always log an error message when they fail [#1294]( https://github.com/sofa-framework/sofa/pull/1294 )
- **[SofaBaseTopology]** Fix GridTopology edge computation [#1323]( https://github.com/sofa-framework/sofa/pull/1323 )
- **[SofaBaseTopology]**[SofaExporter] Fix failing tests due to changes in topology [#1335]( https://github.com/sofa-framework/sofa/pull/1335 )
- [SofaConstraint] Fix test further to #1369 [#1386]( https://github.com/sofa-framework/sofa/pull/1386 )
- **[SofaEigen2Solver]** Fix CMake test on the version of Eigen [#1416]( https://github.com/sofa-framework/sofa/pull/1416 )
- **[SofaEngine]** Fix engine unit tests [#1280]( https://github.com/sofa-framework/sofa/pull/1280 )
- **[SofaEngine]** Fix Engine_test [#1338]( https://github.com/sofa-framework/sofa/pull/1338 )
- **[SofaFramework]** Windows/VS: Remove warnings flags from definitions [#1368]( https://github.com/sofa-framework/sofa/pull/1368 )
- [SofaGuiQt] Fix compilation for SOFA_DUMP_VISITOR_INFO [#1415]( https://github.com/sofa-framework/sofa/pull/1415 )
- [SofaGuiQt] Fix node graph [#1424]( https://github.com/sofa-framework/sofa/pull/1424 )
- [SofaHeadlessRecorder] Update headless recorder to use the new ffmpeg recorder [#1396]( https://github.com/sofa-framework/sofa/pull/1396 )
- **[SofaHelper]** AdvancedTimer wasn't using the good timer ids for the label assignments [#1244]( https://github.com/sofa-framework/sofa/pull/1244 )
- **[SofaHelper]** Fix unloading with PluginManager [#1274]( https://github.com/sofa-framework/sofa/pull/1274 )
- **[SofaHelper]** Fix fixed_array compilation with VS2019 [#1426]( https://github.com/sofa-framework/sofa/pull/1426 )
- **[SofaKernel]** Fix hexahedra detection in BoxROI [#1417]( https://github.com/sofa-framework/sofa/pull/1417 )
- **[SofaKernel]** Fix minor bug in BoxROI and add warning message in RestShapeSpringsForceField [#1391]( https://github.com/sofa-framework/sofa/pull/1391 )
- **[SofaKernel]** Fixes a bug where the camera was not moving with the Qt viewer [#1377]( https://github.com/sofa-framework/sofa/pull/1377 )
- **[SofaKernel]** Improve error message when a component cannot be created. [#1332]( https://github.com/sofa-framework/sofa/pull/1332 )
- **[SofaKernel]** Remove the installation of external system libraries [#1387]( https://github.com/sofa-framework/sofa/pull/1387 )
- **[SofaKernel]** Set default visibility to SOFA_EXPORT_DYNAMIC_LIBRARY [#1410]( https://github.com/sofa-framework/sofa/pull/1410 )
- [SofaMiscTopology] Fix bug in TopologicalChangeProcessor [#1247]( https://github.com/sofa-framework/sofa/pull/1247 )
- **[SofaSimpleFem]** Small Fix [#1403]( https://github.com/sofa-framework/sofa/pull/1403 )
- **[SofaSimulationCore]** FIX resizing of bboxes in UpdateBoundingBoxVisitor [#1268]( https://github.com/sofa-framework/sofa/pull/1268 )
- [SofaTopologyMapping] Fix Tetra2TriangleTopologicalMapping [#1319]( https://github.com/sofa-framework/sofa/pull/1319 )

**Plugins / Projects**
- [Geomagic] Fix several wrong behaviors in driver code [#1378]( https://github.com/sofa-framework/sofa/pull/1378 )
- [MeshSTEPLoader] FIX OCC version check [#1312]( https://github.com/sofa-framework/sofa/pull/1312 )
- [MeshSTEPLoader] FIX build with OCC >= 7.4 [#1310]( https://github.com/sofa-framework/sofa/pull/1310 )
- [Modeler] FIX link error on Windows [#1282]( https://github.com/sofa-framework/sofa/pull/1282 )
- [SofaMiscCollision] Fix topological changes in TetrahedronCollisionModel  [#1354]( https://github.com/sofa-framework/sofa/pull/1354 )
- [SofaSphFluid] Fix broken behavior for ParticleSink and ParticleSource [#1278]( https://github.com/sofa-framework/sofa/pull/1278 )
- [SofaSphFluid] FIX .scene-tests [#1317]( https://github.com/sofa-framework/sofa/pull/1317 )
- [SofaOpenCL] Make it work with 20.06 [#1361]( https://github.com/sofa-framework/sofa/pull/1361 )
- [SofaPython] Restrict the plugin and its dependers to C++11 [#1284]( https://github.com/sofa-framework/sofa/pull/1284 )

**Examples / Scenes**
- [examples] Fix SurfacePressureForceField example [#1273]( https://github.com/sofa-framework/sofa/pull/1273 )
- [examples] Fix caduceus [#1392]( https://github.com/sofa-framework/sofa/pull/1392 )
- [examples] Update the scene StandardTetrahedralFEMForceField.scn [#1064]( https://github.com/sofa-framework/sofa/pull/1064 )


### Cleanings

**Architecture**

**Modules**
- [All] Clean namespace for some classes [#1422]( https://github.com/sofa-framework/sofa/pull/1422 )
- [All] Fix warnings due to visibility attribute [#1421]( https://github.com/sofa-framework/sofa/pull/1421 )
- [All] Clean due to doc [#1398]( https://github.com/sofa-framework/sofa/pull/1398 )
- [All] Clean warnings [#1376]( https://github.com/sofa-framework/sofa/pull/1376 )
- [All] Fix minor warnings [#1388]( https://github.com/sofa-framework/sofa/pull/1388 )
- [All] Fix warnings generated due to change in Aspects [#1329]( https://github.com/sofa-framework/sofa/pull/1329 )
- [All] Minor changes in comment or format [#1411]( https://github.com/sofa-framework/sofa/pull/1411 )
- [All] Multiple fixes scenes [#1289]( https://github.com/sofa-framework/sofa/pull/1289 )
- [All] Remove all DISPLAY_TIME define [#1267]( https://github.com/sofa-framework/sofa/pull/1267 )
- [All] Remove some compilation warning. [#1343]( https://github.com/sofa-framework/sofa/pull/1343 )
- [All] Replace usage of sleep functions for the std one [#1394]( https://github.com/sofa-framework/sofa/pull/1394 )
- [All] Uniform use of M_PI [#1264]( https://github.com/sofa-framework/sofa/pull/1264 )
- [All] Update header for cleaner future management [#1375]( https://github.com/sofa-framework/sofa/pull/1375 )
- [All] replace all serr by msg_error/msg_warning [#1236]( https://github.com/sofa-framework/sofa/pull/1236 )
- [SofaConstraint] Set the use of integration factor true by default [#1369]( https://github.com/sofa-framework/sofa/pull/1369 )
- **[SofaDefaultType]** BoundingBox : Remove annoying warnings [#1425]( https://github.com/sofa-framework/sofa/pull/1425 )
- [SofaGeneralEngine] Fix draw of the sphere in SphereROI [#1318]( https://github.com/sofa-framework/sofa/pull/1318 )
- [SofaGeneralEngine] Remove remaining BoxROI after SofaEngine move [#1333]( https://github.com/sofa-framework/sofa/pull/1333 )
- [SofaGeneralLoader] Allow flip normals in Gmsh and STL loaders [#1418]( https://github.com/sofa-framework/sofa/pull/1418 )
- [SofaGui] Pass QDocBrowser as an option [#1315]( https://github.com/sofa-framework/sofa/pull/1315 )
- **[SofaKernel]** Add Data bool d_checkTopology in Topology container to replace the use of CHECK_TOPOLOGY macro [#1351]( https://github.com/sofa-framework/sofa/pull/1351 )
- **[SofaKernel]** Clean occurrences of CHECK_TOPOLOGY macro in code [#1352]( https://github.com/sofa-framework/sofa/pull/1352 )
- **[SofaKernel]** Clean of Material.h/cpp [#1346]( https://github.com/sofa-framework/sofa/pull/1346 )
- **[SofaKernel]** Remove X11 dependency when SOFA_NO_OPENGL is enabled. [#1370]( https://github.com/sofa-framework/sofa/pull/1370 )
- **[SofaKernel]** Who hates warnings? [#1258]( https://github.com/sofa-framework/sofa/pull/1258 )
- **[SofaKernel]** replace all serr by msg_error/msg_warning [#1237]( https://github.com/sofa-framework/sofa/pull/1237 )
- [SofaSparseSolver] Move CSparse and metis into SofaSparseSolver [#1357]( https://github.com/sofa-framework/sofa/pull/1357 )

**Plugins / Projects**
- [CGALPlugin] Clean and pluginization [#1308]( https://github.com/sofa-framework/sofa/pull/1308 )
- [Geomagic] Move all code related to device model display in a dedicated class. [#1366]( https://github.com/sofa-framework/sofa/pull/1366 )
- [Geomagic] Fix compilation [#1393]( https://github.com/sofa-framework/sofa/pull/1393 )
- [ManifoldTopologies] Remove CHECK_TOPOLOGY macro occurrences [#1353]( https://github.com/sofa-framework/sofa/pull/1353 )
- [ManifoldTopologies] Update the license in init.cpp [#1414]( https://github.com/sofa-framework/sofa/pull/1414 )
- [OpenCTMPlugin] Fix compilation and clean before moving out [#1359]( https://github.com/sofa-framework/sofa/pull/1359 )
- [PluginExample] Update PluginExample [#1356]( https://github.com/sofa-framework/sofa/pull/1356 )
- [Regression] Update hash [#1321]( https://github.com/sofa-framework/sofa/pull/1321 )
- [SofaSphFluid] Clean SofaFluid plugin compilation warning. [#1296]( https://github.com/sofa-framework/sofa/pull/1296 )

**Examples / Scenes**
- [examples] Fix and remove 3 scenes with deprecated component [#1355]( https://github.com/sofa-framework/sofa/pull/1355 )
- [examples] Remove useless files and add MeshMatrixMass example [#1257]( https://github.com/sofa-framework/sofa/pull/1257 )
- [scenes] Fix scenes from alias [#1292]( https://github.com/sofa-framework/sofa/pull/1292 )
- [scenes] Remove scene warnings due to Rigid template [#1295]( https://github.com/sofa-framework/sofa/pull/1295 )
- [scenes] Fix alias warnings in scenes [#1279]( https://github.com/sofa-framework/sofa/pull/1279 )


____________________________________________________________



## [v19.12]( https://github.com/sofa-framework/sofa/tree/v19.12 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v19.06...v19.12 )


### Breaking

**Architecture**
- [All] Improve extlibs integration [#1137]( https://github.com/sofa-framework/sofa/pull/1137 )
- [packages] Move all SofaComponent* + rename SofaAllCommonComponents [#1155]( https://github.com/sofa-framework/sofa/pull/1155 )

**Modules**
- [All] Add SingleLink to Topology to reveal all hidden GetMeshTopology [#1183]( https://github.com/sofa-framework/sofa/pull/1183 )
- [All] Remove ExtVecType [#1055]( https://github.com/sofa-framework/sofa/pull/1055 )
- [All] up change on GetMeshTopology [#1223]( https://github.com/sofa-framework/sofa/pull/1223 )
- [SofaBoundaryConditions] Apply doInternalUpdate API to ConstantForceField [#1145]( https://github.com/sofa-framework/sofa/pull/1145 )
- **[SofaKernel]** Replacing const char* with strings for group / help / widget etc. [#1152]( https://github.com/sofa-framework/sofa/pull/1152 )
- **[SofaKernel]** ADD: static method in events to retrieve the classname [#1118]( https://github.com/sofa-framework/sofa/pull/1118 )
- **[SofaKernel]** Set BaseData to non-persistant by default [#1191]( https://github.com/sofa-framework/sofa/pull/1191 )
- **[SofaKernel]** fix root's getPathName [#1146]( https://github.com/sofa-framework/sofa/pull/1146 )


### Improvements

**Architecture**
- [CMake] v19.06 changes [#1114]( https://github.com/sofa-framework/sofa/pull/1114 )
- [extlibs] Set Eigen as external project + upgrade to 3.2.10 [#1101]( https://github.com/sofa-framework/sofa/pull/1101 )
- [extlibs] Upgrade Qwt extlib from 6.1.2 to 6.1.4 [#1136]( https://github.com/sofa-framework/sofa/pull/1136 )

**Modules**
- [All] Add SingleLink to Topology to reveal hidden GetMeshTopology Part 2 [#1199]( https://github.com/sofa-framework/sofa/pull/1199 )
- [All] Add update internal mechanism [#1131]( https://github.com/sofa-framework/sofa/pull/1131 )
- [All] Update the SOFA Guidelines [#1135]( https://github.com/sofa-framework/sofa/pull/1135 )
- **[SofaBaseMechanics]** Add topological change in barycentric mapping [#1203]( https://github.com/sofa-framework/sofa/pull/1203 )
- **[SofaBaseMechanics]** Use doUpdateInternal API in DiagonalMass [#1150]( https://github.com/sofa-framework/sofa/pull/1150 )
- **[SofaBaseMechanics]** Use doUpdateInternal API in UniformMass [#1149]( https://github.com/sofa-framework/sofa/pull/1149 )
- **[SofaBaseTopology]** Add new geometric methods in TetrahedronSetGeometryAlgorythms [#1160]( https://github.com/sofa-framework/sofa/pull/1160 )
- **[SofaCore]** Remove thread specific declaration [#1129]( https://github.com/sofa-framework/sofa/pull/1129 )
- [SofaGeneralEngine] Added Rigid to Euler orientation export [#1141]( https://github.com/sofa-framework/sofa/pull/1141 )
- [SofaHaptics] Add mutex and option to lock the ForceFeedback computation [#1157]( https://github.com/sofa-framework/sofa/pull/1157 )
- **[SofaKernel]** ADD: DataTypeInfo<vector<string>> & improved  doc [#1113]( https://github.com/sofa-framework/sofa/pull/1113 )
- **[SofaKernel]** Add a strict option to the BoxROI to prevent partially inside element to be in the box. [#1127]( https://github.com/sofa-framework/sofa/pull/1127 )
- **[SofaKernel]** Add fixed_array_algorithm + RGBAColor::lighten [#1172]( https://github.com/sofa-framework/sofa/pull/1172 )
- **[SofaKernel]** Add new events to detect Initialization & Simulation start. [#1173]( https://github.com/sofa-framework/sofa/pull/1173 )
- **[SofaKernel]** Add option in StiffSpringFF to track list of input springs [#1093]( https://github.com/sofa-framework/sofa/pull/1093 )
- **[SofaKernel]** Change several AdvancedTimer logs for a better tracking [#1094]( https://github.com/sofa-framework/sofa/pull/1094 )
- **[SofaKernel]** Consistent SofaFramework modules [#1200]( https://github.com/sofa-framework/sofa/pull/1200 )
- **[SofaKernel]** Make componentState a real data field [#1168]( https://github.com/sofa-framework/sofa/pull/1168 )
- [SofaMiscForceField] Use doUpdateInternal API in MeshMatrixMass [#1151]( https://github.com/sofa-framework/sofa/pull/1151 )
- [SofaQtQuick] Pass extra command-line arguments for python scenes in a more high-level function call [#992]( https://github.com/sofa-framework/sofa/pull/992 )
- [SofaSphFluid] Add sprite-based point render [#1194]( https://github.com/sofa-framework/sofa/pull/1194 )
- [SofaSphFluid] Update rendering & other [#1215]( https://github.com/sofa-framework/sofa/pull/1215 )

**Plugins / Projects**
- [runSofa] Fix DataWidget display Speicherleck and long loading [#1181]( https://github.com/sofa-framework/sofa/pull/1181 )

**Examples / Scenes**
- [Examples] Add some mesh and PR1000 demo scene [#1112]( https://github.com/sofa-framework/sofa/pull/1112 )


### Bug Fixes

**Architecture**
- [CMake]**[SofaFramework]** Remove FFMPEG_exec target from the dependencies of SofaFramework [#1177]( https://github.com/sofa-framework/sofa/pull/1177 )
- [CMake] FIX Eigen finding [#1175]( https://github.com/sofa-framework/sofa/pull/1175 )
- [CMake] FIX unknown compiler option on VS2015 [#1192]( https://github.com/sofa-framework/sofa/pull/1192 )
- [SofaMacros] FIX default module version [#1123]( https://github.com/sofa-framework/sofa/pull/1123 )
- [SofaMacros] FIX sofa_set_install_relocatable escaped chars [#1154]( https://github.com/sofa-framework/sofa/pull/1154 )

**Modules**
- [All] Fix warnings [#1206]( https://github.com/sofa-framework/sofa/pull/1206 )
- [All] Fix warnings [#1167]( https://github.com/sofa-framework/sofa/pull/1167 )
- [All] Fix some warnings and OglAttribute handleTopologyChange [#1159]( https://github.com/sofa-framework/sofa/pull/1159 )
- [SofaBoundaryCondition] Fix FixedRotationConstraint when using more than one locked axis [#1119]( https://github.com/sofa-framework/sofa/pull/1119 )
- **[SofaBaseMechanics]** Make Uniform and DiagonalMass compatible with topo change [#1212]( https://github.com/sofa-framework/sofa/pull/1212 )
- **[SofaBaseTopology]** Fix SparseGrid obj loading + tests [#1231]( https://github.com/sofa-framework/sofa/pull/1231 )
- [SofaComponentAll] FIX SofaAllCommonComponents backward compatibility [#1204]( https://github.com/sofa-framework/sofa/pull/1204 )
- [SofaConstraint] Fix UncoupledConstraintCorrection topology change handling [#1115]( https://github.com/sofa-framework/sofa/pull/1115 )
- [SofaConstraint] Fix crash with PrecomputedConstraintCorrection [#1230]( https://github.com/sofa-framework/sofa/pull/1230 )
- **[SofaCore]** FIX decode functions in BaseClass [#1222]( https://github.com/sofa-framework/sofa/pull/1222 )
- **[SofaDefaulttype]** FIX too many ExtVec warnings with GCC [#1140]( https://github.com/sofa-framework/sofa/pull/1140 )
- [SofaExporter] Move bindings from SofaPython [#1095]( https://github.com/sofa-framework/sofa/pull/1095 )
- **[SofaFramework]** Add other orders for fromEuler() for Quaternions. [#1221]( https://github.com/sofa-framework/sofa/pull/1221 )
- **[SofaFramework]** Install the SofaSimulationCore target back into the SofaFramework package [#1182]( https://github.com/sofa-framework/sofa/pull/1182 )
- [SofaGuiQt] Fix unexpected symbol in CMakeLists [#1132]( https://github.com/sofa-framework/sofa/pull/1132 )
- [SofaGui] FIX missing find_package in SofaGuiConfig.cmake.in [#1198]( https://github.com/sofa-framework/sofa/pull/1198 )
- [SofaGui] Fix VideoRecorder [#1138]( https://github.com/sofa-framework/sofa/pull/1138 )
- [SofaGui] Prevent the GuiManager to store a pointer for the valid gui name [#1108]( https://github.com/sofa-framework/sofa/pull/1108 )
- [SofaHeadlessRecorder] FIX headlessRecorder_test [#1174]( https://github.com/sofa-framework/sofa/pull/1174 )
- **[SofaHelper]** FIX Eigen install path [#1240]( https://github.com/sofa-framework/sofa/pull/1240 )
- **[SofaKernel]** Add bloc access in basematrix [#1143]( https://github.com/sofa-framework/sofa/pull/1143 )
- **[SofaKernel]** Changes for Visual Studio and c++17 [#1162]( https://github.com/sofa-framework/sofa/pull/1162 )
- **[SofaKernel]** FIX regex in SofaMacros.cmake [#1161]( https://github.com/sofa-framework/sofa/pull/1161 )
- **[SofaKernel]** Fix alloc size [#1142]( https://github.com/sofa-framework/sofa/pull/1142 )
- **[SofaKernel]** Fix some AdvanceTimer log missing [#1158]( https://github.com/sofa-framework/sofa/pull/1158 )
- **[SofaKernel]** Fix useless warnings [#1144]( https://github.com/sofa-framework/sofa/pull/1144 )
- **[SofaKernel]** Several fix in draw methods and topology warnings [#1111]( https://github.com/sofa-framework/sofa/pull/1111 )
- **[SofaKernel]** Small Fix in CollisionModel [#1202]( https://github.com/sofa-framework/sofa/pull/1202 )
- **[SofaKernel]** Use links for input and output topologies of the barycentric mapping [#1125]( https://github.com/sofa-framework/sofa/pull/1125 )
- [SofaMisc] Fix compilation with SOFA_NO_OPENGL [#1193]( https://github.com/sofa-framework/sofa/pull/1193 )
- **[SofaSimulationGraph]** Fix CollisionGroupManager wrong search of deformable object node [#1060]( https://github.com/sofa-framework/sofa/pull/1060 )
- **[SofaSimulationGraph]** Stop DAGNode get parent topology process in BarycentricMapping [#1176]( https://github.com/sofa-framework/sofa/pull/1176 )
- [SofaSphFluid] Clean, Fix, Update ParticleSink [#1195]( https://github.com/sofa-framework/sofa/pull/1195 )

**Plugins / Projects**
- [All] Fix minor compilation issue in plugins [#1106]( https://github.com/sofa-framework/sofa/pull/1106 )
- [Carving plugin] Small fix at init. [#1110]( https://github.com/sofa-framework/sofa/pull/1110 )
- [Cgal plugin] Fix windows cmake dll path and add a scene example [#958]( https://github.com/sofa-framework/sofa/pull/958 )
- [Regression_test] Update regression test references for CollisionGroup [#1102]( https://github.com/sofa-framework/sofa/pull/1102 )


### Cleanings

**Architecture**
- [CMake] Use cmake_dependent_option for plugin tests [#1164]( https://github.com/sofa-framework/sofa/pull/1164 )

**Modules**
- [All] Fix order warnings [#1239]( https://github.com/sofa-framework/sofa/pull/1239 )
- [All] Fix override warning in option mode [#1210]( https://github.com/sofa-framework/sofa/pull/1210 )
- [All] Small cleaning on sout and serr [#1234]( https://github.com/sofa-framework/sofa/pull/1234 )
- [All] Standardize epsilons in SOFA [#1049]( https://github.com/sofa-framework/sofa/pull/1049 )
- [All] Code cleaning of multiple classes [#1116]( https://github.com/sofa-framework/sofa/pull/1116 )
- [All] Remove deprecated macro SOFA_TRANGLEFEM [#1233]( https://github.com/sofa-framework/sofa/pull/1233 )
- [All] Remove references to "isToPrint" because it's broken [#1197]( https://github.com/sofa-framework/sofa/pull/1197 )
- [All] Replace NULL by nullptr [#1179]( https://github.com/sofa-framework/sofa/pull/1179 )
- [All] Try to reduce the number of compilation warnings [#1196]( https://github.com/sofa-framework/sofa/pull/1196 )
- [SceneCreator] Pluginizing... [#1109]( https://github.com/sofa-framework/sofa/pull/1109 )
- **[SofaBaseLinearSolver]** Remove virtual function BaseLinearSolver::isMultiGroup [#1063]( https://github.com/sofa-framework/sofa/pull/1063 )
- **[SofaBaseLinearSolver][FullMatrix]**  Restore fast clear function [#1128]( https://github.com/sofa-framework/sofa/pull/1128 )
- **[SofaFramework]** Remove (painful) check/warning with Rigids [#1229]( https://github.com/sofa-framework/sofa/pull/1229 )
- [SofaGUI] Split OpenGL and Qt dependency [#1121]( https://github.com/sofa-framework/sofa/pull/1121 )
- [SofaGeneralObjectInteraction] Create delegate functions in AttachConstraint [#1185]( https://github.com/sofa-framework/sofa/pull/1185 )
- [SofaGraphComponent] Update sceneCheckerAPI and deprecate MatrixMass [#1107]( https://github.com/sofa-framework/sofa/pull/1107 )
- [SofaHAPI] Fixes for HAPI [#1189]( https://github.com/sofa-framework/sofa/pull/1189 )
- **[SofaKernel]** ADD const specifier on notify methods in Node [#1169]( https://github.com/sofa-framework/sofa/pull/1169 )
- **[SofaKernel]** Remove deprecated SOFA_DEBUG macro  [#1232]( https://github.com/sofa-framework/sofa/pull/1232 )
- **[SofaMeshCollision]** Clean deprecated code [#1201]( https://github.com/sofa-framework/sofa/pull/1201 )
- [SofaSphFluid] Clean code of ParticleSource and update scenes [#1190]( https://github.com/sofa-framework/sofa/pull/1190 )
- [SofaSphFluid] Reorder plugin code and scenes files [#1165]( https://github.com/sofa-framework/sofa/pull/1165 )
- [SofaSphFluid] Several clean in the plugin components [#1186]( https://github.com/sofa-framework/sofa/pull/1186 )
- [SofaSphFluid] missing namespace [#1188]( https://github.com/sofa-framework/sofa/pull/1188 )
- [SofaTest] CLEAN msg in Multi2Mapping_test [#1097]( https://github.com/sofa-framework/sofa/pull/1097 )
- [SofaTopologyMapping] Cleanups of some topological mappings + better initialization [#1126]( https://github.com/sofa-framework/sofa/pull/1126 )
- [SofaViewer] Prevent the GUI to ouput every CTRL actions in the console [#1130]( https://github.com/sofa-framework/sofa/pull/1130 )

**Plugins / Projects**
- [CGALPlugin] Some cleanups to CylinderMesh [#1124]( https://github.com/sofa-framework/sofa/pull/1124 )
- [CGal plugin][CImgPlugin] Image data moved from Image/ to CImgPlugin/ [#1104]( https://github.com/sofa-framework/sofa/pull/1104 )
- [Geomagic] Reorder plugin files for better modularization [#1208]( https://github.com/sofa-framework/sofa/pull/1208 )
- [ManifoldTopologies] Undust and clean [#1156]( https://github.com/sofa-framework/sofa/pull/1156 )

**Examples / Scenes**
- [Scenes] Clean some alias warnings [#1098]( https://github.com/sofa-framework/sofa/pull/1098 )
- [scenes] Change OglModel to use a MeshObjLoader instead of loading the mesh internally. [#1096]( https://github.com/sofa-framework/sofa/pull/1096 )


____________________________________________________________



## [v19.06]( https://github.com/sofa-framework/sofa/tree/v19.06 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v18.12...v19.06 )


### Breaking

**Modules**
- [All] Run clang-tidy and update license headers [#899]( https://github.com/sofa-framework/sofa/pull/899 )
- [All] Refactor the loading of Xsp files. [#918]( https://github.com/sofa-framework/sofa/pull/918 )
- **[SofaBaseTopology]** Change triangles orientation in tetrahedron [#878]( https://github.com/sofa-framework/sofa/pull/878 )
- **[SofaBaseTopology]** Major Change in Topology Containers [#967]( https://github.com/sofa-framework/sofa/pull/967 )
- **[SofaKernel]** Refactor the MutationListener [#917]( https://github.com/sofa-framework/sofa/pull/917 )
- **[SofaKernel]** Some Topology cleaning... [#866]( https://github.com/sofa-framework/sofa/pull/866 )
- [SofaOpenglVisual] Fix ogl perf problem [#1069]( https://github.com/sofa-framework/sofa/pull/1069 )


### Modularizations

- [SofaExporter] Modularize (+minor dependency cleaning) [#915]( https://github.com/sofa-framework/sofa/pull/915 )
- [SofaHaptics] Modularize sofa haptics [#945]( https://github.com/sofa-framework/sofa/pull/945 )
- [SofaOpenglVisual] Pluginize. [#1080]( https://github.com/sofa-framework/sofa/pull/1080 )


### Improvements

**Architecture**
- [CMake] Rework sofa_generate_package [#951]( https://github.com/sofa-framework/sofa/pull/951 )
- [CMake] SofaMacros.cmake: deprecating sofa_create_package [#909]( https://github.com/sofa-framework/sofa/pull/909 )

**Modules**
- [All] Improve install and packaging [#1018]( https://github.com/sofa-framework/sofa/pull/1018 )
- [All] Plugins finding and loading [#913]( https://github.com/sofa-framework/sofa/pull/913 )
- [All] Replace deprecated c++ standard binder component [#908]( https://github.com/sofa-framework/sofa/pull/908 )
- **[SofaBaseMechanics]** BarycentricMapping: spatial hashing, handle limit cases [#896]( https://github.com/sofa-framework/sofa/pull/896 )
- **[SofaBaseTopology]** Clean Topology logs and add AdvanceTimer logs [#874]( https://github.com/sofa-framework/sofa/pull/874 )
- **[SofaBaseVisual]** Add default texcoord in VisualModel [#933]( https://github.com/sofa-framework/sofa/pull/933 )
- [SofaConstraint] ADD control on constraint force in UniformConstraint [#1027]( https://github.com/sofa-framework/sofa/pull/1027 )
- **[SofaCore]** Add possibilities to draw lines on surfaces in DrawTool [#937]( https://github.com/sofa-framework/sofa/pull/937 )
- **[SofaCore]** Collision visitor primitive tests count [#930]( https://github.com/sofa-framework/sofa/pull/930 )
- **[SofaCore]** ADD Datacallback and datalink [#911]( https://github.com/sofa-framework/sofa/pull/911 )
- [SofaEngine] Avoid Crash in BoxROI when rest_position is not yet defined [#1031]( https://github.com/sofa-framework/sofa/pull/1031 )
- [SofaExporter] Add option for Regression_test to check first and last iteration [#1061]( https://github.com/sofa-framework/sofa/pull/1061 )
- [SofaGeneralAnimationLoop] Improve MechanicalMatrixMapper [#882]( https://github.com/sofa-framework/sofa/pull/882 )
- [SofaGraphComponent] Run SceneChecker at each load [#938]( https://github.com/sofa-framework/sofa/pull/938 )
- [SofaGuiQt] Change the keyboard shortcut associated to camera mode [#997]( https://github.com/sofa-framework/sofa/pull/997 )
- [SofaGuiQt] Add a profiling window based on AdvanceTimer records [#1028]( https://github.com/sofa-framework/sofa/pull/1028 )
- **[SofaKernel]** Some small changes in debug topology drawing [#952]( https://github.com/sofa-framework/sofa/pull/952 )
- **[SofaKernel]** Update Static Solver [#950]( https://github.com/sofa-framework/sofa/pull/950 )
- **[SofaKernel]** Rename TModels into CollisionModels and update all scenes [#1034]( https://github.com/sofa-framework/sofa/pull/1034 )
- **[SofaKernel]** Add a new video recorder class VideoRecorderFFMPEG [#883]( https://github.com/sofa-framework/sofa/pull/883 )
- **[SofaSimulationCore]** Cpu task and scheduled thread support [#970]( https://github.com/sofa-framework/sofa/pull/970 )
- **[SofaSimulationCore]** call BaseObject::draw() during the Transparent pass [#929]( https://github.com/sofa-framework/sofa/pull/929 )
- [SofaTopologyMapping] Clean, fix, upgrade Tetra2TriangleTopologicalMapping [#876]( https://github.com/sofa-framework/sofa/pull/876 )

**Plugins / Projects**
- [Geomagic] Add some better check at init and method to free driver [#925]( https://github.com/sofa-framework/sofa/pull/925 )
- [Icons] EDIT Sofa icons [#881]( https://github.com/sofa-framework/sofa/pull/881 )
- [MultiThreading] TaskAllocator Interface [#906]( https://github.com/sofa-framework/sofa/pull/906 )
- [PluginExample] Update example + add comments [#1053]( https://github.com/sofa-framework/sofa/pull/1053 )
- [Regression] ADD Regression as external project [#1052]( https://github.com/sofa-framework/sofa/pull/1052 )
- [runSofa] ADD possibility to jump to source/instanciation of selected component [#1013]( https://github.com/sofa-framework/sofa/pull/1013 )
- [SofaCUDA] Fix cuda with latest API [#912]( https://github.com/sofa-framework/sofa/pull/912 )
- [SofaPython] Add Sofa.hasViewer function [#964]( https://github.com/sofa-framework/sofa/pull/964 )
- [SofaPython] Change Base.addNewData [#1004]( https://github.com/sofa-framework/sofa/pull/1004 )

**Examples / Scenes**
- [examples] Rename TModels into CollisionModels and update all scenes [#1034]( https://github.com/sofa-framework/sofa/pull/1034 )


### Bug Fixes

**Architecture**
- [CMake] Add check to prevent the inclusion of non-existant file in cmake 3.13 [#897]( https://github.com/sofa-framework/sofa/pull/897 )
- [CMake] Fix relocatable plugins [#1059]( https://github.com/sofa-framework/sofa/pull/1059 )
- [CMake] FIX: exporting options in SofaFrameworkConfig.cmake [#927]( https://github.com/sofa-framework/sofa/pull/927 )
- [CMake] FIX: wrong paths of installed headers in SofaBaseMechanics [#887]( https://github.com/sofa-framework/sofa/pull/887 )
- [CMake] FIX build/install plugins directory [#959]( https://github.com/sofa-framework/sofa/pull/959 )

**Modules**
- [All] Three small fixes in SofaBaseLinearSolver, SofaBoundaryCondition, runSofa [#931]( https://github.com/sofa-framework/sofa/pull/931 )
- [All] FIXES made for RoboSoft2019 [#1003]( https://github.com/sofa-framework/sofa/pull/1003 )
- [All] Fix some warnings [#873]( https://github.com/sofa-framework/sofa/pull/873 )
- [All] Several bug fixes [#985]( https://github.com/sofa-framework/sofa/pull/985 )
- [All] Some fixes to have a ... green dashboard! [#982]( https://github.com/sofa-framework/sofa/pull/982 )
- [All] Fix compilation with SOFA_NO_OPENGL flag [#1032]( https://github.com/sofa-framework/sofa/pull/1032 )
- [SofaConstraint] Convert static sized arrays to dynamic ones in GenericConstraintSolver [#920]( https://github.com/sofa-framework/sofa/pull/920 )
- **[SofaBaseMechanics]** Fix barycentric mapping again [#924]( https://github.com/sofa-framework/sofa/pull/924 )
- **[SofaBaseTopology]** Fix Crash when loading a vtk file generated by Gmsh using TetrahedronSetTopologyContainer as container [#1008]( https://github.com/sofa-framework/sofa/pull/1008 )
- **[SofaBaseTopology]** Fix right setDirty/clean topologyData  [#889]( https://github.com/sofa-framework/sofa/pull/889 )
- **[SofaBaseTopology]**[DrawTools] Some fix/update in topology internal draw methods. [#877]( https://github.com/sofa-framework/sofa/pull/877 )
- **[SofaBaseTopology]** Yet another fix in Tetra2triangleTopologicalMapping [#998]( https://github.com/sofa-framework/sofa/pull/998 )
- **[SofaBaseTopology]** Clean, fix, upgrade Triangle2EdgeTopologicalMapping [#875]( https://github.com/sofa-framework/sofa/pull/875 )
- **[SofaBaseTopology]** Fix crashes in Tetra2TriangleTopologicalMapping  [#960]( https://github.com/sofa-framework/sofa/pull/960 )
- [SofaBoundaryCondition] Fix draw function in ConstantForcefield [#1017]( https://github.com/sofa-framework/sofa/pull/1017 )
- **[SofaDeformable]** FIX issue 928 [#942]( https://github.com/sofa-framework/sofa/pull/942 )
- **[SofaDeformable]** Merge 2 ctor in SpringForceField [#948]( https://github.com/sofa-framework/sofa/pull/948 )
- [SofaExporter] FIX: out-of-tree include of SofaExporter header files [#975]( https://github.com/sofa-framework/sofa/pull/975 )
- [SofaGeneralLoader] Compute subElement by default for Gmsh format [#986]( https://github.com/sofa-framework/sofa/pull/986 )
- [SofaGeneralObjectInteraction] Fix AttachConstraint in case of FreeMotion (LM solving) [#949]( https://github.com/sofa-framework/sofa/pull/949 )
- [SofaGeneralObjectInteraction] Fix attach constraint radius [#650]( https://github.com/sofa-framework/sofa/pull/650 )
- [SofaGui] Fix missing profiling timers for BatchGUI and HeadlessRecorder [#890]( https://github.com/sofa-framework/sofa/pull/890 )
- [SofaGuiGlut] Fix compilation [#1044]( https://github.com/sofa-framework/sofa/pull/1044 )
- [SofaGuiQt] FIX: component/nodes ordering in runSofa scene graph [#1001]( https://github.com/sofa-framework/sofa/pull/1001 )
- [SofaGuiQt] REMOVE: public export of target SofaExporter [#963]( https://github.com/sofa-framework/sofa/pull/963 )
- [SofaGuiQt] Fix: several QWidget do not have a parent [#1030]( https://github.com/sofa-framework/sofa/pull/1030 )
- **[SofaHelper]** FIX compilation on Visual Studio 2015 with QWT plugin [#935]( https://github.com/sofa-framework/sofa/pull/935 )
- **[SofaHelper]** FIX WinDepPack INSTALL_INTERFACE [#1042]( https://github.com/sofa-framework/sofa/pull/1042 )
- **[SofaHelper]** REMOVE PluginManager::m_searchPaths [#947]( https://github.com/sofa-framework/sofa/pull/947 )
- **[SofaKernel]** Clean & Fix TopologyChangeVisitor and StateChangeVisitor behavior [#880]( https://github.com/sofa-framework/sofa/pull/880 )
- **[SofaKernel]** Clean output data when doUpdate in BoxROI [#1056]( https://github.com/sofa-framework/sofa/pull/1056 )
- **[SofaKernel]** FIX deprecation message related to template types. [#939]( https://github.com/sofa-framework/sofa/pull/939 )
- **[SofaKernel]** FIX in TetrahedronFEMForceField & TetrahedronSetTopologyAlgorithm [#973]( https://github.com/sofa-framework/sofa/pull/973 )
- **[SofaKernel]** FIX operator>> in Mat.h and add corresponding test. [#993]( https://github.com/sofa-framework/sofa/pull/993 )
- **[SofaKernel]** FIX: A few fix to compile on Mac OSX Xcode 9 and Linux gcc 7.3.0 [#969]( https://github.com/sofa-framework/sofa/pull/969 )
- **[SofaKernel]** FIX: force name data to contain something [#1009]( https://github.com/sofa-framework/sofa/pull/1009 )
- **[SofaKernel]** Fix error in MapperHexahedron and MapperQuad barycentric coef computation [#1057]( https://github.com/sofa-framework/sofa/pull/1057 )
- **[SofaKernel]** Fix: remove unwanted AdvanceTimer::begin command [#1029]( https://github.com/sofa-framework/sofa/pull/1029 )
- **[SofaKernel]** Remove warnings [#968]( https://github.com/sofa-framework/sofa/pull/968 )
- **[SofaKernel]** several small fix [#953]( https://github.com/sofa-framework/sofa/pull/953 )
- [SofaLoader] Fix positions when handleSeams is activated in MeshObjLoader [#923]( https://github.com/sofa-framework/sofa/pull/923 )
- [SofaMeshCollision] Fix TriangleModel to handle topology changes [#903]( https://github.com/sofa-framework/sofa/pull/903 )
- **[SofaSimulationCore]** Remove unjustified Assert in getSimulation() [#1082]( https://github.com/sofa-framework/sofa/pull/1082 )
- **[SofaSimulationCore]** FIX CollisionVisitor::processCollisionPipeline [#962]( https://github.com/sofa-framework/sofa/pull/962 )
- [SofaTests] Fix small bugs in the Multi2Mapping_test [#1078]( https://github.com/sofa-framework/sofa/pull/1078 )

**Plugins / Projects**
- [CImgPlugin] FIX: messed up package prefix in CImg [#921]( https://github.com/sofa-framework/sofa/pull/921 )
- [Geomagic] FIX compilation error in Geomagic plugin with removal of SOFA_FLOAT/DOUBLE [#898]( https://github.com/sofa-framework/sofa/pull/898 )
- [image] Fix image_gui plugin loading [#1015]( https://github.com/sofa-framework/sofa/pull/1015 )
- [image] Message API is needed even if no python [#1068]( https://github.com/sofa-framework/sofa/pull/1068 )
- [runSofa] FIX the opening of ModifyObject view. [#1010]( https://github.com/sofa-framework/sofa/pull/1010 )
- [runSofa] Fix runSofa -a option with a gui. [#1058]( https://github.com/sofa-framework/sofa/pull/1058 )
- [runSofa] User experience fixes in the ModifyData view. [#1011]( https://github.com/sofa-framework/sofa/pull/1011 )
- [Sensable] Fix the compilation of the Sensable plugin [#1019]( https://github.com/sofa-framework/sofa/pull/1019 )
- [SofaCUDA] Compilation error fix (CudaStandardTetrahedralFEMForceField.cu) [#991]( https://github.com/sofa-framework/sofa/pull/991 )
- [SofaCUDA] Fix several Cuda example scenes [#1000]( https://github.com/sofa-framework/sofa/pull/1000 )
- [SofaCUDA] Fix windows compilation. [#966]( https://github.com/sofa-framework/sofa/pull/966 )
- [SofaPython] FIX allow the derivTypeFromParentValue to work with node. [#984]( https://github.com/sofa-framework/sofa/pull/984 )
- [SofaPython] FIX example broken by PR#459 [#1020]( https://github.com/sofa-framework/sofa/pull/1020 )
- [SofaPython] FIX the broken Binding_Data::setValue()  [#1006]( https://github.com/sofa-framework/sofa/pull/1006 )
- [SofaPython] Fix duplicate symbol [#1036]( https://github.com/sofa-framework/sofa/pull/1036 )
- [SofaPython] FIX: removing PythonLibs target from SofaPython [#891]( https://github.com/sofa-framework/sofa/pull/891 )
- [SofaPython] REMOVE: public export of target SofaExporter [#963]( https://github.com/sofa-framework/sofa/pull/963 )

**Examples / Scenes**
- [examples] Remove warnings in Demos/ scenes [#1021]( https://github.com/sofa-framework/sofa/pull/1021 )
- [scenes] Fix chainAll demo scenario [#987]( https://github.com/sofa-framework/sofa/pull/987 )


### Cleanings

**Modules**
- [All] For each data field's with a "filename" alias flip it with the data's name.  [#1024]( https://github.com/sofa-framework/sofa/pull/1024 )
- [All] Quick changes diffusion and mass [#983]( https://github.com/sofa-framework/sofa/pull/983 )
- [All] Remove duplicate ctor + prettify some code [#1054]( https://github.com/sofa-framework/sofa/pull/1054 )
- [All] Replace serr with the new msg_error() API. [#916]( https://github.com/sofa-framework/sofa/pull/916 )
- [All] Several STC fixes [#1048]( https://github.com/sofa-framework/sofa/pull/1048 )
- [All] Sofa defrost sprint week2 [#884]( https://github.com/sofa-framework/sofa/pull/884 )
- [All] minor cleaning of warnings and bugfix [#886]( https://github.com/sofa-framework/sofa/pull/886 )
- [All] Remove bunch of warnings (again) [#1065]( https://github.com/sofa-framework/sofa/pull/1065 )
- [All] remove #ifdef SOFA_HAVE_GLEW [#1077]( https://github.com/sofa-framework/sofa/pull/1077 )
- **[SofaLoader]** Change error into warning in MeshVTKLoader [#1037]( https://github.com/sofa-framework/sofa/pull/1037 )
- [SofaConstraint] Replaced sout calls by msg_info() in LCPConstraintSolver [#981]( https://github.com/sofa-framework/sofa/pull/981 )
- [SofaGeneralLinearSolver] Clean BTDLinearSolver [#907]( https://github.com/sofa-framework/sofa/pull/907 )
- [SofaHaptics] Replace deprecated INCLUDE_ROOT_DIR in CMakeLists.txt [#1023]( https://github.com/sofa-framework/sofa/pull/1023 )
- **[SofaKernel]** Brainless Warnings cleaning [#971]( https://github.com/sofa-framework/sofa/pull/971 )
- **[SofaKernel]** Minor code refactor in BaseData & new StringUtils functions. [#860]( https://github.com/sofa-framework/sofa/pull/860 )
- **[SofaKernel]** Refactor DataTrackerEngine so it match the DataCallback [#1073]( https://github.com/sofa-framework/sofa/pull/1073 )
- **[SofaKernel]** Remove annoying warning [#1062]( https://github.com/sofa-framework/sofa/pull/1062 )
- **[SofaKernel]** Remove boost::locale dependency [#1033]( https://github.com/sofa-framework/sofa/pull/1033 )
- **[SofaKernel]** Remove usage of helper::system::atomic<int> (replaced by STL's) [#1035]( https://github.com/sofa-framework/sofa/pull/1035 )
- **[SofaKernel]** Several changes in Topology components [#999]( https://github.com/sofa-framework/sofa/pull/999 )
- **[SofaKernel]** minor cleaning in mesh loader [#1025]( https://github.com/sofa-framework/sofa/pull/1025 )
- **[SofaKernel]** Remove multigroup option in MatrixLinearSolver [#901]( https://github.com/sofa-framework/sofa/pull/901 )
- [SofaRigid] Clean JointSpringFF [#850]( https://github.com/sofa-framework/sofa/pull/850 )
- [SofaRigid] Cosmetic clean in RigidRigidMapping & msg_* update. [#1005]( https://github.com/sofa-framework/sofa/pull/1005 )
- **[SofaSimpleFem]** Use msg and size_t in TetraDiff [#1016]( https://github.com/sofa-framework/sofa/pull/1016 )

**Plugins / Projects**
- [image] Add warning guiding users regarding pluginization of DiffusionSolver [#1067]( https://github.com/sofa-framework/sofa/pull/1067 )
- [Modeler] Deactivate Modeler by default, since it is deprecated [#972]( https://github.com/sofa-framework/sofa/pull/972 )

**Examples / Scenes**
- [Scenes] Apply script on all scenes using VisualModel/OglModel [#1081]( https://github.com/sofa-framework/sofa/pull/1081 )


____________________________________________________________



## [v18.12]( https://github.com/sofa-framework/sofa/tree/v18.12 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v18.06...v18.12 )


### Deprecated

**Removed in v18.12**
- [SofaBoundaryCondition] BuoyantForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaBoundaryCondition] VaccumSphereForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- **[SofaHelper]** Utils::getPluginDirectory() [#518]( https://github.com/sofa-framework/sofa/pull/518) - Use PluginRepository.getFirstPath( ) instead
- [SofaMisc] ParallelCGLinearSolver [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscForceField] ForceMaskOff [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscForceField] LineBendingSprings [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscForceField] WashingMachineForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- ~~[SofaMiscForceField] LennardJonesForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )~~
- [SofaMiscMapping] CatmullRomSplineMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] CenterPointMechanicalMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] CurveMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] ExternalInterpolationMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] ProjectionToLineMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] ProjectionToPlaneMapping
- ~~[SofaOpenglVisual] OglCylinderModel [#457]( https://github.com/sofa-framework/sofa/pull/457 )~~
- ~~[SofaOpenglVisual] OglGrid [#457]( https://github.com/sofa-framework/sofa/pull/457 )~~
- ~~[SofaOpenglVisual] OglRenderingSRGB [#457]( https://github.com/sofa-framework/sofa/pull/457 )~~
- ~~[SofaOpenglVisual] OglLineAxis [#457]( https://github.com/sofa-framework/sofa/pull/457 )~~
- ~~[SofaOpenglVisual] OglSceneFrame [#457]( https://github.com/sofa-framework/sofa/pull/457 )~~
- [SofaUserInteraction] ArticulatedHierarchyBVHController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] ArticulatedHierarchyController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] DisabledContact [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] EdgeSetController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] GraspingManager [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] InterpolationController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] MechanicalStateControllerOmni [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] NodeToggleController [#457]( https://github.com/sofa-framework/sofa/pull/457 )


### Breaking

**Modules**
- **[SofaBaseMechanics]**[SofaMiscForceField] Homogeneization of mass components in SOFA [#637]( https://github.com/sofa-framework/sofa/pull/637 )
- **[SofaBaseMechanics]** Clean barycentric mapping [#797]( https://github.com/sofa-framework/sofa/pull/797 )
- [SofaBoundaryCondition] Refactor FixedPlaneConstraint (breaking)  [#803]( https://github.com/sofa-framework/sofa/pull/803 )
- **[SofaFramework]** [BREAKING] Replacing DataEngine with SimpleDataEngine [#814]( https://github.com/sofa-framework/sofa/pull/814 )
- **[SofaFramework]** [BREAKING] Rename: data tracker has changed [#822]( https://github.com/sofa-framework/sofa/pull/822 )
- [SofaPreconditioner] modularization [#668]( https://github.com/sofa-framework/sofa/pull/668 )
- [SofaSparseSolver] modularization [#668]( https://github.com/sofa-framework/sofa/pull/668 )


### Improvements

**Architecture**
- [CMake] use ccache if available [#692]( https://github.com/sofa-framework/sofa/pull/692 )
- [Cmake] Add a findCython.cmake [#734]( https://github.com/sofa-framework/sofa/pull/734 )
- [CMake] ADD QtIFW generator + improvements [#796]( https://github.com/sofa-framework/sofa/pull/796 )
- [SofaMacros] ADD CMake macro to create pybind11 & cython targets and modules #859( https://github.com/sofa-framework/sofa/pull/859 )

**Modules**
- [All] Use drawtool everywhere [#704]( https://github.com/sofa-framework/sofa/pull/704 )
- [All] Sofa add mechanical matrix mapper [#721]( https://github.com/sofa-framework/sofa/pull/721 )
- **[SofaBaseTopology]** Add battery of tests on topology containers [#708]( https://github.com/sofa-framework/sofa/pull/708 )
- **[SofaBaseTopology]** Topology change propagation to Mechanical State [#838]( https://github.com/sofa-framework/sofa/pull/838 )
- **[SofaBaseMechanics]** Optimize barycentric mapping initialization [#798]( https://github.com/sofa-framework/sofa/pull/798 )
- [SofaBoundaryCondition] Factorize partial fixedconstraint [#718]( https://github.com/sofa-framework/sofa/pull/718 )
- [SofaConstraint] add a force data field in SlidingConstraint [#780]( https://github.com/sofa-framework/sofa/pull/780 )
- [SofaConstraint] ADD Data to show constraint forces [#840]( https://github.com/sofa-framework/sofa/pull/840 )
- [SofaConstraint] allow call of constraints' storeLambda() [#854]( https://github.com/sofa-framework/sofa/pull/854 )
- **[SofaCore]** Add new (simpler) DataEngine implementation [#760]( https://github.com/sofa-framework/sofa/pull/760 )
- [SofaExporter] ADD in WriteState all required tests on data and clean export with msg API [#714]( https://github.com/sofa-framework/sofa/pull/714 )
- **[SofaFramework]** Improve external dirs fetching in SofaMacros [#759]( https://github.com/sofa-framework/sofa/pull/759 )
- [SofaGeneralAnimationLoop] Improvement on MMMapper [#772]( https://github.com/sofa-framework/sofa/pull/772 )
- **[SofaHelper]** EDIT FileSystem and FileRepository for regression tests [#830]( https://github.com/sofa-framework/sofa/pull/830 )
- **[SofaKernel]** Improve Displayflags [#671]( https://github.com/sofa-framework/sofa/pull/671 )
- **[SofaKernel]** Add a "sofa_add_module" in SofaMacro.cmake [#732]( https://github.com/sofa-framework/sofa/pull/732 )
- **[SofaKernel]** use string in base object description [#862]( https://github.com/sofa-framework/sofa/pull/862 )
- [SofaMeshCollision] TriangleModel optimization when topology changes occur [#839]( https://github.com/sofa-framework/sofa/pull/839 )
- [SofaSparseSolver] ADD saveMatrixToFile to SparseLDLSolver [#845]( https://github.com/sofa-framework/sofa/pull/845 )
- [SofaTest] ADD a PrintTo method so test failure shows human readable informations. [#730]( https://github.com/sofa-framework/sofa/pull/730 )
- [VisualModel] Improve the messages when loading mesh inside VisualModel [#778]( https://github.com/sofa-framework/sofa/pull/778 )
- [WriteState] minor fix with the time attribute, default values [#776]( https://github.com/sofa-framework/sofa/pull/776 )

**Plugins / Projects**
- [Geomagic] ADD an inputForceFeedback data in Geomagic [#648]( https://github.com/sofa-framework/sofa/pull/648 )
- [Geomagic] Fix dll export and some enhancements [#786]( https://github.com/sofa-framework/sofa/pull/786 )
- [MultiThreading] removed the boost thread dependency [#701]( https://github.com/sofa-framework/sofa/pull/701 )
- [MultiThreading] New components and Task scheduler classes refactoring  [#745]( https://github.com/sofa-framework/sofa/pull/745 )
- [MultiThreading] Add Image plugin Data types in DataExchange component [#770]( https://github.com/sofa-framework/sofa/pull/770 )
- [MultiThreading] TaskScheduler Interface [#775]( https://github.com/sofa-framework/sofa/pull/775 )
- [runSofa] Add data field value change on mouse move [#750]( https://github.com/sofa-framework/sofa/pull/750 )
- [SofaCarving] Refresh and enhancement [#712]( https://github.com/sofa-framework/sofa/pull/712 )
- [SofaCarving] plugin enhancement [#787]( https://github.com/sofa-framework/sofa/pull/787 )
- [SofaPython] ADD forwarding of onMouseMove event into the script controller [#731]( https://github.com/sofa-framework/sofa/pull/731 )
- [SofaPython] ADD: Bindings for BoundingBox [#736]( https://github.com/sofa-framework/sofa/pull/736 )
- [SofaPython][PSDE] Psde derive io [#742]( https://github.com/sofa-framework/sofa/pull/742 )
- [SofaPython][PSDE] Update on demand as designed initially [#751]( https://github.com/sofa-framework/sofa/pull/751 )
- [SofaPython] ADD a custom __dir__ method in Binding_Base. [#762]( https://github.com/sofa-framework/sofa/pull/762 )
- [SofaPython] add getLinkedBase to the binding of a link. [#843]( https://github.com/sofa-framework/sofa/pull/843 )
- [SofaPython] ADD binding python to getCategories [#868]( https://github.com/sofa-framework/sofa/pull/868 )


### Bug Fixes

**Architecture**
- [CMake] FIX: cyclic recursion [#766]( https://github.com/sofa-framework/sofa/pull/766 )
- [CMake] Backport fixes [#791]( https://github.com/sofa-framework/sofa/pull/791 )
- [CMake] Fix compilation issues due to CPackNSIS [#867]( https://github.com/sofa-framework/sofa/pull/867 )
- [CMake] Add check to prevent the inclusion of non-existant file in cmake 3.13 [#897]( https://github.com/sofa-framework/sofa/pull/897 )

**Modules**
- [All] ISSofa bugfix, lot of fixes [#756]( https://github.com/sofa-framework/sofa/pull/756 )
- [All] FIX Windows linkage [#910]( https://github.com/sofa-framework/sofa/pull/910 )
- [SofaGuiQt] Change method to allow antialiased screenshots in QtViewer [#728]( https://github.com/sofa-framework/sofa/pull/728 )
- **[SofaBaseMechanics]** Fix warning scene mass [#779]( https://github.com/sofa-framework/sofa/pull/779 )
- **[SofaBaseMechanics]** FIX DiagonalMass_test [#832]( https://github.com/sofa-framework/sofa/pull/832 )
- **[SofaBaseTopology]** Fix SparseGridTopology initialization with an input mesh [#670]( https://github.com/sofa-framework/sofa/pull/670 )
- [SofaBoundaryCondition] FIX AffineMovementConstraint orientation issue [#824]( https://github.com/sofa-framework/sofa/pull/824 )
- [SofaCarving] Modify the CMake config file to allow other plugins link to Sofa Carving  [#781]( https://github.com/sofa-framework/sofa/pull/781 )
- **[SofaCore]** FIX: enable ExtVecXf mappings with double floating type [#827]( https://github.com/sofa-framework/sofa/pull/827 )
- [SofaDeformable] Fix MeshSpring ForceField and Loader [#815]( https://github.com/sofa-framework/sofa/pull/815 )
- **[SofaFramework]** Keep SOFA_EXTERN_TEMPLATE macro definition [#870]( https://github.com/sofa-framework/sofa/pull/870 )
- [SofaGui] ADD option to enable VSync (default: OFF) [#722]( https://github.com/sofa-framework/sofa/pull/722 )
- [SofaOpenglVisual] Rollback removal of Ogl components [#905]( https://github.com/sofa-framework/sofa/pull/905 )
- **[SofaKernel]** FIX bug in toEulerVector [#399]( https://github.com/sofa-framework/sofa/pull/399 )
- **[SofaKernel]** FIX segfault created by static initialisers on OSX/clang compiler [#642]( https://github.com/sofa-framework/sofa/pull/642 )
- **[SofaKernel]** Fix: correct path writing in sofa_set_python_directory macro [#763]( https://github.com/sofa-framework/sofa/pull/763 )
- **[SofaKernel]** Check if Quaternion has norm 1 [#790]( https://github.com/sofa-framework/sofa/pull/790 )
- [SofaPreconditioner] FIX ShewchukPCGLinearSolver [#737]( https://github.com/sofa-framework/sofa/pull/737 )

**Plugins / Projects**
- [CGALPlugin] fix compilation [#783]( https://github.com/sofa-framework/sofa/pull/783 )
- [CGALPlugin] Fix compilation for cgal 4.12+ (cgal::polyhedron_3 is now forward declared) [#812]( https://github.com/sofa-framework/sofa/pull/812 )
- [CImgPlugin] CMake/Mac: lower priority for XCode's libpng [#720]( https://github.com/sofa-framework/sofa/pull/720 )
- [Geomagic] Fix broken behavior [#761]( https://github.com/sofa-framework/sofa/pull/761 )
- [Geomagic] Fix scenes [#858]( https://github.com/sofa-framework/sofa/pull/858 )
- [Multithreading] FIX compiling error on Mac [#727]( https://github.com/sofa-framework/sofa/pull/727 )
- [MultiThreading] Fix Livers scenes crash  [#792]( https://github.com/sofa-framework/sofa/pull/792 )
- [runSofa] ADD Modules to plugin_list.conf.default [#753]( https://github.com/sofa-framework/sofa/pull/753 )
- [SofaPython][examples] FIX: Fixing the scene... and the typo in the name [#765]( https://github.com/sofa-framework/sofa/pull/765 )
- [Tutorials] FIX oneTetrahedron and chainHybrid [#773]( https://github.com/sofa-framework/sofa/pull/773 )

**Examples / Scenes**
- [examples] Fix TopologySurfaceDifferentMesh.scn [#716]( https://github.com/sofa-framework/sofa/pull/716 )
- [examples] Fix the examples  missing a <RequiredPlugin name="SofaSparseSolver"/> [#748]( https://github.com/sofa-framework/sofa/pull/748 )
- [examples] Fix scenes having issue with CollisionGroup [#821]( https://github.com/sofa-framework/sofa/pull/821 )


### Cleanings

**Modules**
- [All] Fix some recent compilation warnings [#726]( https://github.com/sofa-framework/sofa/pull/726 )
- [All] Replace some int/unit by size_t [#729]( https://github.com/sofa-framework/sofa/pull/729 )
- [All] Fix some C4661 warnings [#738]( https://github.com/sofa-framework/sofa/pull/738 )
- [All] FIX warnings [#739]( https://github.com/sofa-framework/sofa/pull/739 )
- [All] Fix some compilation warnings [#755]( https://github.com/sofa-framework/sofa/pull/755 )
- [All] FIX a bunch of static-analysis errors (cppcheck) [#782]( https://github.com/sofa-framework/sofa/pull/782 )
- [All] Remove SOFA_DECL_CLASS and SOFA_LINK_CLASS [#837]( https://github.com/sofa-framework/sofa/pull/837 )
- [All] Remove SOFA_SUPPORT_MOVING_FRAMES and SOFA_SUPPORT_MULTIRESOLUTION [#849]( https://github.com/sofa-framework/sofa/pull/849 )
- **[SofaBaseCollision]** CLEAN CylinderModel [#847]( https://github.com/sofa-framework/sofa/pull/847 )
- **[SofaBaseLinearSolver]** CLEAN GraphScatteredTypes [#844]( https://github.com/sofa-framework/sofa/pull/844 )
- **[SofaFramework]** Various cleaning (non-breaking) [#841]( https://github.com/sofa-framework/sofa/pull/841 )
- **[SofaFramework]** CLEAN: removing unused PS3 files [#851]( https://github.com/sofa-framework/sofa/pull/851 )
- [SofaGeneralSimpleFEM] Clean BeamFemForceField [#846]( https://github.com/sofa-framework/sofa/pull/846 )
- **[SofaHelper]** Change drawTriangle and drawQuad with internal functions [#813]( https://github.com/sofa-framework/sofa/pull/813 )
- **[SofaHelper]** Update ComponentChange with removed Components [#905]( https://github.com/sofa-framework/sofa/pull/905 )
- **[SofaKernel]** Remove commented code since years in SofaBaseMechanics [#733]( https://github.com/sofa-framework/sofa/pull/733 )
- **[SofaKernel]** Move ScriptEvent class from SofaPython to core/objectModel [#764]( https://github.com/sofa-framework/sofa/pull/764 )
- [SofaMiscFem] Clean BaseMaterial::handleTopologyChange [#817]( https://github.com/sofa-framework/sofa/pull/817 )
- [SofaMiscMapping] Clean DeformableOnRigidFrameMapping [#848]( https://github.com/sofa-framework/sofa/pull/848 )
- **[SofaSimpleFem]** Stc clean simplefem [#826]( https://github.com/sofa-framework/sofa/pull/826 )

**Plugins / Projects**
- [Multithreading] Move TaskScheduler files from MultiThreading plugin to SofaKernel [#805]( https://github.com/sofa-framework/sofa/pull/805 )

**Examples / Scenes**
- [examples] Remove scenes about deprecated components [#922]( https://github.com/sofa-framework/sofa/pull/922 )


____________________________________________________________



## [v18.06]( https://github.com/sofa-framework/sofa/tree/v18.06 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v17.12...v18.06 )


### Deprecated

**Will be removed in v18.06**
- **[SofaHelper]** Utils::getPluginDirectory() [#518]( https://github.com/sofa-framework/sofa/pull/518) - Use PluginRepository.getFirstPath( ) instead

**Will be removed in v18.12**
- [SofaBoundaryCondition] BuoyantForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaBoundaryCondition] VaccumSphereForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMisc] ParallelCGLinearSolver [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscForceField] ForceMaskOff [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscForceField] LineBendingSprings [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscForceField] WashingMachineForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscForceField] LennardJonesForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] CatmullRomSplineMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] CenterPointMechanicalMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] CurveMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] ExternalInterpolationMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] ProjectionToLineMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaMiscMapping] ProjectionToPlaneMapping
- [SofaOpenglVisual] OglCylinderModel [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaOpenglVisual] OglGrid [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaOpenglVisual] OglRenderingSRGB [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaOpenglVisual] OglLineAxis [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaOpenglVisual] OglSceneFrame
- [SofaUserInteraction] ArticulatedHierarchyBVHController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] ArticulatedHierarchyController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] DisabledContact [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] EdgeSetController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] GraspingManager [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] InterpolationController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] MechanicalStateControllerOmni [#457]( https://github.com/sofa-framework/sofa/pull/457 )
- [SofaUserInteraction] NodeToggleController [#457]( https://github.com/sofa-framework/sofa/pull/457 )


### Breaking

**Modules**
- [SofaConstraint] Update FreeMotionAnimationLoop so that it can compute a linearised version of the constraint force. [#459]( https://github.com/sofa-framework/sofa/pull/459 )
- **[SofaCore]** Update FreeMotionAnimationLoop so that it can compute a linearised version of the constraint force. [#459]( https://github.com/sofa-framework/sofa/pull/459 )
- **[SofaHelper]** Unifying the way we report file related errors [#669]( https://github.com/sofa-framework/sofa/pull/669 )


### Improvements

**Architecture**
- [CMake] ADD external projects handling [#649]( https://github.com/sofa-framework/sofa/pull/649 )
- [CMake] ADD the CMAKE_WARN_DEPRECATED option in SOFA [#662]( https://github.com/sofa-framework/sofa/pull/662 )
- [CMake] Improve SOFA installation and packaging [#635]( https://github.com/sofa-framework/sofa/pull/635 )
- [CMake] Cleans for packaging [#789]( https://github.com/sofa-framework/sofa/pull/789 )

**Modules**
- [All] Refactoring in Camera, BackgroundSetting and Light [#676]( https://github.com/sofa-framework/sofa/pull/676 )
- **[SofaBaseLinearSolver]** Improve warning emission for CG [#658]( https://github.com/sofa-framework/sofa/pull/658 )
- **[SofaBaseLinearSolver]** Add ability to activate printing of debug information at runtime [#667]( https://github.com/sofa-framework/sofa/pull/667 )
- [SofaGeneralImplicitOdeSolver] FIX data field name in VariationalSymplecticSolver [#624]( https://github.com/sofa-framework/sofa/pull/624 )
- [SofaGraphComponent] ADD alias usage detection [#702]( https://github.com/sofa-framework/sofa/pull/702 )
- **[SofaLoader]** ADD support to load VTK polylines in legacy formated files [#576]( https://github.com/sofa-framework/sofa/pull/576 )
- [SofaMiscMapping] Fix rigid barycentric mapping [#710]( https://github.com/sofa-framework/sofa/pull/710 )
- **[SofaHelper]** PluginManager now checks for file existence instead of library extension match. [#621]( https://github.com/sofa-framework/sofa/pull/621 )

**Plugins / Projects**
- [HeadlessRecorder] ADD frameskip option to headless recorder [#615]( https://github.com/sofa-framework/sofa/pull/615 )
- [HeadlessRecorder] Remove avcodec dependency in HeadlessRecorder.h [#752]( https://github.com/sofa-framework/sofa/pull/752 )
- [runSofa] Save&restore the scenegraph state when live-code & add info panel [#657]( https://github.com/sofa-framework/sofa/pull/657 )
- [SofaPython] PythonScriptDataEngine (PSDE) [#583]( https://github.com/sofa-framework/sofa/pull/583 )
- [SofaPython] Small fix & new features. [#656]( https://github.com/sofa-framework/sofa/pull/656 )

**Tools**
- [tools] FIX sofa-launcher stdout [#592]( https://github.com/sofa-framework/sofa/pull/592 )


### Bug Fixes

**Modules**
- [All] FIX VS2017 build (Windows) [#630]( https://github.com/sofa-framework/sofa/pull/630 )
- [All] Fix computeBBox() [#634]( https://github.com/sofa-framework/sofa/pull/634 )
- [All] FIX warnings [#584]( https://github.com/sofa-framework/sofa/pull/584 )
- [All] Various small changes in error messages & bugfix  from defrost branches [#660]( https://github.com/sofa-framework/sofa/pull/660 )
- [SofaConstraint] FIX: Moving semicolon under preprocessor define [#680]( https://github.com/sofa-framework/sofa/pull/680 )
- **[SofaEngine]** FIX Bug in BoxROI that is not properly initialized [#627]( https://github.com/sofa-framework/sofa/pull/627 )
- **[SofaFramework]** Fix plugin list configuration [#645]( https://github.com/sofa-framework/sofa/pull/645 )
- [SofaGraphComponent] FIX SceneChecker_test + ADD alias test [#711]( https://github.com/sofa-framework/sofa/pull/711 )
- [SofaGraphComponent] FIX SceneCheck build on MacOS [#719]( https://github.com/sofa-framework/sofa/pull/719 )
- [SofaGuiQt] FIX missing resources [#758]( https://github.com/sofa-framework/sofa/pull/758 )
- [SofaGeneralEngine] FIX disabled tests [#675]( https://github.com/sofa-framework/sofa/pull/675 )
- **[SofaHelper]** More robust method to test end of string [#617]( https://github.com/sofa-framework/sofa/pull/617 )
- **[SofaKernel]** FIX macro issue resulted from the #include cleaning. [#672]( https://github.com/sofa-framework/sofa/pull/672 )
- [SofaMiscFem] FIX dependencies [#588]( https://github.com/sofa-framework/sofa/pull/588 )
- [SofaOpenglVisual] FIX MacOS crash in batch mode [#646]( https://github.com/sofa-framework/sofa/pull/646 )
- **[SofaSimulationGraph]** FIX dependencies [#588]( https://github.com/sofa-framework/sofa/pull/588 )
- [SofaSparseSolver] FIX SparseLDL crash and add proper SOFA_FLOAT/DOUBLE mangement [#655]( https://github.com/sofa-framework/sofa/pull/655 )

**Plugins / Projects**
- [CGALPlugin] FIX compilation issue with recent version of CGAL (4.11) & Ubunu 18.04 LTS [#664]( https://github.com/sofa-framework/sofa/pull/664 )
- [CImgPlugin] Export CImg_CFLAGS [#595]( https://github.com/sofa-framework/sofa/pull/595 )
- [CImgPlugin] FIX CMakeLists install fail since pluginization [#609]( https://github.com/sofa-framework/sofa/pull/609 )
- [CImgPlugin] FIX malformed cflag append [#622]( https://github.com/sofa-framework/sofa/pull/622 )
- [HeadlessRecorder] Fix headless recorder stream definition [#666]( https://github.com/sofa-framework/sofa/pull/666 )
- [MultiThreading] FIX: add createSubelements param in MeshGmshLoader [#626]( https://github.com/sofa-framework/sofa/pull/626 )
- [runSofa] Fix compilation when SofaGuiQt is not activated [#599]( https://github.com/sofa-framework/sofa/pull/599 )
- [runSofa] ADD infinite iterations option to batch gui [#613]( https://github.com/sofa-framework/sofa/pull/613 )
- [runSofa] FIX missing resources [#758]( https://github.com/sofa-framework/sofa/pull/758 )
- [SofaDistanceGrid] ADD .scene-tests to ignore scene [#594]( https://github.com/sofa-framework/sofa/pull/594 )
- [SofaPython] FIX build for MacOS >10.13.0 [#614]( https://github.com/sofa-framework/sofa/pull/614 )

**Examples / Scenes**
- FIX collision of the fontain example [#612]( https://github.com/sofa-framework/sofa/pull/612 )
- FIX failing scenes on CI [#641]( https://github.com/sofa-framework/sofa/pull/641 )
- FIX missing RequiredPlugin [#628]( https://github.com/sofa-framework/sofa/pull/628 )

**Extlibs**
- [extlibs/gtest] Update gtest & clean the CMakeLists.txt [#604]( https://github.com/sofa-framework/sofa/pull/604 )


### Cleanings

**Architecture**
- [CMake] Remove the option SOFA_GUI_INTERACTION and its associated codes/macro [#643]( https://github.com/sofa-framework/sofa/pull/643 )

**Modules**
- [All] CMake: Remove COMPONENTSET, keep DEPRECATED [#586]( https://github.com/sofa-framework/sofa/pull/586 )
- [All] CLEAN topology classes [#693]( https://github.com/sofa-framework/sofa/pull/693 )
- **[SofaHelper]** CLEAN commented code and double parentheses in Messaging.h [#587]( https://github.com/sofa-framework/sofa/pull/587 )
- **[SofaKernel]** Header include cleanup [#638]( https://github.com/sofa-framework/sofa/pull/638 )
- **[SofaKernel]** Remove unused function "renumberConstraintId" [#691]( https://github.com/sofa-framework/sofa/pull/691 )

**Plugins / Projects**
- [CImgPlugin] Less scary config warnings [#607]( https://github.com/sofa-framework/sofa/pull/607 )
- [HeadlessRecorder] Handle errors in target config [#608]( https://github.com/sofa-framework/sofa/pull/608 )
- [SofaGUI] Move GlutGUI to projects and remove all glut references in SofaFramework [#598]( https://github.com/sofa-framework/sofa/pull/598 )
- [SofaGUI] CMake: Remove useless if block in qt CMakelists.txt [#590]( https://github.com/sofa-framework/sofa/pull/590 )
- [SofaPhysicsAPI] FIX: remove the include of glut [#659]( https://github.com/sofa-framework/sofa/pull/659 )


____________________________________________________________



## [v17.12]( https://github.com/sofa-framework/sofa/tree/v17.12 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v17.06...v17.12 )


### Deprecated

**Kernel modules**
- Will be removed in v17.12
    - [All]
        - SMP support [#457]( https://github.com/sofa-framework/sofa/pull/457 - no more maintained )
    - [SofaDefaultType]
        - LaparoscopicRigidType [#457]( https://github.com/sofa-framework/sofa/pull/457 ) - not used/dont compiled for a really long time

- Will be removed in v18.06
    - [SofaHelper]
        - Utils::getPluginDirectory() [#518]( https://github.com/sofa-framework/sofa/pull/518) - Use PluginRepository.getFirstPath( ) instead

**Other modules**
- Will be removed in v18.12
    - [SofaBoundaryCondition]
        - BuoyantForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - VaccumSphereForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
    - [SofaMisc]
        - ParallelCGLinearSolver [#457]( https://github.com/sofa-framework/sofa/pull/457 )
    - [SofaMiscForceField]
        - ForceMaskOff [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - LineBendingSprings [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - WashingMachineForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
		- LennardJonesForceField [#457]( https://github.com/sofa-framework/sofa/pull/457 )
    - [SofaMiscMapping]
        - CatmullRomSplineMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - CenterPointMechanicalMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - CurveMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - ExternalInterpolationMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - ProjectionToLineMapping [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - ProjectionToPlaneMapping
    - [SofaOpenglVisual]
        - OglCylinderModel [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - OglGrid [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - OglRenderingSRGB [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - OglLineAxis [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - OglSceneFrame
    - [SofaUserInteraction]
        - AddRecordedCameraPerformer [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - ArticulatedHierarchyBVHController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - ArticulatedHierarchyController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - DisabledContact [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - EdgeSetController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - FixParticlePerformer [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - GraspingManager [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - InciseAlongPathPerformer [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - InterpolationController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - MechanicalStateControllerOmni [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - NodeToggleController [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - RemovePrimitivePerformer [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - StartNavigationPerformer [#457]( https://github.com/sofa-framework/sofa/pull/457 )
        - SuturePointPerformer [#457]( https://github.com/sofa-framework/sofa/pull/457 )


### Breaking

**Kernel modules**
- [All]
    - issofa_visitors: Changing the way projective constraints are propagated in visitors [#216]( https://github.com/sofa-framework/sofa/pull/216 )
- [SofaDeformable]
    - Change how rest shape is given in RestShapeSpringsForceField [#315]( https://github.com/sofa-framework/sofa/pull/315 )

**Other modules**
- [SofaHelper]
    - Rewrite ArgumentParser [#513]( https://github.com/sofa-framework/sofa/pull/513 )
- [SofaValidation]
    - ADD Monitor test [#312]( https://github.com/sofa-framework/sofa/pull/312 )


### Improvements

**Kernel modules**
- [All]
    - issofa_topology: Improvement, BugFix and Cleaning on Topology [#243]( https://github.com/sofa-framework/sofa/pull/243 )
    - issofa_constraintsolving: improve constraints [#484]( https://github.com/sofa-framework/sofa/pull/484 )
    - Improve File:line info in error message (for python and xml error reporting) [#314]( https://github.com/sofa-framework/sofa/pull/314 )
- [SofaDeformable]
    - issofa_meshspringff [#497]( https://github.com/sofa-framework/sofa/pull/497 )
    - issofa_springff [#498]( https://github.com/sofa-framework/sofa/pull/498 )
- [SofaHelper]
    - add ability to use NoArgument  for BaseCreator and Factory [#385]( https://github.com/sofa-framework/sofa/pull/385 )
    - Remove legacy ImageBMP/ImagePNG and ImageQt [#424]( https://github.com/sofa-framework/sofa/pull/424 )
    - Improve advanced timer [#468]( https://github.com/sofa-framework/sofa/pull/468 )
- [SofaLoader]
    - ADD normals and vectors to Legacy vtk import [#536]( https://github.com/sofa-framework/sofa/pull/536 )
- [SofaSimpleFem]
    - Add check of vector size in TetrahedronFEMForceField [#341]( https://github.com/sofa-framework/sofa/pull/341 )

**Other modules**
- [All]
    - Fix default value rayleigh params [#350]( https://github.com/sofa-framework/sofa/pull/350 )
    - PSL branch prerequisites [#410]( https://github.com/sofa-framework/sofa/pull/410 )
    - template alias sptr for downsizing the include graph [#436]( https://github.com/sofa-framework/sofa/pull/436 )
    - Removing the typedef files + SOFA_DECL + SOFA_LINK [#453]( https://github.com/sofa-framework/sofa/pull/453 )
    - CHANGE USE_MASK option to off by default [#532]( https://github.com/sofa-framework/sofa/pull/532 )
- [SofaBoundaryCondition]
    - ADD flag PROJECTVELOCITY [#288]( https://github.com/sofa-framework/sofa/pull/288 )
    - make LinearMovementConstraint accept absolute movements [#394]( https://github.com/sofa-framework/sofa/pull/394 )
- [SofaCore]
    - Add some read/write free method to construct Data Read/WriteAccessor [#450]( https://github.com/sofa-framework/sofa/pull/450 )
- [SofaDefaulttype]
    - MapMapSparseMatrix conversion utils with Eigen Sparse [#452]( https://github.com/sofa-framework/sofa/pull/452 )
    - multTranspose method between MapMapSparseMatrix and BaseVector [#456]( https://github.com/sofa-framework/sofa/pull/456 )
- [SofaDeformable]
    - Rest shape can now be given using SingleLink [#315]( https://github.com/sofa-framework/sofa/pull/315 )
    - Add AngularSpringForceField [#334]( https://github.com/sofa-framework/sofa/pull/334 )
- [SofaEulerianFluid]
    - Pluginizing the module [#396]( https://github.com/sofa-framework/sofa/pull/396 )
- [SofaExporter]
    - Clean & test the exporter  [#372]( https://github.com/sofa-framework/sofa/pull/372 )
- [SofaGraphComponent]
    - Add SceneCheckerVisitor to detect missing RequiredPlugin [#306]( https://github.com/sofa-framework/sofa/pull/306 )
    - Add a mechanism (SceneChecker) to report API & SceneChange to users [#329]( https://github.com/sofa-framework/sofa/pull/329 )
    - Refactor the SceneChecker and add a new SceneChecker to test dumplicated names [#392]( https://github.com/sofa-framework/sofa/pull/392 )
- [SofaGeneralEngine]
    - Add test and minor cleaning for IndexValueMapper [#319]( https://github.com/sofa-framework/sofa/pull/319 )
    - Add a computeBBox to the SmoothMeshEngine [#409]( https://github.com/sofa-framework/sofa/pull/409 )
- [SofaGeneralObjectInteraction]
    - issofa_attachconstraint [#501]( https://github.com/sofa-framework/sofa/pull/501 )
- [SofaMisc]
    - Add a data repository at start-up time [#402]( https://github.com/sofa-framework/sofa/pull/402 )
- [SofaMiscCollision]
    - Pluginizing the module [#453]( https://github.com/sofa-framework/sofa/pull/453 )
- [SofaMiscFem]
    - Add hyperelasticity fem code in SOFA [#349]( https://github.com/sofa-framework/sofa/pull/349 )
- [SofaSimpleFem]
    - ADD computeBBox and test to HexaFEMForceField [#289]( https://github.com/sofa-framework/sofa/pull/289 )
- [SofaSphFluid]
    - Pluginizing the module [#453]( https://github.com/sofa-framework/sofa/pull/453 )
- [SofaVolumetricData]
    - Split the module in two plugins [#389]( https://github.com/sofa-framework/sofa/pull/389 )

**Plugins / Projects**
- [CGALPlugin]
    - Add new functionality for mesh generation from image: definition of features [#294]( https://github.com/sofa-framework/sofa/pull/294 )
- [meshconv]
    - Improve the CMake config of meshconv requiring miniflowVR to compile [#358]( https://github.com/sofa-framework/sofa/pull/358 )
- [PSL]
    - Experimental : Add PSL for 17.12 [#541]( https://github.com/sofa-framework/sofa/pull/541 )
- [runSofa]
    - autoload plugins (2nd version) [#301]( https://github.com/sofa-framework/sofa/pull/301 )
    - Extend the live coding support, message API available for nodes, add an openInEditor [#337]( https://github.com/sofa-framework/sofa/pull/337 )
    - add verification if help is not null from displayed data [#382]( https://github.com/sofa-framework/sofa/pull/382 )
    - improve the html DocBrowser  [#540]( https://github.com/sofa-framework/sofa/pull/540 )
- [SceneChecker]
    - Add mechanism to report API & SceneChange to users [#329]( https://github.com/sofa-framework/sofa/pull/329 )
- [SofaDistanceGrid]
    - Pluginizing SofaVolumetricData [#389]( https://github.com/sofa-framework/sofa/pull/389 )
- [SofaImplicitField]
    - Pluginizing SofaVolumetricData [#389]( https://github.com/sofa-framework/sofa/pull/389 )
- [SofaMiscCollision]
    - Pluginizing the module [#453]( https://github.com/sofa-framework/sofa/pull/453 )
- [SofaPython]
    - Add unicode to string convertion and a warning message in Binding_BaseContext::pythonToSofaDataString [#313]( https://github.com/sofa-framework/sofa/pull/313 )
    - Add unicode to string convertion in Binding_BaseData::SetDataValuePython [#313]( https://github.com/sofa-framework/sofa/pull/313 )
    - Add a test [#313]( https://github.com/sofa-framework/sofa/pull/313 )
    - Add GIL management [#326]( https://github.com/sofa-framework/sofa/pull/326 )
    - Add support for Sofa.msg_ with emitter other than a string [#335]( https://github.com/sofa-framework/sofa/pull/335 )
    - Adding new features for AdvancedTimer [#360]( https://github.com/sofa-framework/sofa/pull/360 )
    - forward sys.argv to python scripts [#368]( https://github.com/sofa-framework/sofa/pull/368 )
    - sparse matrix aliasing scipy/eigen [#411]( https://github.com/sofa-framework/sofa/pull/411 )
- [SofaSphFluid]
    - Pluginizing the module [#453]( https://github.com/sofa-framework/sofa/pull/453 )

**Other**
- [Tools]
    - Update astyle config [#550]( https://github.com/sofa-framework/sofa/pull/550 )


### Bug Fixes

**Kernel modules**
- [All]
    - CMake: Fix and clean boost, when using Sofa as an external lib [#421]( https://github.com/sofa-framework/sofa/pull/421 )
    - Fix computeBBox functions [#527]( https://github.com/sofa-framework/sofa/pull/527 )
    - CMake: FIX Boost::program_options finding in install [#618]( https://github.com/sofa-framework/sofa/pull/618 )
- [SofaBaseLinearSolver]
    - FIX no step if condition on denominator is met at first step [#521]( https://github.com/sofa-framework/sofa/pull/521 )
    - FIX breaking condition in CG at first step regarding threshold [#556]( https://github.com/sofa-framework/sofa/pull/556 )
- [SofaBaseMechanics]
    - Make sure the mechanical object's state vectors size matches their respective argument size [#406]( https://github.com/sofa-framework/sofa/pull/406 )
- [SofaBaseTopology]
    - FIX wrong clean in PointSetTopologyModifier [#380]( https://github.com/sofa-framework/sofa/pull/380 )
- [SofaComponentCommon]
    - Always register all its components in the object factory [#488]( https://github.com/sofa-framework/sofa/pull/488 )
- [SofaCore]
    - FIX CreateString problem on root node [#377]( https://github.com/sofa-framework/sofa/pull/377 )
    - FIX don't inline exported functions [#449]( https://github.com/sofa-framework/sofa/pull/449 )
- [SofaDefaultType]
    - FIX Mat::transpose() and Mat::invert() [#317]( https://github.com/sofa-framework/sofa/pull/317 )
    - Correct CMake include_directories directive for SofaDefaultType target's [#403]( https://github.com/sofa-framework/sofa/pull/403 )
    - Fix compilation errors when working with transform class [#506]( https://github.com/sofa-framework/sofa/pull/506 )
- [SofaHelper]
    - Fix CUDA compilation with pointer of data [#320]( https://github.com/sofa-framework/sofa/pull/320 )
    - FIX livecoding of shaders [#415]( https://github.com/sofa-framework/sofa/pull/415 )
    - fixing Polynomial_LD [#442]( https://github.com/sofa-framework/sofa/pull/442 )
    - Replacing afficheResult with resultToString [#473]( https://github.com/sofa-framework/sofa/pull/473 )
    - FIX Remove override warnings [#520]( https://github.com/sofa-framework/sofa/pull/520 )
    - Fix memory leak while capturing screenshot [#533]( https://github.com/sofa-framework/sofa/pull/533 )
    - FIX Windows relative path from runSofa [#568]( https://github.com/sofa-framework/sofa/pull/568 )
- [SofaRigid]
    - RigidMapping: fixed setRepartition backward compatibility [#441]( https://github.com/sofa-framework/sofa/pull/441 )
- [SofaSimulationCommon]
    - Fix a minor regression introduced during the post-sprint [#476]( https://github.com/sofa-framework/sofa/pull/476 )
- [SofaSimulationCore]
    - Add stop in add_mbktomatrixVisitor [#439]( https://github.com/sofa-framework/sofa/pull/439 )

**Other modules**
- [All]
    - Fix warnings and strange double incrementation on iterator [#364]( https://github.com/sofa-framework/sofa/pull/364 )
    - installing gtest headers for separate plugin builds [#395]( https://github.com/sofa-framework/sofa/pull/395 )
    - Fix override warnings [#423]( https://github.com/sofa-framework/sofa/pull/423 )
    - FIX Sofa installation failure (tries to install non-existing files) [#470]( https://github.com/sofa-framework/sofa/pull/470 )
    - ADD _d suffix for debug libs [#511]( https://github.com/sofa-framework/sofa/pull/511 )
- [SofaBoundaryCondition]
    - Fix LinearForceField that disappears [#525]( https://github.com/sofa-framework/sofa/pull/525 )
    - FIX Removed incorrect warning from LinearForceField [#384]( https://github.com/sofa-framework/sofa/pull/384 )
- [SofaConstraint]
    - Fix error due to MacOS >= 10.11 using a relative filename [#325]( https://github.com/sofa-framework/sofa/pull/325 )
    - Fix issue in GenericConstraintCorrection  [#567]( https://github.com/sofa-framework/sofa/pull/567 )
- [SofaDeformable]
    - Fix RestShapeSpringsForceField  [#367]( https://github.com/sofa-framework/sofa/pull/367 )
- [SofaExporter]
    - FIX allow to extend VTKExporter in a plugin [#309]( https://github.com/sofa-framework/sofa/pull/309 )
- [SofaGeneralEngine]
    - Fix some XyzTransformMatrixEngine::update() function [#343]( https://github.com/sofa-framework/sofa/pull/343 )
- [SofaGeneralImplicitOdeSolver]
    - fix test [#478]( https://github.com/sofa-framework/sofa/pull/478 )
- [SofaGraphComponent]
    - Fix the test that was wrong and thus failing in SceneChecker [#405]( https://github.com/sofa-framework/sofa/pull/405 )
    - Fix a crashing bug in SceneCheckAPIChange. [#479]( https://github.com/sofa-framework/sofa/pull/479 )
    - FIX SceneChecker & RequiredPlugin [#563]( https://github.com/sofa-framework/sofa/pull/563 )
- [SofaGeneralObjectInteraction]
    - Remove stiffness multiplicator in SpringForceField [#290]( https://github.com/sofa-framework/sofa/pull/290 )
- [SofaMiscFem]
    - Fix FastTetrahedralCorotationalFF topology change [#554]( https://github.com/sofa-framework/sofa/pull/554 )
- [SofaOpenglVisual]
    - Fix a bug crashing sofa when the textures cannot be loaded. [#474]( https://github.com/sofa-framework/sofa/pull/474 )

**Extlibs**
- [CImg]
    - Refactor CImg & CImgPlugin [#562]( https://github.com/sofa-framework/sofa/pull/562 )
- [Eigen]
    - Warning fix with eigen when compiling with msvc [#447]( https://github.com/sofa-framework/sofa/pull/447 )
- [libQGLViewer]
    - FIX missing headers [#461]( https://github.com/sofa-framework/sofa/pull/461 )
    - Update libQGLViewer to 2.7.1 [#545]( https://github.com/sofa-framework/sofa/pull/545 )

**Plugins / Projects**
- [CGALPlugin]
    - Fix build problem [#351]( https://github.com/sofa-framework/sofa/pull/351 )
    - FIX build error with CGAL > 4.9.1 [#515]( https://github.com/sofa-framework/sofa/pull/515 )
- [CImgPlugin]
    - Use sofa cmake command to create proper package [#544]( https://github.com/sofa-framework/sofa/pull/544 )
    - Refactor CImg & CImgPlugin [#562]( https://github.com/sofa-framework/sofa/pull/562 )
    - prevent INT32 redefinition by libjpeg on Windows [#566]( https://github.com/sofa-framework/sofa/pull/566 )
- [ColladaSceneLoader]
    - FIX Assimp copy on Windows [#504]( https://github.com/sofa-framework/sofa/pull/504 )
- [Flexible]
    - Refactor CImg & CImgPlugin [#562]( https://github.com/sofa-framework/sofa/pull/562 )
- [image]
    - Refactor CImg & CImgPlugin [#562]( https://github.com/sofa-framework/sofa/pull/562 )
- [Meshconv]
    -  fix build if no miniflowVR [#358]( https://github.com/sofa-framework/sofa/pull/358 )
- [MultiThreading]
    - FIX: examples installation [#299]( https://github.com/sofa-framework/sofa/pull/299 )
    - Fix build with Boost 1.64.0 [#359]( https://github.com/sofa-framework/sofa/pull/359 )
    - FIX Windows dll loading [#507]( https://github.com/sofa-framework/sofa/pull/507 )
- [runSofa]
    - FIX plugin config copy on Windows [#451]( https://github.com/sofa-framework/sofa/pull/451 )
    - remove non-ASCII chars in string [#327]( https://github.com/sofa-framework/sofa/pull/327 )
    - FIX PluginRepository initialization [#502]( https://github.com/sofa-framework/sofa/pull/502 )
- [SofaCUDA]
    - Fix compilation problem with cuda.  [#320]( https://github.com/sofa-framework/sofa/pull/320 )
    - Fix: Headers of the SofaCUDA plugin are now installed in the include folder [#383]( https://github.com/sofa-framework/sofa/pull/383 )
    - FIX Configuration/compilation issue with CUDA plugin [#523]( https://github.com/sofa-framework/sofa/pull/523 )
    - Fix linearforcefield that disappears [#525]( https://github.com/sofa-framework/sofa/pull/525 )
- [SofaGui]
    - FIX draw scenes on classical and retina screens [#311]( https://github.com/sofa-framework/sofa/pull/311 )
    - Fixes #183 : Use the qt menu instead of the native one in Mac OS [#366]( https://github.com/sofa-framework/sofa/pull/366 )
    - fix ImageBMP issue + remove Laparoscopic stuff [#499]( https://github.com/sofa-framework/sofa/pull/499 )
    - Pickhandler minor fixs [#522]( https://github.com/sofa-framework/sofa/pull/522 )
    - Fix: Intel graphics on linux now overrides the core profile context [#526]( https://github.com/sofa-framework/sofa/pull/526 )
- [SofaImplicitField]
    - Fix package configuration [#422]( https://github.com/sofa-framework/sofa/pull/422 )
- [SofaPhysicsAPI]
    - FIX: compilation due to Sofa main API changes [#549]( https://github.com/sofa-framework/sofa/pull/549 )
- [SofaPython]
    - Fix python live coding that is broken [#414]( https://github.com/sofa-framework/sofa/pull/414 )
    - FIX crash in python script when visualizing advanced timer output [#458]( https://github.com/sofa-framework/sofa/pull/458 )
    - FIX error in script for plotting advancedTimer output [#519]( https://github.com/sofa-framework/sofa/pull/519 )
    - Fix python unicode data [#313]( https://github.com/sofa-framework/sofa/pull/313 )
- [SofaSPHFluid]
    - Fix invalid plugin initialization. [#467]( https://github.com/sofa-framework/sofa/pull/467 )
- [SofaVolumetricData]
    - Fix package configuration [#422]( https://github.com/sofa-framework/sofa/pull/422 )
- [SceneCreator]
    - FIX build error with AppleClang 6.0.0.6000051 [#483]( https://github.com/sofa-framework/sofa/pull/483 )
- [THMPGSpatialHashing]
    - Fix build with Boost 1.64.0 [#359]( https://github.com/sofa-framework/sofa/pull/359 )

**Scenes**
- Fix scenes [#310]( https://github.com/sofa-framework/sofa/pull/310 )
- Fix scenes with bad RegularGrid position relative to 270 [#324]( https://github.com/sofa-framework/sofa/pull/324 )
- Fix scenes errors and crashes [#505]( https://github.com/sofa-framework/sofa/pull/505 )
- FIX all scenes failures 17.12 [#565]( https://github.com/sofa-framework/sofa/pull/565 )


### Cleanings

**Kernel modules**
- [All]
    - replace a bunch of std::cerr, std::cout, prinf to use msg_* instead [#339]( https://github.com/sofa-framework/sofa/pull/339 )
    - More std::cout to msg_* cleaning [#370]( https://github.com/sofa-framework/sofa/pull/370 )
    - FIX removed compilation warnings [#386]( https://github.com/sofa-framework/sofa/pull/386 )
    - Clean BaseLoader & Remove un-needed #includes  [#393]( https://github.com/sofa-framework/sofa/pull/393 )
    - Remove last direct calls of OpenGL in modules [#425]( https://github.com/sofa-framework/sofa/pull/425 )
    - warning removal [#454]( https://github.com/sofa-framework/sofa/pull/454 )
    - Refactor SofaTest to cut dependencies [#471]( https://github.com/sofa-framework/sofa/pull/471 )
    - Modularizing config.h [#475]( https://github.com/sofa-framework/sofa/pull/475 )
    - EDIT: use PluginRepository instead of Utils::getPluginDirectory [#518]( https://github.com/sofa-framework/sofa/pull/518 )
- [SofaBaseLinearSolver]
    - Add comments in the CGLinearSolver [#469]( https://github.com/sofa-framework/sofa/pull/469 )
- [SofaBaseVisual]
    - Clean API message [#503]( https://github.com/sofa-framework/sofa/pull/503 )
- [SofaDefaultType]
    - remove warning generated by MapMapSparseMatrixEigenUtils [#485]( https://github.com/sofa-framework/sofa/pull/485 )
- [SofaHelper]
    - mute extlibs warnings [#397]( https://github.com/sofa-framework/sofa/pull/397 )
    - FileMonitor_windows compilation [#448]( https://github.com/sofa-framework/sofa/pull/448 )
    - Clean API message [#503]( https://github.com/sofa-framework/sofa/pull/503 )

**Other modules**
- [SofaGeneralEngine]
    - add test and minor cleaning for IndexValueMapper [#319]( https://github.com/sofa-framework/sofa/pull/319 )
- [SofaGeneralObjectInteraction]
    - Remove stiffness multiplicator in SpringForceField [#290]( https://github.com/sofa-framework/sofa/pull/290 )
- [SofaValidation]
    - move code to set default folder for monitor to init function [#500]( https://github.com/sofa-framework/sofa/pull/500 )

**Plugins / Projects**
- [All]
    - FIX: compilation warnings [#361]( https://github.com/sofa-framework/sofa/pull/361 )
- [CGALPlugin]
    - Fix warnings [#361]( https://github.com/sofa-framework/sofa/pull/361 )
- [image]
    - Fix warnings [#361]( https://github.com/sofa-framework/sofa/pull/361 )
- [Registration]
    - Remove deprecated scene [#331]( https://github.com/sofa-framework/sofa/pull/331 )
- [SofaPython]
    - General cleaning [#304]( https://github.com/sofa-framework/sofa/pull/304 )
    - Fix warnings [#361]( https://github.com/sofa-framework/sofa/pull/361 )
    - print cleaning + SofaPython quaternion dot product [#404]( https://github.com/sofa-framework/sofa/pull/404 )
- [runSofa]
    - Clean : remove non-ASCII chars in string [#327]( https://github.com/sofa-framework/sofa/pull/327 )


____________________________________________________________



## [v17.06]( https://github.com/sofa-framework/sofa/tree/v17.06 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v16.12...v17.06 )


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
- CImgPlugin : creation of a dedicated plugin for image loading based on CImg [#185]( https://github.com/sofa-framework/sofa/pull/185 )
- Remove deprecated miniBoost dependency [#273]( https://github.com/sofa-framework/sofa/pull/273 )


### Improvements

**Modules**
- [All]
    - update containers to support c++x11 features [#113]( https://github.com/sofa-framework/sofa/pull/113 )
    - speed up spheres rendering + code cleaning [#170]( https://github.com/sofa-framework/sofa/pull/170 )
    - updates externs/gtest to a fresh checkout [#213]( https://github.com/sofa-framework/sofa/pull/213 )
    - auto-init/cleanup libraries [#168]( https://github.com/sofa-framework/sofa/pull/168 )
    - Improve and clean msg_api and logging of message (#190, #255, #275). See [documentation](https://www.sofa-framework.org/community/doc/programming-with-sofa/logger/) for more information.
    - Add CMake option to limit cores used to build specific targets [#254]( https://github.com/sofa-framework/sofa/pull/254 )
    - Fix rgbacolor parsing [#305]( https://github.com/sofa-framework/sofa/pull/305 )
    - CMake: installing gtest headers for separate plugin builds [#395]( https://github.com/sofa-framework/sofa/pull/395 )
- [SofaKernel]
    - Update the RichConsoleStyleMessageFormatter  [#126]( https://github.com/sofa-framework/sofa/pull/126 )
    - creation of a defaulttype::RGBAColor [#119]( https://github.com/sofa-framework/sofa/pull/119 )
    - add a new method in BaseObjectDescription [#161]( https://github.com/sofa-framework/sofa/pull/161 )
    - adding listener mechanism to SceneLoader [#205]( https://github.com/sofa-framework/sofa/pull/205 )
    - common usage for DiagonalMass and tests [#230]( https://github.com/sofa-framework/sofa/pull/230 )
    - add tests for DataFileName [#250]( https://github.com/sofa-framework/sofa/pull/250 )
    - add tests for DefaultAnimationLoop [#258]( https://github.com/sofa-framework/sofa/pull/258 )
    - add tests for LocalMinDistance [#258]( https://github.com/sofa-framework/sofa/pull/258 )
    - add a way to convert message type to string in Message.cpp [#213]( https://github.com/sofa-framework/sofa/pull/213 )
    - MeshSTL.cpp replace a std:cerr by a msg_error so that FIX the corresponding failing test [#213]( https://github.com/sofa-framework/sofa/pull/213 )
    - adding listener mechanism to SceneLoader [#204]( https://github.com/sofa-framework/sofa/pull/204 )
    - Grid Topologies cleanup + new SphereGrid [#164]( https://github.com/sofa-framework/sofa/pull/164 )
    - Add CMake option SOFA_WITH_EXPERIMENTAL_FEATURES (default OFF) to enable MechanicalObject::buildIdentityBlocksInJacobian [#276]( https://github.com/sofa-framework/sofa/pull/276 )
- [SofaGraphComponents]
    - add tests for RequiredPlugin [#258]( https://github.com/sofa-framework/sofa/pull/258 )
- [SofaHelper]
    - GLSL: load shader source code from a standard string [#158]( https://github.com/sofa-framework/sofa/pull/158 )
- [SofaBaseTopology]
    - GridTopology : implement "flat" grids in 1 or 2 dimension by using setting grid resolution to "1" in the corresponding axis, and associated examples [#270]( https://github.com/sofa-framework/sofa/pull/270 )
    - add tests for RegularGridTopology [#270]( https://github.com/sofa-framework/sofa/pull/270 )
- [SofaEngine]
    - BREAKING: Add oriented box feature to BoxROI [#108]( https://github.com/sofa-framework/sofa/pull/108 )
- [SofaConstraint]
    - add instantiation of constraint corrections with Vec2f [#157]( https://github.com/sofa-framework/sofa/pull/157 )
- [SofaOpenglVisual]
    - add tests for ClipPlane [#258]( https://github.com/sofa-framework/sofa/pull/258 )
- [SofaVolumetricData]
    - add tests for DistanceGrid [#258]( https://github.com/sofa-framework/sofa/pull/258 )
    - add tests for Light [#258]( https://github.com/sofa-framework/sofa/pull/258 )
- [SofaBoundaryCondition]
    - add tests for ConstantForceField, some of them are OpenIssue demonstrating existing problem, as crashing sofa when using negative or too large values in indices  [#258]( https://github.com/sofa-framework/sofa/pull/258 )
- [CI]
    - improvement of all test scripts

**Plugins / Projects**
- [GUI]
    - mouse events are now transmitted to the scene with QtGLViewer [#132]( https://github.com/sofa-framework/sofa/pull/132 )
- [SceneCreator]
    - Cosmetic changes and remove un-needed include [#169]( https://github.com/sofa-framework/sofa/pull/169 )
- [SofaPython]
    - Macros to bind "sequence" types [#165]( https://github.com/sofa-framework/sofa/pull/165 )
    - ModuleReload [#214]( https://github.com/sofa-framework/sofa/pull/214 )
    - light module reload [#202]( https://github.com/sofa-framework/sofa/pull/202 )
    - change the way createObject() handle its arguments to simplify scene writing + batch of tests [#286]( https://github.com/sofa-framework/sofa/pull/286 )
- [SofaTest]
    - add Backtrace::autodump to all tests to ease debugging [#191]( https://github.com/sofa-framework/sofa/pull/191 )
    - add automatic tests for updateForceMask [#209]( https://github.com/sofa-framework/sofa/pull/209 )
    - add tests on PluginManager [#240]( https://github.com/sofa-framework/sofa/pull/240 )
    - TestMessageHandler : new and robust implementation to connect msg_* message to test failure  [#213]( https://github.com/sofa-framework/sofa/pull/213 )
    - update to use the new TestMessageHandler where msg_error generates test failures [#213]( https://github.com/sofa-framework/sofa/pull/213 )
    - add tests for TestMessageHandler [#213]( https://github.com/sofa-framework/sofa/pull/213 )
- [SofaCUDA]
    - FIX NVCC flags for debug build on Windows [#300]( https://github.com/sofa-framework/sofa/pull/300 )


### Bug Fixes

**Modules**
- Warnings have been fixed [#229]( https://github.com/sofa-framework/sofa/pull/229 )
- [All]
    - check that SofaPython is found before lauching the cmake sofa_set_python_directory command [#137]( https://github.com/sofa-framework/sofa/pull/137 )
    - use the cmake install DIRECTORY instead of FILES to preserve the files hierarchy when installing [#138]( https://github.com/sofa-framework/sofa/pull/138 )
    - fixing issue related to parsing attributes with atof/atoi [#161]( https://github.com/sofa-framework/sofa/pull/161 )
    - unify color datafield [#206]( https://github.com/sofa-framework/sofa/pull/206 )
    - Fix CMakeLists bug on Sofa.ini and installedSofa.ini creation [#291]( https://github.com/sofa-framework/sofa/pull/291 )
    - Fix a lot of failing tests (#271, #279)
    - Fix compilation with SOFA_FLOATING_POINT_TYPE as float [#262]( https://github.com/sofa-framework/sofa/pull/262 )
    - CMake: Fix and clean boost, when using Sofa as an external lib [#421]( https://github.com/sofa-framework/sofa/pull/421 )
- [SofaKernel]
    - Fix the Filemonitor_test random failure on MacOs [#143]( https://github.com/sofa-framework/sofa/pull/143 )
    - implement a numerical integration for triangle [#249]( https://github.com/sofa-framework/sofa/pull/249 )
    - add brace initializer to helper::vector class [#252]( https://github.com/sofa-framework/sofa/pull/252 )
    - Activates thread-safetiness on MessageDispatcher. [#257]( https://github.com/sofa-framework/sofa/pull/257 )
    - Fix getRelativePath() in DataFileName [#250]( https://github.com/sofa-framework/sofa/pull/250 )
    - FileRepository::getRelativePath() lowering the case on WIN32 is now a deprecated behavior [#264]( https://github.com/sofa-framework/sofa/pull/264 )
    - Fix FileRepository should not be optional [#122]( https://github.com/sofa-framework/sofa/pull/122 )
    - FileMonitor: fix the recurrent problem with file 'SofaKernel/framework/framework_test/resources/existing.txt' pointed in Issue #146 [#258]( https://github.com/sofa-framework/sofa/pull/258 )
    - Fix wrong inline in exported functions [#449]( https://github.com/sofa-framework/sofa/pull/449 )
- [SofaFramework]
    - fix the integration scheme for Quaternion [#172]( https://github.com/sofa-framework/sofa/pull/172 ) and fix values with which the quaternion is being compared after creation from euler angles
- [SofaHelper]
    - VisualToolGL: fix single primitive calls [#293]( https://github.com/sofa-framework/sofa/pull/293 )
    - ImagePNG: Fix library linking in debug configuration under MSVS [#298]( https://github.com/sofa-framework/sofa/pull/298 )
- [SofaBaseMechanics]
    - MechanicalObject: cleaning: symbols & include [#249]( https://github.com/sofa-framework/sofa/pull/249 )
- [SofaPhysicsAPI]
    - fix compilation of the project [#167]( https://github.com/sofa-framework/sofa/pull/167 )
- [SofaUserInteraction]
    - MouseInteractor: FIX the mouse picking on Mechanical Object [#282]( https://github.com/sofa-framework/sofa/pull/282 )

**Plugins / Projects**
- [image]
    - Fixes #135 : Check that SofaPython is found before including python directory [#137]( https://github.com/sofa-framework/sofa/pull/137 )
    - Fixes #136 : Use the cmake install DIRECTORY instead of FILES [#138]( https://github.com/sofa-framework/sofa/pull/138 )
- [LeapMotion]
    - FIX compilation for LeapMotion plugin due to moved files [#296]( https://github.com/sofa-framework/sofa/pull/296 )
- [runSofa]
    - Fix minor consistency issues related to the readOnly flag [#115]( https://github.com/sofa-framework/sofa/pull/115 )
- [SofaTest]
    - repair the minor API breaks introduced by PR #213 [#269]( https://github.com/sofa-framework/sofa/pull/269 )

**Scenes**
- Components/engine/GenerateGrid.scn was fixed [#303]( https://github.com/sofa-framework/sofa/pull/303 )


### Cleanings

**Modules**
- [All]
    - clean the consistency issues related to the readOnly flag [#115]( https://github.com/sofa-framework/sofa/pull/115 )
    - Clean licenses [#139]( https://github.com/sofa-framework/sofa/pull/139 )
- [SofaKernel]
    - clean DefaultPipeline.cpp/h (API BREAKING)
    - clean the attributes names in BoxROI (API BREAKING)
    - TetrahedronFEMForceField clean code [#270]( https://github.com/sofa-framework/sofa/pull/270 )
    - GridTopology : clean the code & factor the constructor [#270]( https://github.com/sofa-framework/sofa/pull/270 )
    - RegularGridTopology : clean the constructor's code & remove NDEBUG code [#270]( https://github.com/sofa-framework/sofa/pull/270 )
    - MechanicalObject : removal of code specific to the grid [#270]( https://github.com/sofa-framework/sofa/pull/270 )

- [SofaVolumetricData]
    - Light: clean and strenghening the interface [#258]( https://github.com/sofa-framework/sofa/pull/258 )
    - DistanceGrid
- [SofaBoundaryCondition]
    - ConstantForceField: clean to follow sofa guideline & fix the "visible dependencies" [#258]( https://github.com/sofa-framework/sofa/pull/258 )
    - ConstantForceField: replace the "points" attribute by "indices" with backward compatibility & deprecation message [#258]( https://github.com/sofa-framework/sofa/pull/258 )

**Plugins / Projects**
- [SceneCreator]
    - clean with cosmetic changes and removed un-needed includes
- [SofaPython]
    - cleaning data binding [#166]( https://github.com/sofa-framework/sofa/pull/166 )


### Moved files

- The module handling HighOrderTopologies moved into a [separate repository]( https://github.com/sofa-framework/plugin.HighOrder) [#222](https://github.com/sofa-framework/sofa/pull/222 )


____________________________________________________________



## [v16.12]( https://github.com/sofa-framework/sofa/tree/v16.12 )

**Last commit: on Jan 08, 2017**  
[Full log]( https://github.com/sofa-framework/sofa/compare/v16.08...v16.12 )

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



## [v16.08]( https://github.com/sofa-framework/sofa/tree/v16.08 )

**Last commit: on Jul 28, 2016**  
[Full log]( https://github.com/sofa-framework/sofa/compare/v15.12...v16.08 )

### New features

- SOFA on GitHub - [https://github.com/sofa-framework/sofa]( https://github.com/sofa-framework/sofa )
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



## [v15.12]( https://github.com/sofa-framework/sofa/tree/v15.12 )

[Full log]( https://github.com/sofa-framework/sofa/compare/v15.09...v15.12 )


____________________________________________________________



## [v15.09]( https://github.com/sofa-framework/sofa/tree/v15.09 )

[Full log]( https://github.com/sofa-framework/sofa/compare/release-v15.12...v15.09 )
