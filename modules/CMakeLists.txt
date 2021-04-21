# These are real modules. If you need components from there you
# need to first add find_package(XXX) in your CMakeLists.txt

# General
sofa_add_plugin(SofaGeneralVisual SofaGeneralVisual ON)
sofa_add_plugin(SofaGraphComponent SofaGraphComponent ON)
sofa_add_plugin(SofaGeneralMeshCollision SofaGeneralMeshCollision ON)
sofa_add_plugin(SofaBoundaryCondition SofaBoundaryCondition ON)
sofa_add_plugin(SofaGeneralAnimationLoop SofaGeneralAnimationLoop ON)
sofa_add_plugin(SofaGeneralDeformable SofaGeneralDeformable ON)
sofa_add_plugin(SofaGeneralEngine SofaGeneralEngine ON)
sofa_add_plugin(SofaGeneralExplicitOdeSolver SofaGeneralExplicitOdeSolver ON)
sofa_add_plugin(SofaGeneralImplicitOdeSolver SofaGeneralImplicitOdeSolver ON)
sofa_add_plugin(SofaGeneralLinearSolver SofaGeneralLinearSolver ON)
sofa_add_plugin(SofaGeneralRigid SofaGeneralRigid ON)
sofa_add_plugin(SofaGeneralObjectInteraction SofaGeneralObjectInteraction ON)
sofa_add_plugin(SofaGeneralSimpleFem SofaGeneralSimpleFem ON)
sofa_add_plugin(SofaGeneralTopology SofaGeneralTopology ON)
sofa_add_plugin(SofaTopologyMapping SofaTopologyMapping ON)
sofa_add_plugin(SofaUserInteraction SofaUserInteraction ON)
sofa_add_plugin(SofaConstraint SofaConstraint ON)
sofa_add_plugin(SofaGeneralLoader SofaGeneralLoader ON)
sofa_add_plugin(SofaExporter SofaExporter ON)

sofa_add_plugin(SofaSparseSolver SofaSparseSolver ON)      #
sofa_add_plugin(SofaPreconditioner SofaPreconditioner ON)  # depends on SofaSparseSolver
sofa_add_plugin(SofaHaptics SofaHaptics ON)
sofa_add_plugin(SofaValidation SofaValidation ON)

find_package(Newmat QUIET)
sofa_add_plugin(SofaDenseSolver SofaDenseSolver ON
    WHEN_TO_SHOW "Newmat_FOUND"
    VALUE_IF_HIDDEN OFF)

sofa_add_plugin(SofaNonUniformFem SofaNonUniformFem ON)

find_package(Sofa.GL QUIET)
sofa_add_plugin(SofaOpenglVisual SofaOpenglVisual ON
    WHEN_TO_SHOW "Sofa.GL_FOUND"
    VALUE_IF_HIDDEN OFF)

# Misc
sofa_add_plugin(SofaMiscTopology SofaMiscTopology ON)
sofa_add_plugin(SofaMiscExtra SofaMiscExtra ON)
sofa_add_plugin(SofaMiscForceField SofaMiscForceField ON)
sofa_add_plugin(SofaMiscEngine SofaMiscEngine ON)
sofa_add_plugin(SofaMiscSolver SofaMiscSolver ON)
sofa_add_plugin(SofaMiscFem SofaMiscFem ON)
sofa_add_plugin(SofaMiscMapping SofaMiscMapping ON)
