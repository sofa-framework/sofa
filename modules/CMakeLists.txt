# These are real modules. If you need components from there you
# need to first add find_package(XXX) in your CMakeLists.txt

# General
sofa_add_collection(SofaGeneralVisual SofaGeneralVisual ON)
sofa_add_collection(SofaGraphComponent SofaGraphComponent ON)
sofa_add_collection(SofaGeneralMeshCollision SofaGeneralMeshCollision ON)
sofa_add_collection(SofaBoundaryCondition SofaBoundaryCondition ON)
sofa_add_collection(SofaGeneralAnimationLoop SofaGeneralAnimationLoop ON)
sofa_add_collection(SofaGeneralDeformable SofaGeneralDeformable ON)
sofa_add_collection(SofaGeneralEngine SofaGeneralEngine ON)
sofa_add_collection(SofaGeneralExplicitOdeSolver SofaGeneralExplicitOdeSolver ON)
sofa_add_collection(SofaGeneralImplicitOdeSolver SofaGeneralImplicitOdeSolver ON)
sofa_add_collection(SofaGeneralLinearSolver SofaGeneralLinearSolver ON)
sofa_add_collection(SofaGeneralRigid SofaGeneralRigid ON)
sofa_add_collection(SofaGeneralSimpleFem SofaGeneralSimpleFem ON)
sofa_add_collection(SofaGeneralTopology SofaGeneralTopology ON)
sofa_add_collection(SofaTopologyMapping SofaTopologyMapping ON)
sofa_add_collection(SofaUserInteraction SofaUserInteraction ON)
sofa_add_collection(SofaConstraint SofaConstraint ON)
sofa_add_collection(SofaGeneralLoader SofaGeneralLoader ON)
sofa_add_collection(SofaExporter SofaExporter ON)

sofa_add_collection(SofaSparseSolver SofaSparseSolver ON)      #
sofa_add_collection(SofaPreconditioner SofaPreconditioner ON)  # depends on SofaSparseSolver

sofa_add_collection(SofaDenseSolver SofaDenseSolver ON)

sofa_add_collection(SofaNonUniformFem SofaNonUniformFem ON)

find_package(Sofa.GL QUIET)
sofa_add_collection(SofaOpenglVisual SofaOpenglVisual ON
    WHEN_TO_SHOW "Sofa.GL_FOUND"
    VALUE_IF_HIDDEN OFF)

# Misc
sofa_add_collection(SofaMiscTopology SofaMiscTopology ON)
sofa_add_collection(SofaMiscExtra SofaMiscExtra ON)
sofa_add_collection(SofaMiscForceField SofaMiscForceField ON)
sofa_add_collection(SofaMiscEngine SofaMiscEngine ON)
sofa_add_collection(SofaMiscSolver SofaMiscSolver ON)
sofa_add_collection(SofaMiscFem SofaMiscFem ON)
sofa_add_collection(SofaMiscMapping SofaMiscMapping ON)
sofa_add_collection(SofaMiscCollision SofaMiscCollision ON)

sofa_add_collection(SofaHaptics SofaHaptics ON)
sofa_add_collection(SofaValidation SofaValidation ON)
sofa_add_collection(SofaGeneralObjectInteraction SofaGeneralObjectInteraction ON)
