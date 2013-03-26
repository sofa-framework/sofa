cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaComponent")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/pre.cmake)

include("SofaBaseAnimationLoop.cmake")
include("SofaBaseLinearSolver.cmake")
include("SofaBaseTopology.cmake") 
include("SofaBaseMechanics.cmake")
include("SofaBaseCollision.cmake")
include("SofaBaseVisual.cmake")
include("SofaSphFluid.cmake")
include("SofaDeformable.cmake")
include("SofaExplicitOdeSolver.cmake")
include("SofaImplicitOdeSolver.cmake")
include("SofaGraphComponent.cmake")
include("SofaEulerianFluid.cmake")
include("SofaMiscSolver.cmake")
include("SofaObjectInteraction.cmake")
include("SofaMeshCollision.cmake")
include("SofaEngine.cmake")
include("SofaPardisoSolver.cmake") 
include("SofaRigid.cmake")
include("SofaEigen2Solver.cmake")         #needs: eigen-unsupported ? (what is that?)
include("SofaBoundaryCondition.cmake")
include("SofaDenseSolver.cmake")
include("SofaMiscTopology.cmake")
#include("SofaUserInteraction.cmake") #needs: sofa_mesh_collision sofa_topology_mapping sofa_deformable sofa_boundary_condition
#include("SofaConstraint.cmake") #needs: sofa_mesh_collision sofa_simple_fem sofa_eigen2_solver eigen sofa_user_interaction
#include("SofaHaptics.cmake") #needs: sofa_constraint
#include("SofaSparseSolver.cmake") #needs: metis 
#include("SofaPreconditioner.cmake") #needs: sofa_sparse_solver sofa_simple_fem
#include("SofaVolumetricData.cmake") #needs: sofa_user_interaction sofa_mesh_collision
#include("SofaNonUniformFem.cmake") #needs: sofa_volumetric_data sofa_opengl_visual sofa_simple_fem
#include("SofaMiscForcefield.cmake") #needs: sofa_misc_topology
#include("SofaMiscFem.cmake") #needs: sofa_simple_fem
#include("SofaMiscEngine.cmake") #needs: sofa_non_uniform_fem
#include("SofaMiscCollision.cmake") #needs: sofa_constriant sofa_volumetric_data
#include("SofaMisc.cmake") #needs: sofa_misc_topology sofa_topology_mapping
#include("SofaValidation.cmake") #needs: sofa_loader

#include("SofaOpenglVisual.cmake") #link
#include("SofaSimpleFem.cmake") #needs: sofa_opengl_visual
#include("SofaTopologyMapping.cmake")
#include("SofaComponentAdvanced.cmake") #needs: SofaVolumetricData
#include("SofaComponentCommon.cmake") #needs: sofa_loader
#include("SofaComponent.cmake") #needs: sofa_component_base
#include("SofaComponentGeneral.cmake") #needs: a lot of stuff
#include("SofaComponentMisc.cmake") #needs: all sofa_misc_*


include(${SOFA_CMAKE_DIR}/post.cmake)
