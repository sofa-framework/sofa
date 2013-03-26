cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaComponent")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/pre.cmake)

include("sofa_base_animation_loop.cmake")
include("sofa_base_linear_solver.cmake")
include("sofa_base_topology.cmake") 
include("sofa_base_mechanics.cmake")
include("sofa_base_collision.cmake")
include("sofa_base_visual.cmake")
include("sofa_sph_fluid.cmake")
include("sofa_deformable.cmake")
include("sofa_explicit_ode_solver.cmake")
include("sofa_implicit_ode_solver.cmake")
include("sofa_graph_component.cmake")
include("sofa_eulerian_fluid.cmake")
include("sofa_misc_solver.cmake")
include("sofa_object_interaction.cmake")
include("sofa_mesh_collision.cmake")
include("sofa_engine.cmake")
include("sofa_pardiso_solver.cmake") 
include("sofa_rigid.cmake")
include("sofa_eigen2_solver.cmake")         #needs: eigen-unsupported ? (what is that?)
include("sofa_boundary_condition.cmake")
include("sofa_dense_solver.cmake")
include("sofa_misc_topology.cmake")
#include("sofa_user_interaction.cmake") #needs: sofa_mesh_collision sofa_topology_mapping sofa_deformable sofa_boundary_condition
#include("sofa_constraint.cmake") #needs: sofa_mesh_collision sofa_simple_fem sofa_eigen2_solver eigen sofa_user_interaction
#include("sofa_haptics.cmake") #needs: sofa_constraint
#include("sofa_sparse_solver.cmake") #needs: metis 
#include("sofa_preconditioner.cmake") #needs: sofa_sparse_solver sofa_simple_fem
#include("sofa_volumetric_data.cmake") #needs: sofa_user_interaction sofa_mesh_collision
#include("sofa_non_uniform_fem.cmake") #needs: sofa_volumetric_data sofa_opengl_visual sofa_simple_fem
#include("sofa_misc_forcefield.cmake") #needs: sofa_misc_topology
#include("sofa_misc_fem.cmake") #needs: sofa_simple_fem
#include("sofa_misc_engine.cmake") #needs: sofa_non_uniform_fem
#include("sofa_misc_collision.cmake") #needs: sofa_constriant sofa_volumetric_data
#include("sofa_misc.cmake") #needs: sofa_misc_topology sofa_topology_mapping
#include("sofa_validation.cmake") #needs: sofa_loader

#include("sofa_opengl_visual.cmake") #link
#include("sofa_simple_fem.cmake") #needs: sofa_opengl_visual
#include("sofa_topology_mapping.cmake")
#include("sofa_component_advanced.cmake") #needs: SofaVolumetricData
#include("sofa_component_common.cmake") #needs: sofa_loader
#include("sofa_component.cmake") #needs: sofa_component_base
#include("sofa_component_general.cmake") #needs: a lot of stuff
#include("sofa_component_misc.cmake") #needs: all sofa_misc_*


include(${SOFA_CMAKE_DIR}/post.cmake)
