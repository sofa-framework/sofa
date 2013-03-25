cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaComponent")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/pre.cmake)

include("sofa_base_animation_loop.cmake")
include("sofa_base_linear_solver.cmake")
#include("sofa_base_topology.cmake") #needs: eigen
#include("sofa_base_mechanics.cmake") #needs: sofa_base_topology
#include("sofa_base_collision.cmake") #needs: sofa_base_mechanics sofa_rigid sofa_sph_fluid
#include("sofa_base_visual.cmake") #needs: sofa_base_topology

#include("sofa_sph_fluid.cmake") #needs: sofa_base_mechanics
#include("sofa_deformable.cmake") #needs: sofa_base_topology
include("sofa_explicit_ode_solver.cmake")
include("sofa_implicit_ode_solver.cmake")
include("sofa_graph_component.cmake")
#include("sofa_topology_mapping.cmake") #needs: sofa_base_topology
include("sofa_eulerian_fluid.cmake")
include("sofa_misc_solver.cmake")
#include("sofa_opengl_visual.cmake") #needs: sofa_base_visual
#include("sofa_simple_fem.cmake") #needs: sofa_opengl_visual
#include("sofa_object_interaction.cmake")  #needs: sofa_deformable
#include("sofa_mesh_collision.cmake") #needs: sofa_object_interaction sofa_base_collision
#include("sofa_engine.cmake") #needs: sofa_mesh_collision
#include("sofa_paradiso_solver.cmake") 
#include("sofa_rigid.cmake")
#include("sofa_boundary_condition.cmake") 
#include("sofa_component_advanced.cmake")
#include("sofa_component_common.cmake")
#include("sofa_component.cmake")


include(${SOFA_CMAKE_DIR}/post.cmake)
