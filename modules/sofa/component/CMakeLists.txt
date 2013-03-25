cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaComponent")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/pre.cmake)

include("sofa_base_animation_loop.cmake")
include("sofa_base_linear_solver.cmake")
include("sofa_base_topology.cmake")
include("sofa_base_mechanics.cmake")
#include("sofa_base_collision.cmake")
include("sofa_base_visual.cmake")

include("sofa_sph_fluid.cmake")
#include("sofa_deformable.cmake")
include("sofa_explicit_ode_solver.cmake")
include("sofa_implicit_ode_solver.cmake")
include("sofa_graph_component.cmake")

include(${SOFA_CMAKE_DIR}/post.cmake)
