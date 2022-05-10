cmake_minimum_required(VERSION 3.12)

# add Config only if it was not loaded before
if(NOT TARGET Sofa.Config)
    add_subdirectory(Config)
endif()

add_subdirectory(Type)
add_subdirectory(Geometry)
add_subdirectory(Topology)
add_subdirectory(Helper)
add_subdirectory(LinearAlgebra)
add_subdirectory(DefaultType)
add_subdirectory(Core)
add_subdirectory(Simulation)
add_subdirectory(Testing)

# Meta-module gathering all framework modules
add_subdirectory(Framework)
