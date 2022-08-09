cmake_minimum_required(VERSION 3.12)

# framework libraries
if(NOT TARGET Sofa.Config)
    # add Config only if it was not loaded before
    sofa_add_subdirectory(library Config Sofa.Config ON)
endif()
sofa_add_subdirectory(library Type Sofa.Type ON)
sofa_add_subdirectory(library Geometry Sofa.Geometry ON)
sofa_add_subdirectory(library Topology Sofa.Topology ON)
sofa_add_subdirectory(library Helper Sofa.Helper ON)
sofa_add_subdirectory(library LinearAlgebra Sofa.LinearAlgebra ON)
sofa_add_subdirectory(library DefaultType Sofa.DefaultType ON)
sofa_add_subdirectory(library Core Sofa.Core ON)
sofa_add_subdirectory(library Simulation Sofa.Simulation ON)
sofa_add_subdirectory(library Testing Sofa.Testing ON)

# Library gathering all framework libraries
sofa_add_subdirectory(library Framework Sofa.Framework ON)
