cmake_minimum_required(VERSION 3.12)

# framework
if(NOT TARGET Sofa.Config)
    # add Config only if it was not loaded before
    sofa_add_subdirectory(module framework/Config Sofa.Config ON)
endif()
sofa_add_subdirectory(library framework/Type Sofa.Type ON)
sofa_add_subdirectory(library framework/Geometry Sofa.Geometry ON)
sofa_add_subdirectory(library framework/Topology Sofa.Topology ON)
sofa_add_subdirectory(library framework/Helper Sofa.Helper ON)
sofa_add_subdirectory(library framework/LinearAlgebra Sofa.LinearAlgebra ON)
sofa_add_subdirectory(library framework/DefaultType Sofa.DefaultType ON)
sofa_add_subdirectory(library framework/Core Sofa.Core ON)
sofa_add_subdirectory(library framework/Simulation Sofa.Simulation ON)
sofa_add_subdirectory(library framework/Testing Sofa.Testing ON)

# Collection gathering all framework modules
sofa_add_subdirectory(collection framework/Framework Sofa.Framework ON)

# Component
sofa_add_subdirectory(module Component Sofa.Component ON)

# GL
option(SOFA_WITH_OPENGL "Compile Sofa with OpenGL support. This will define the SOFA_WITH_OPENGL and SOFA_NO_OPENGL CMake variables." ON)
sofa_add_subdirectory(library GL Sofa.GL ON WHEN_TO_SHOW "SOFA_WITH_OPENGL" VALUE_IF_HIDDEN OFF)

# GUI
sofa_add_subdirectory(library GUI Sofa.GUI ON)
