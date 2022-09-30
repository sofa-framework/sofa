cmake_minimum_required(VERSION 3.12)

# framework
add_subdirectory(framework)

# Component
sofa_add_subdirectory(module Component Sofa.Component ON)

# GL
option(SOFA_WITH_OPENGL "Compile Sofa with OpenGL support. This will define the SOFA_WITH_OPENGL and SOFA_NO_OPENGL CMake variables." ON)
sofa_add_subdirectory(library GL Sofa.GL ON WHEN_TO_SHOW "SOFA_WITH_OPENGL" VALUE_IF_HIDDEN OFF)

# GUI
add_subdirectory(GUI)
