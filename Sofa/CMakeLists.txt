cmake_minimum_required(VERSION 3.12)

### Options
option(SOFA_ENABLE_LEGACY_HEADERS "Enable the compatibility layer (headers located in legacy folders)." ON)

### Framework

# add Config only if it was not loaded before
if(NOT TARGET Sofa.Config)
	add_subdirectory(Config)
endif()

if(SOFA_ENABLE_LEGACY_HEADERS)
    message("Using legacy headers is enabled.")
    message("Headers from v21.12 are usable, but it is advised to change your code to use the new headers introduced in v22.06.")

    add_subdirectory(Compat)
    add_subdirectory(Component/Compat)
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

### Component
add_subdirectory(Component)
