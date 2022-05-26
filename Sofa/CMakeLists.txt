cmake_minimum_required(VERSION 3.12)

### Options
option(SOFA_ENABLE_LEGACY_HEADERS "Enable the compatibility layer (headers located in legacy folders)." ON)

if(SOFA_ENABLE_LEGACY_HEADERS)
    message("Using legacy headers is enabled.")
    message("Headers from v21.12 are usable, but it is advised to change your code to use the new headers introduced in v22.06.")

    add_subdirectory(framework/Compat)
    add_subdirectory(Component/Compat)
endif()

### Framework
add_subdirectory(framework)

### Component
add_subdirectory(Component)
