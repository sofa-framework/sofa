######################
# Wrapper macro to set boolean value to a variable
macro(setSofaOption name value)
    set(${name} "${value}" CACHE BOOL "" FORCE)
    set(quiet "${ARGV2}")
    if(NOT quiet)
        message("${name} ${${name}}")
    endif()
endmacro()

macro(setSofaPath name value)
    set(${name} "${value}" CACHE PATH "" FORCE)
    message("${name} ${${name}}")
endmacro()

macro(setSofaString name value)
    set(${name} "${value}" CACHE STRING "" FORCE)
    message("${name} ${${name}}")
endmacro()

macro(setSofaFilePath name value)
    set(${name} "${value}" CACHE FILEPATH "" FORCE)
    message("${name} ${${name}}")
endmacro()
######################

setSofaString(CMAKE_BUILD_TYPE Release)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	setSofaString(CMAKE_CXX_FLAGS "-std=c++11")
endif()

setSofaOption(APPLICATION_RUNSOFA ON)
setSofaOption(APPLICATION_MODELER OFF)

setSofaOption(SOFA_USE_MASK OFF)

setSofaOption(SOFA_BUILD_TESTS ON)
setSofaOption(SOFA_BUILD_TUTORIALS OFF)
setSofaOption(SOFA_BUILD_METIS ON)

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    if(_variableName MATCHES "^PLUGIN_" OR _variableName MATCHES "^MODULE_")
        setSofaOption(${_variableName} OFF TRUE)
    endif()
endforeach()
message("All plugins/modules are OFF")

message("Setting some plugins/modules ON")
setSofaOption(PLUGIN_SOFAALLCOMMONCOMPONENTS ON)

setSofaOption(PLUGIN_CIMGPLUGIN ON)
setSofaOption(PLUGIN_COMPLIANT ON)
setSofaOption(PLUGIN_EXTERNALBEHAVIORMODEL ON)
setSofaOption(PLUGIN_FLEXIBLE ON) # Depends on image & CImgPlugin
setSofaOption(PLUGIN_IMAGE ON) # Depends on CImgPlugin, soft dependency to MultiThreading
setSofaOption(PLUGIN_INVERTIBLEFVM ON)
setSofaOption(PLUGIN_MANIFOLDTOPOLOGIES ON)
setSofaOption(PLUGIN_MANUALMAPPING ON)
setSofaOption(PLUGIN_MULTITHREADING ON)
setSofaOption(PLUGIN_PREASSEMBLEDMASS ON) # Depends on Flexible and Compliant
setSofaOption(PLUGIN_REGISTRATION ON)
setSofaOption(PLUGIN_RIGIDSCALE ON)
setSofaOption(PLUGIN_SOFACARVING ON)
setSofaOption(PLUGIN_SOFADISTANCEGRID ON)
setSofaOption(PLUGIN_SOFAEULERIANFLUID ON)
setSofaOption(PLUGIN_SOFAIMPLICITFIELD ON)
setSofaOption(PLUGIN_SOFAMISCCOLLISION ON)
setSofaOption(PLUGIN_SOFAPYTHON ON)
setSofaOption(PLUGIN_SOFASPHFLUID ON)
setSofaOption(PLUGIN_THMPGSPATIALHASHING ON)

#setSofaOption(PLUGIN_VOLUMETRICRENDERING ON)

setSofaOption(MODULE_SOFAEXPORTER ON)
setSofaOption(MODULE_SOFASPARSESOLVER ON)
setSofaOption(MODULE_SOFAPRECONDITIONER ON)

# Copy resources files (etc/, share/, examples/) when installing 
setSofaOption(SOFA_INSTALL_RESOURCES_FILES ON)

# install GTest even if SOFA_BUILD_TESTS=OFF
#add_subdirectory(extlibs/gtest)
