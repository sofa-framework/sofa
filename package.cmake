######################
# Wrapper macro to set boolean value to a variable
macro(setSofaOption name value)
    set(${name} "${value}" CACHE BOOL "" FORCE)
    message("${name} ${value}")
endmacro()

macro(setSofaPath name value)
    set(${name} "${value}" CACHE PATH "" FORCE)
    message("${name} ${value}")
endmacro()

macro(setSofaString name value)
    set(${name} "${value}" CACHE STRING "" FORCE)
    message("${name} ${value}")
endmacro()

macro(setSofaFilePath name value)
    set(${name} "${value}" CACHE FILEPATH "" FORCE)
    message("${name} ${value}")
endmacro()
######################

setSofaString(CMAKE_BUILD_TYPE Release)

setSofaOption(APPLICATION_RUNSOFA ON)
setSofaOption(APPLICATION_MODELER OFF)

setSofaOption(SOFA_USE_MASK OFF)

setSofaOption(SOFA_BUILD_TESTS OFF)
setSofaOption(SOFA_BUILD_TUTORIALS OFF)

setSofaOption(PLUGIN_SOFAALLCOMMONCOMPONENTS ON)
setSofaOption(PLUGIN_CIMGPLUGIN ON)
setSofaOption(PLUGIN_SOFAPYTHON ON)
setSofaOption(PLUGIN_SOFAMISCCOLLISION ON)
setSofaOption(PLUGIN_SOFAEULERIANFLUID OFF)
setSofaOption(PLUGIN_SOFASPHFLUID OFF)
setSofaOption(PLUGIN_SOFADISTANCEGRID OFF)
setSofaOption(PLUGIN_SOFAIMPLICITFIELD OFF)

setSofaOption(PLUGIN_PSL OFF)

# Copy resources files (etc/, share/, examples/) when installing 
setSofaOption(SOFA_INSTALL_RESOURCES_FILES ON)
# MacOS bundle creation
setSofaOption(RUNSOFA_INSTALL_AS_BUNDLE ON)
