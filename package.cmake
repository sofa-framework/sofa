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

setSofaOption(PLUGIN_CIMGPLUGIN ON)
setSofaOption(PLUGIN_SOFAPYTHON ON)
setSofaOption(PLUGIN_SOFAMISCCOLLISION ON)
setSofaOption(PLUGIN_SOFAEULERIANFLUID OFF)
setSofaOption(PLUGIN_SOFASPHFLUID OFF)
setSofaOption(PLUGIN_SOFADISTANCEGRID OFF)
setSofaOption(PLUGIN_SOFAIMPLICITFIELD OFF)

# Copy resources files (etc/, share/, examples/) when installing 
setSofaOption(SOFA_INSTALL_RESOURCES_FILES ON)
# MacOS bundle creation
setSofaOption(RUNSOFA_INSTALL_AS_BUNDLE ON)


# Windows specific
# setSofaString(CMAKE_C_COMPILER "C:/dev/clcache/4.1.0/bin/clcache.bat")
# setSofaString(CMAKE_CXX_COMPILER "C:/dev/clcache/4.1.0/bin/clcache.bat")
# setSofaPath(CMAKE_PREFIX_PATH "$CMAKE_PREFIX_PATH;C:/dev/Qt/5.7/msvc2015_64")
# setSofaPath(BOOST_ROOT "C:/dev/boost/1.64.0")
# setSofaPath(BOOST_LIBRARYDIR "C:/dev/boost/1.64.0/lib64-msvc-14.0")
