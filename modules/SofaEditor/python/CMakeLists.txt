cmake_minimum_required(VERSION 3.1)
project(PySofaEditor)

####################################################################################################
# Dependencies
####################################################################################################
find_package(SofaFramework REQUIRED)
find_package(PythonLibs 2.7 QUIET)
if(!PythonLibs_FOUND)
    message("-- PythonSofaEditor is disabled because the PythonLibs 2.7 cmake package is missing.")
    return()
endif()

find_package(SofaEditor QUIET)
if(!SofaEditor_FOUND)
    message("-- PythonSofaEditor is disabled because the 'SofaEditor' module is missing or not activated.")
    return()
endif()

find_package(Cython REQUIRED)
if(Cython_FOUND)
    message("-- PythonSofaEditor found 'cython' the .cpp will be regenerated from .pxd.")
    set(CYTHON_ACTIVATION "--use-cython")
else()
    message("-- PythonSofaEditor didn't found the 'cython' application, the binding with use default .cpp files.")
endif()

####################################################################################################
# Files
####################################################################################################
set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/cpp_SofaEditor.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/SofaEditor.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/SofaEditor.cpp
)

####################################################################################################
# Targets & Builds
####################################################################################################
# Retrieve the include directories needed by the SofaEditor target and replace the separated with a ':'
set(INCLUDES_PATH "$<JOIN:$<TARGET_PROPERTY:SofaEditor,INTERFACE_INCLUDE_DIRECTORIES>,:>")
set(LIB_PATH "$<TARGET_FILE_DIR:SofaEditor>")

### Defines a new custom command that compile the binding.
### The pyx are first converted into .cpp. Those cpp are then compiled into pyton modules.
add_custom_command(COMMAND python setup.py build_ext --inplace --force -I ${INCLUDES_PATH} -L ${LIB_PATH} ${CYTHON_ACTIVATION}  # Execute this command,
                   COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/SofaEditor.timestamp
                   DEPENDS ${SRC_FILES} SofaEditor          # The target depens on these files...
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                                    # In this working directory
                   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/SofaEditor.timestamp)

add_custom_target(PythonSofaEditor
                  ALL
                  SOURCES ${SRC_FILES}
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/SofaEditor.timestamp)

## Python configuration file (build tree), the lib in the source dir (easier while developping .py files)
file(WRITE "${CMAKE_BINARY_DIR}/etc/sofa/python.d/SofaEditor" "${CMAKE_CURRENT_SOURCE_DIR}")

## Python configuration file (install tree)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaEditor-config"
     "modules/SofaEditor/python/")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaEditor-config"
         DESTINATION "etc/sofa/python.d"
         RENAME "SofaEditor")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor DESTINATION modules/SofaEditor/python)

####################################################################################################
# Tests
####################################################################################################
add_subdirectory(tests)
