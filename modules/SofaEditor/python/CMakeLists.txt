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

find_package(Cython QUIET)
if(Cython_FOUND)
    message("-- PythonSofaEditor found 'cython' the .cpp will be regenerated from .pxd.")
    set(CYTHON_ACTIVATION "--use-cython")
else()
    message("-- PythonSofaEditor didn't found the 'cython' application, the binding with use default .cpp files.")
endif()

find_package(SofaPython REQUIRED)

####################################################################################################
# Files
####################################################################################################
set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/cpp_SofaEditor.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/SofaEditor.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/SofaEditor.cpp
)

set(CPP_FILES
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
#add_custom_command(COMMAND python setup.py build_ext --inplace --force -I ${INCLUDES_PATH} -L ${LIB_PATH} ${CYTHON_ACTIVATION} # Execute this command,
#                   COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/SofaEditor.timestamp
#                   DEPENDS ${SRC_FILES} SofaEditor          # The target depens on these files...
#                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                                    # In this working directory
#                   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/SofaEditor.timestamp)

set(deps "")
foreach( file_i ${CPP_FILES} )
                    message("========= ADDING: ${file_i} target: PythonLibs")
                    # Build the library out of the .cpp
                    get_filename_component(SHORT_NAME ${file_i} NAME_WE)
                    get_filename_component(DIR_PATH ${file_i} DIRECTORY)


                   add_library("Pyx${SHORT_NAME}" SHARED ${file_i})

              if(Cython_FOUND)
                   add_dependencies(Pyx${SHORT_NAME} generate-pyx-${SHORT_NAME})
                   # Build the .cpp out of the .pyx
                   add_custom_command(
                      COMMAND cython ${DIR_PATH}/${SHORT_NAME}.pyx --cplus -2 --fast-fail -I ${INCLUDES_PATH}              # Execute this command,
                      COMMAND ${CMAKE_COMMAND} -E touch "${file_i}.timestamp"
                      DEPENDS ${SRC_FILES} SofaEditor                                                  # The target depens on these files...
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                                    # In this working directory
                      OUTPUT "${file_i}.timestamp")

                  add_custom_target(
                      generate-pyx-${SHORT_NAME}
                      DEPENDS "${file_i}.timestamp"
                      )
              endif()

                   target_link_libraries("Pyx${SHORT_NAME}" SofaEditor )
                   target_include_directories("Pyx${SHORT_NAME}" PUBLIC ${PYTHON_INCLUDE_DIRS})
                   set_target_properties("Pyx${SHORT_NAME}"
                       PROPERTIES
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/SofaEditor/"
                   )

                   set_target_properties("Pyx${SHORT_NAME}" PROPERTIES PREFIX "")
                   set_target_properties("Pyx${SHORT_NAME}" PROPERTIES OUTPUT_NAME "SofaEditor")

                   set(deps ${deps} "Pyx${SHORT_NAME}")
endforeach( file_i )


add_custom_target(PythonSofaEditor
                  ALL
                  SOURCES ${SRC_FILES}
                  DEPENDS ${deps})

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
