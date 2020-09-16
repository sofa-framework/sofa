include_guard(GLOBAL)
include(CMakePackageConfigHelpers)
include(CMakeParseLibraryList)


# Declare a (unique, TODO?) directory containing the python scripts of
# a plugin.  This macro:
# - creates rules to install all the .py scripts in ${directory} to
#   lib/python2.7/site-packages/${plugin_name}
# - creates a etc/sofa/python.d/${plugin_name} file in the build tree
#   pointing to the source tree
# - creates a etc/sofa/python.d/${plugin_name} file in the install
#   tree, containing a relative path to the installed script directory
#
# Assumes relative path.
macro(sofa_set_python_directory plugin_name directory)
    message(WARNING "sofa_set_python_directory is deprecated. Use sofa_install_pythonscripts instead.")
    sofa_install_pythonscripts(PLUGIN_NAME "${plugin_name}" PYTHONSCRIPTS_SOURCE_DIR "${directory}")
endmacro()

macro(sofa_install_pythonscripts)
    set(oneValueArgs PLUGIN_NAME PYTHONSCRIPTS_SOURCE_DIR PYTHONSCRIPTS_INSTALL_DIR)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments("ARG" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Required arguments
    foreach(arg ARG_PLUGIN_NAME ARG_PYTHONSCRIPTS_SOURCE_DIR)
        if("${${arg}}" STREQUAL "")
            string(SUBSTRING "${arg}" 4 -1 arg_name) # arg name without "ARG_"
            message(SEND_ERROR "Missing parameter ${arg_name}.")
        endif()
    endforeach()

    set(include_install_dir "lib/python2.7/site-packages")
    if(ARG_PYTHONSCRIPTS_INSTALL_DIR)
        set(include_install_dir "${ARG_PYTHONSCRIPTS_INSTALL_DIR}")
    endif()

    ## Install python scripts, preserving the file tree
    file(GLOB_RECURSE ALL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_PYTHONSCRIPTS_SOURCE_DIR}/*")
    file(GLOB_RECURSE PYC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_PYTHONSCRIPTS_SOURCE_DIR}/*.pyc")
    if(PYC_FILES)
        list(REMOVE_ITEM ALL_FILES ${PYC_FILES})
    endif()
    foreach(python_file ${ALL_FILES})
        file(RELATIVE_PATH script "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_PYTHONSCRIPTS_SOURCE_DIR}" "${python_file}")
        get_filename_component(path ${script} DIRECTORY)
        install(FILES ${ARG_PYTHONSCRIPTS_SOURCE_DIR}/${script}
                DESTINATION "${include_install_dir}/${path}"
                COMPONENT applications)
    endforeach()

    ## Python configuration file (build tree)
    file(WRITE "${CMAKE_BINARY_DIR}/etc/sofa/python.d/${ARG_PLUGIN_NAME}"
         "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_PYTHONSCRIPTS_SOURCE_DIR}")
    ## Python configuration file (install tree)
     file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaPython-config"
         "${include_install_dir}")
     install(FILES "${CMAKE_CURRENT_BINARY_DIR}/installed-SofaPython-config"
             DESTINATION "etc/sofa/python.d"
             RENAME "${ARG_PLUGIN_NAME}"
             COMPONENT applications)
endmacro()


# - Create a target for a python binding module relying on pybind11
#
# sofa_add_pybind11_module(TARGET OUTPUT SOURCES DEPENDS CYTHONIZE)
#  TARGET             - (input) the name of the generated target.
#  OUTPUT             - (input) the output location.
#  SOURCES            - (input) list of input files. It can be .cpp, .h ...
#  DEPENDS            - (input) set of target the generated target will depends on.
#  NAME               - (input) The actual name of the generated .so file
#                       (most commonly equals to TARGET, without the "python" prefix)
#
# The typical usage scenario is to build a python module out of cython binding.
#
# Example:
# find_package(pybind11)
# set(SOURCES_FILES
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/initbindings.cpp
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/binding1.cpp
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/binding2.cpp
#       [...]
#    )
# sofa_add_pybind11_module( TARGET MyModule SOURCES ${SOURCE_FILES}
#                           DEPENDS Deps1 Deps2  OUTPUT ${CMAKE_CURRENT_BIN_DIR}
#                           NAME python_module_name)
function(sofa_add_pybind11_module)
    set(options)
    set(oneValueArgs TARGET OUTPUT NAME)
    set(multiValueArgs SOURCES DEPENDS)
    cmake_parse_arguments("" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    set(PYBIND11_CPP_STANDARD -std=c++11)
    pybind11_add_module(${_TARGET} SHARED ${_SOURCES} NO_EXTRAS)
    target_link_libraries(${_TARGET} PRIVATE ${_DEPENDS} ${PYTHON_LIBRARIES} pybind11::module)
    set_target_properties(${_TARGET} PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${_OUTPUT}
      LIBRARY_OUTPUT_DIRECTORY ${_OUTPUT}
      RUNTIME_OUTPUT_DIRECTORY ${_OUTPUT}
      OUTPUT_NAME ${_NAME})
endfunction()


# - Create a target for a mixed python module composed of .py and binding code (in .cpp or .pyx)
#
# sofa_add_python_module(TARGET OUTPUT SOURCES DEPENDS CYTHONIZE)
#  TARGET             - (input) the name of the generated target.
#  OUTPUT             - (input) the output location, if not provided ${CMAKE_CURRENT_SOURCE_DIR} will be used. 
#  SOURCES            - (input) list of input files. It can be .py, .pyx, .pxd, .cpp
#                               .cpp are compiled, .pyx can generate .cpp if CYTHONIZE param is set to true
#  DEPENDS            - (input) set of target the generated target will depends on.
#  CYTHONIZE          - (input) boolean indicating wether or not
#                               we need to call cython on the .pyx file to re-generate the .cpp file.
#
# The typical usage scenario is to build a python module out of cython binding.
#
# Example:
# find_package(Cython QUIET)
# set(SOURCES_FILES
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/__init__.py
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/purepython.py
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/binding_withCython.pyx
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/binding_withCython.pxd
#       ${CMAKE_CURRENT_SOURCE_DIR}/ModuleDir/binding_withCPython.cpp
#    )
# sofa_add_python_module( TARGET MyModule SOURCES ${SOURCE_FILES} DEPENDS Deps1 Deps2 CYTHONIZE True OUTPUT ${CMAKE_CURRENT_BIN_DIR})
function(sofa_add_python_module)
    set(options)
    set(oneValueArgs TARGET OUTPUT CYTHONIZE DIRECTORY)
    set(multiValueArgs SOURCES DEPENDS)
    cmake_parse_arguments("" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(INCLUDE_DIRS)
    set(LIB_DIRS)

    add_custom_target(${_TARGET}
                       ALL
                       SOURCES ${_SOURCES}
                       DEPENDS ${_DEPENDS})

    if(NOT PYTHON_BINDING_VERSION)
        set(PYTHON_BINDING_VERSION 3)
    endif()

    set(_DIRECTORY ${_OUTPUT})

    foreach( source ${_SOURCES} )
        unset(cppfile)
        get_filename_component(pathdir ${source} DIRECTORY)
        get_filename_component(filename ${source} NAME_WE)
        get_filename_component(ext ${source} EXT)

        if((${ext} STREQUAL ".cpp"))
            set(cppfile "${pathdir}/${filename}.cpp")
        endif()

        if(_CYTHONIZE AND (${ext} STREQUAL ".pyx"))
            set(pyxfile "${pathdir}/${filename}.pyx")
            set(cppfile "${pathdir}/${filename}.cpp")

            # Build the .cpp out of the .pyx
            add_custom_command(
                COMMAND cython ${pathdir}/${filename}${ext} --cplus -${PYTHON_BINDING_VERSION} --fast-fail --force # Execute this command,
                DEPENDS ${_SOURCES} ${_DEPENDS}                                                     # The target depens on these files...
                WORKING_DIRECTORY ${_DIRECTORY}                                   # In this working directory
                OUTPUT ${cppfile}
            )

            message("-- ${_TARGET} cython generated '${cppfile}' from '${filename}${ext}'" )
        endif()

        if(cppfile)
            set(pyxtarget "${_TARGET}_${filename}")
            add_library(${pyxtarget} SHARED ${cppfile})

            # The implementation of Python deliberately breaks strict-aliasing rules, so we
            # compile with -fno-strict-aliasing to prevent the compiler from relying on
            # those rules to optimize the code.
            if(${CMAKE_COMPILER_IS_GNUCC})
                set(SOFACYTHON_COMPILER_FLAGS "-fno-strict-aliasing")
            endif()

            target_link_libraries(${pyxtarget} ${_DEPENDS} ${PYTHON_LIBRARIES})
            target_include_directories(${pyxtarget} PRIVATE ${PYTHON_INCLUDE_DIRS})
            target_compile_options(${pyxtarget} PRIVATE ${SOFACYTHON_COMPILER_FLAGS})
            set_target_properties(${pyxtarget}
                PROPERTIES
                ARCHIVE_OUTPUT_DIRECTORY "${_OUTPUT}"
                LIBRARY_OUTPUT_DIRECTORY "${_OUTPUT}"
                RUNTIME_OUTPUT_DIRECTORY "${_OUTPUT}"
                )

            set_target_properties(${pyxtarget} PROPERTIES PREFIX "")
            set_target_properties(${pyxtarget} PROPERTIES OUTPUT_NAME "${filename}")

            add_dependencies(${_TARGET} ${pyxtarget})
        endif()
    endforeach()
endfunction()
