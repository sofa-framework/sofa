include(CMakePackageConfigHelpers)

# sofa_write_package_config_files(Foo <version> <build-include-dirs> <install-include-dirs>)
#
# Create CMake package configuration files
# - In the build tree:
#   - ${CMAKE_CURRENT_BINARY_DIR}/FooConfig.cmake
# - In the install tree:
#   - lib/cmake/Foo/FooConfigVersion.cmake
#   - lib/cmake/Foo/FooConfig.cmake
#   - lib/cmake/Foo/FooTargets.cmake
#
# This macro factorizes boilerplate CMake code for the different
# packages in Sofa.  Is assumes the following conventions hold:
# - There is a FooConfig.cmake.in file template in the same directory.
#   This file can refer to @PACKAGE_FOO_INCLUDE_DIR@, which will
#   expand to the include directories for this package, either in the
#   build tree or in the install tree. See example below.


# Assuming Foo depends on Bar and Baz, and creates the targets Foo and
# Qux, here is a typical FooConfig.cmake.in:
#
# @PACKAGE_INIT@
#
# find_package(Bar REQUIRED)
# find_package(Baz REQUIRED)
#
# if(NOT TARGET Qux)
# 	include("${CMAKE_CURRENT_LIST_DIR}/SofaBaseTargets.cmake")
# endif()
#
# check_required_components(Foo Qux)
#
# set(Foo_LIBRARIES Foo Qux)
# set(Foo_INCLUDE_DIRS @PACKAGE_FOO_INCLUDE_DIR@ ${Bar_INCLUDE_DIRS} ${Baz_INCLUDE_DIRS})


macro(sofa_write_package_config_files package_name version build_include_dirs install_include_dirs)

    string(TOUPPER ${package_name} uppercase_package_name)

    ## <package_name>ConfigVersion.cmake
    set(filename ${package_name}ConfigVersion.cmake)
    write_basic_package_version_file(${filename} VERSION ${version} COMPATIBILITY ExactVersion)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${filename}" DESTINATION lib/cmake/${package_name})


    ### <package_name>Config.cmake

    set(input_file ${package_name}Config.cmake.in)
    set(output_file ${package_name}Config.cmake)
    ## Build tree
    set(${uppercase_package_name}_INCLUDE_DIR ${build_include_dirs} "${CMAKE_CURRENT_BINARY_DIR}")
    configure_package_config_file(${input_file}
                                  ${output_file}
                                  INSTALL_DESTINATION none
                                  PATH_VARS ${uppercase_package_name}_INCLUDE_DIR)
    ## Install tree
    set(${uppercase_package_name}_INCLUDE_DIR ${install_include_dirs})
    configure_package_config_file(${input_file}
                                  Installed${output_file}
                                  INSTALL_DESTINATION lib/cmake/${package_name}
                                  PATH_VARS ${uppercase_package_name}_INCLUDE_DIR)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Installed${output_file}
            DESTINATION lib/cmake/${package_name}
            RENAME ${output_file})


    ## <package_name>Targets.cmake
    install(EXPORT ${package_name}Targets DESTINATION lib/cmake/${package_name})

endmacro()
