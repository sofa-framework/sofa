include(CMakePackageConfigHelpers)

# sofa_write_package_config_files(Foo 1.0)
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
# - There is a FooConfig.cmake.in file template in the same directory
macro(sofa_write_package_config_files package_name version)

    string(TOUPPER ${package_name} uppercase_package_name)

    ## <package_name>ConfigVersion.cmake
    set(filename ${package_name}ConfigVersion.cmake)
    write_basic_package_version_file(${filename} VERSION ${version} COMPATIBILITY ExactVersion)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${filename}" DESTINATION lib/cmake/${package_name})


    ### <package_name>Config.cmake

    set(input_file ${package_name}Config.cmake.in)
    set(output_file ${package_name}Config.cmake)
    ## Build tree
    set(${uppercase_package_name}_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    configure_package_config_file(${input_file}
                                  ${output_file}
                                  INSTALL_DESTINATION none
                                  PATH_VARS ${uppercase_package_name}_INCLUDE_DIR)
    ## Install tree
    set(${uppercase_package_name}_INCLUDE_DIR include)
    configure_package_config_file(${input_file}
                                  Installed${output_file}
                                  INSTALL_DESTINATION lib/cmake/${package_name}
                                  PATH_VARS ${uppercase_package_name}_INCLUDE_DIR)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Installed${output_file}
            DESTINATION lib/cmake/${package_name}
            RENAME ${output_file})


    ## <package_name>Targets.cmake
    install(EXPORT ${package_name}Targets
            DESTINATION lib/cmake/${package_name})

endmacro()
