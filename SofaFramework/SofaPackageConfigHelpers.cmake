include(CMakePackageConfigHelpers)

# sofa_write_package_config_files(Foo <version> <build-include-dirs>)
#
# Create CMake package configuration files
# - In the build tree:
#   - ${CMAKE_BINARY_DIR}/cmake/FooConfig.cmake
#   - ${CMAKE_BINARY_DIR}/cmake/FooConfigVersion.cmake
# - In the install tree:
#   - lib/cmake/Foo/FooConfigVersion.cmake
#   - lib/cmake/Foo/FooConfig.cmake
#   - lib/cmake/Foo/FooTargets.cmake
#
# This macro factorizes boilerplate CMake code for the different
# packages in Sofa.  It assumes that there is a FooConfig.cmake.in
# file template in the same directory.  For example, if a package Foo
# depends on Bar and Baz, and creates the targets Foo and Qux, here is
# a typical FooConfig.cmake.in:
#
# @PACKAGE_INIT@
#
# find_package(Bar REQUIRED)
# find_package(Baz REQUIRED)
#
# if(NOT TARGET Qux)
# 	include("${CMAKE_CURRENT_LIST_DIR}/FooTargets.cmake")
# endif()
#
# check_required_components(Foo Qux)

macro(sofa_install_targets package_name the_targets install_include_subdir)

    install(TARGETS ${the_targets}
            EXPORT ${package_name}Targets
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            PUBLIC_HEADER DESTINATION include/${install_include_subdir})

endmacro()

macro(sofa_write_package_config_files package_name version)

    ## <package_name>Targets.cmake
    install(EXPORT ${package_name}Targets DESTINATION lib/cmake/${package_name})

    ## <package_name>ConfigVersion.cmake
    set(filename ${package_name}ConfigVersion.cmake)
    write_basic_package_version_file(${filename} VERSION ${version} COMPATIBILITY ExactVersion)
    configure_file("${CMAKE_CURRENT_BINARY_DIR}/${filename}"
                   "${CMAKE_BINARY_DIR}/cmake/${filename}" COPYONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${filename}" DESTINATION lib/cmake/${package_name})

    ### <package_name>Config.cmake
    configure_package_config_file(${package_name}Config.cmake.in
                                  "${CMAKE_BINARY_DIR}/cmake/${package_name}Config.cmake"
                                  INSTALL_DESTINATION lib/cmake/${package_name})
    install(FILES "${CMAKE_BINARY_DIR}/cmake/${package_name}Config.cmake"
            DESTINATION lib/cmake/${package_name})

endmacro()

macro(sofa_create_package package_name version the_targets include_subdir)
    sofa_install_targets("${package_name}" "${the_targets}" "${include_subdir}")
    sofa_write_package_config_files("${package_name}" "${version}")
endmacro()
