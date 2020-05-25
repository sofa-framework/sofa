cmake_minimum_required(VERSION 3.1)
project(SofaImplicitField VERSION 1.0)

sofa_find_package(SofaDistanceGrid REQUIRED)
sofa_find_package(SofaPython QUIET)

set(HEADER_FILES
    config.h.in
    initSofaImplicitField.h

    # This is backward compatibility
    deprecated/SphereSurface.h
    deprecated/ImplicitSurfaceContainer.h    # This is a backward compatibility file toward ScalarField
    deprecated/InterpolatedImplicitSurface.h # This is a backward compatibility file toward DiscreteridField

    components/geometry/DiscreteGridField.h
    components/geometry/SphericalField.h
    components/geometry/ScalarField.h
    components/mapping/ImplicitSurfaceMapping.h
    components/mapping/ImplicitSurfaceMapping.inl
)

set(SOURCE_FILES
    initSofaImplicitField.cpp

    ## This is a backward compatibility..
    deprecated/SphereSurface.cpp
    deprecated/InterpolatedImplicitSurface.cpp

    components/geometry/ScalarField.cpp
    components/geometry/DiscreteGridField.cpp
    components/geometry/SphericalField.cpp
    components/mapping/ImplicitSurfaceMapping.cpp
)

set(EXTRA_FILES
    README.md
    examples/README.md
    )

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(SofaImplicitField_test)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_compile_options(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SOFAIMPLICITFIELD")
target_link_libraries(${PROJECT_NAME} PRIVATE SofaDistanceGrid)


## Install rules for the library and headers; CMake package configurations files
sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
