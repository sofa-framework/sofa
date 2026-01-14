cmake_minimum_required(VERSION 3.22)
project(SofaImplicitField VERSION 1.0)

sofa_find_package(Sofa.Component.Topology.Container.Constant REQUIRED)

set(HEADER_FILES
    config.h.in
    initSofaImplicitField.h
    MarchingCube.h

    # This is backward compatibility
    deprecated/SphereSurface.h
    deprecated/ImplicitSurfaceContainer.h    # This is a backward compatibility file toward ScalarField
    deprecated/InterpolatedImplicitSurface.h # This is a backward compatibility file toward DiscreteGridField

    components/engine/FieldToSurfaceMesh.h
    components/geometry/BottleField.h
    components/geometry/DiscreteGridField.h
    components/geometry/SphericalField.h
    components/geometry/ScalarField.h
    components/geometry/StarShapedField.h
    components/mapping/ImplicitSurfaceMapping.h
    components/mapping/ImplicitSurfaceMapping.inl
)

set(SOURCE_FILES
    initSofaImplicitField.cpp
    MarchingCube.cpp

    ## This is a backward compatibility..
    deprecated/SphereSurface.cpp
    deprecated/InterpolatedImplicitSurface.cpp

    components/engine/FieldToSurfaceMesh.cpp
    components/geometry/BottleField.cpp
    components/geometry/ScalarField.cpp
    components/geometry/DiscreteGridField.cpp
    components/geometry/SphericalField.cpp
    components/geometry/StarShapedField.cpp
    components/mapping/ImplicitSurfaceMapping.cpp
)

set(EXTRA_FILES
    README.md
    )

if(SOFA_BUILD_TESTS)
    add_subdirectory(SofaImplicitField_test)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE Sofa.Component.Topology.Container.Constant)

find_package(SofaPython3 REQUIRED)
if (SofaPython3_FOUND)
  add_subdirectory(python)
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    RELOCATABLE "plugins"
    )
