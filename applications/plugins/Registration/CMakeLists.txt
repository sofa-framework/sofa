cmake_minimum_required(VERSION 3.1)
project(Registration VERSION 0.1)


set(HEADER_FILES
    ClosestPointRegistrationForceField.h
    ClosestPointRegistrationForceField.inl
    GroupwiseRegistrationEngine.h
    InertiaAlign.h
    RegistrationContact.h
    RegistrationContact.inl
    RegistrationContactForceField.h
    RegistrationContactForceField.inl
    RegistrationExporter.h
)

set(SOURCE_FILES
    ClosestPointRegistrationForceField.cpp
    GroupwiseRegistrationEngine.cpp
    InertiaAlign.cpp
    RegistrationContact.cpp
    RegistrationContactForceField.cpp
    RegistrationExporter.cpp
    initRegistration.cpp
)

set(PYTHON_FILES

    python/Registration/__init__.py
    python/Registration/sml.py

)

set(README_FILES README.txt)

find_package(SofaGui REQUIRED)
find_package(SofaMiscCollision REQUIRED)
find_package(SofaOpenglVisual REQUIRED)
find_package(image QUIET)

if(image_FOUND)
    list(APPEND HEADER_FILES IntensityProfileRegistrationForceField.h)
    list(APPEND HEADER_FILES IntensityProfileRegistrationForceField.inl)
    list(APPEND SOURCE_FILES IntensityProfileRegistrationForceField.cpp)
    list(APPEND HEADER_FILES IntensityProfileCreator.h)
    list(APPEND SOURCE_FILES IntensityProfileCreator.cpp)
else()
    message(STATUS "Registration: could not find image, won't build IntensityProfileRegistrationForceField")
endif()

# Config files and install rules for pythons scripts
sofa_set_python_directory(Registration "python")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaMiscCollision SofaBaseCollision SofaGuiCommon SofaBaseVisual SofaExporter SofaOpenglVisual SofaLoader SofaMiscForceField SofaGeneralEngine)

if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE image)
endif()
if(NOT ${SOFA_NO_OPENGL})
    target_link_libraries(${PROJECT_NAME} PRIVATE SofaOpenglVisual)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_REGISTRATION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(Registration_test)
endif()

find_package(SofaDistanceGrid QUIET)
if(SofaDistanceGrid_FOUND)
    add_definitions("-DSOFA_HAVE_SOFADISTANCEGRID")
    target_link_libraries(${PROJECT_NAME} PRIVATE SofaDistanceGrid)
else()
    message(STATUS "Registration: could not find SOFADISTANCEGRID, won't build RegistrationContact for RigidDistanceGridCollisionModel.")
endif()

## Install rules for the library; CMake package configurations files
sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
