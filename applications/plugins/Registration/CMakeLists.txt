cmake_minimum_required(VERSION 2.8.12)
project(Registration)

set(REGISTRATION_MAJOR_VERSION 0)
set(REGISTRATION_MINOR_VERSION 1)
set(REGISTRATION_VERSION ${REGISTRATION_MAJOR_VERSION}.${REGISTRATION_MINOR_VERSION})

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
find_package(image QUIET)

if(image_FOUND)
    list(APPEND HEADER_FILES IntensityProfileRegistrationForceField.h)
    list(APPEND HEADER_FILES IntensityProfileRegistrationForceField.inl)
    list(APPEND SOURCE_FILES IntensityProfileRegistrationForceField.cpp)
else()
    message(STATUS "Registration: could not find image, won't build IntensityProfileRegistrationForceField")
endif()

## Registration/config.h
configure_file(config.h.in Registration/config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Registration/config.h DESTINATION "include/Registration")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseCollision SofaGuiCommon SofaBaseVisual SofaExporter SofaLoader SofaOpenglVisual SofaMiscForceField)
if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} image)
endif()
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${REGISTRATION_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_REGISTRATION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(Registration_test)
endif()

install(TARGETS ${PROJECT_NAME}
        COMPONENT Registration_libraries
        EXPORT RegistrationTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}")

## CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(Registration ${REGISTRATION_VERSION} "" "include")
