cmake_minimum_required(VERSION 2.8.12)
project(Registration)

set(REGISTRATION_MAJOR_VERSION 0)
set(REGISTRATION_MINOR_VERSION 0)
set(REGISTRATION_BUILD_VERSION 1)

set(REGISTRATION_VERSION ${REGISTRATION_MAJOR_VERSION}.${REGISTRATION_MINOR_VERSION}.${REGISTRATION_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")


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
    config.h
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
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.h")
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.inl")
    list(APPEND SOURCE_FILES "IntensityProfileRegistrationForceField.cpp")
else()
    message(STATUS "Registration: could not find image")
endif()


set(REGISTRATION_DEPENDENCY_INCLUDE_DIR ${SofaBase_INCLUDE_DIRS} ${Newmat_INCLUDE_DIRS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Registration.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/Registration/Registration.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Registration/Registration.h
        DESTINATION "include/Registration")


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseCollision SofaGuiCommon SofaBaseVisual SofaExporter SofaLoader SofaOpenglVisual SofaMiscForceField)
if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} image)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${REGISTRATION_DEPENDENCY_INCLUDE_DIR}>")
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


include(CMakePackageConfigHelpers)
# RegistrationConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/RegistrationConfigVersion.cmake
                                 VERSION ${REGISTRATION_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/RegistrationConfigVersion.cmake
        DESTINATION lib/cmake/Registration)


# RegistrationConfig.cmake (build tree)
set(REGISTRATION_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(RegistrationConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/RegistrationConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS REGISTRATION_INCLUDE_DIR)

# RegistrationConfig.cmake (installed)
set(REGISTRATION_INCLUDE_DIR include)
configure_package_config_file(RegistrationConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/RegistrationConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/Registration
                              PATH_VARS REGISTRATION_INCLUDE_DIR)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/RegistrationConfigInstall.cmake
        DESTINATION lib/cmake/Registration
        RENAME RegistrationConfig.cmake)

# RegistrationTargets.cmake (installed)
install(EXPORT RegistrationTargets
        DESTINATION lib/cmake/Registration)

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
