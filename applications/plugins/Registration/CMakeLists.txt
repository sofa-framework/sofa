cmake_minimum_required(VERSION 2.8.12)
project(Registration)

set(REGISTRATION_MAJOR_VERSION 0)
set(REGISTRATION_MINOR_VERSION 0)
set(REGISTRATION_BUILD_VERSION 1)

set(REGISTRATION_VERSION ${REGISTRATION_MAJOR_VERSION}.${REGISTRATION_MINOR_VERSION}.${REGISTRATION_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(REGISTRATION_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(REGISTRATION_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(REGISTRATION_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(REGISTRATION_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/Registration)

set(REGISTRATION_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(REGISTRATION_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(REGISTRATION_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(REGISTRATION_INSTALL_CMAKE_DIR   ${REGISTRATION_INSTALL_LIB_DIR}/cmake/Registration )

set(HEADER_FILES
    initRegistration.h
    RegistrationContact.h
    RegistrationContact.inl
    RegistrationContactForceField.h
    RegistrationContactForceField.inl
    ClosestPointRegistrationForceField.h
    ClosestPointRegistrationForceField.inl
    RegistrationExporter.h
    GroupwiseRegistrationEngine.h
    InertiaAlign.h)

set(SOURCE_FILES
    initRegistration.cpp
    RegistrationContact.cpp
    RegistrationContactForceField.cpp
    ClosestPointRegistrationForceField.cpp
    RegistrationExporter.cpp
    GroupwiseRegistrationEngine.cpp
    InertiaAlign.cpp)

set(README_FILES README.txt)


find_package(SofaGui REQUIRED)
find_package(image)

if(image_FOUND)
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.h")
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.inl")
    list(APPEND SOURCE_FILES "IntensityProfileRegistrationForceField.cpp")
endif()


set(REGISTRATION_DEPENDENCY_INCLUDE_DIR ${SofaBase_INCLUDE_DIRS} ${Newmat_INCLUDE_DIRS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Registration.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/Registration/Registration.h)
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/Registration/Registration.h
         DESTINATION "${REGISTRATION_INSTALL_INCLUDE_DIR}/Registration")


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME} SofaGuiQt)
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
        EXPORT  RegistrationTargets
        RUNTIME DESTINATION "${REGISTRATION_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${REGISTRATION_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${REGISTRATION_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${REGISTRATION_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}")


include(CMakePackageConfigHelpers)
# RegistrationConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${REGISTRATION_BUILD_CMAKE_DIR}/RegistrationConfigVersion.cmake
                                 VERSION ${REGISTRATION_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${REGISTRATION_BUILD_CMAKE_DIR}/RegistrationConfigVersion.cmake
        DESTINATION ${REGISTRATION_INSTALL_CMAKE_DIR})


# RegistrationConfig.cmake (build tree)
set(REGISTRATION_INCLUDE_DIR "${REGISTRATION_BUILD_INCLUDE_DIR}")
configure_package_config_file(RegistrationConfig.cmake.in
                              ${REGISTRATION_BUILD_CMAKE_DIR}/RegistrationConfig.cmake
                              INSTALL_DESTINATION ${REGISTRATION_BUILD_CMAKE_DIR}
                              PATH_VARS REGISTRATION_INCLUDE_DIR)

# RegistrationConfig.cmake (installed)
set(REGISTRATION_INCLUDE_DIR "${REGISTRATION_INSTALL_INCLUDE_DIR}")
configure_package_config_file(RegistrationConfig.cmake.in
                              ${REGISTRATION_BUILD_CMAKE_DIR}/RegistrationConfigInstall.cmake
                              INSTALL_DESTINATION ${REGISTRATION_INSTALL_CMAKE_DIR}
                              PATH_VARS REGISTRATION_INCLUDE_DIR)

install(FILES ${REGISTRATION_BUILD_CMAKE_DIR}/RegistrationConfigInstall.cmake
        DESTINATION ${REGISTRATION_INSTALL_CMAKE_DIR}
        RENAME RegistrationConfig.cmake)

# RegistrationTargets.cmake (installed)
install(EXPORT RegistrationTargets
        DESTINATION ${REGISTRATION_INSTALL_CMAKE_DIR})
