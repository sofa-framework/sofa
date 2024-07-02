cmake_minimum_required(VERSION 3.22)
project(Sofa.Http LANGUAGES CXX)

set(SOFAHTTP_SRC "src/sofa/http")

set(HEADER_FILES
    ${SOFAHTTP_SRC}/config.h.in
    ${SOFAHTTP_SRC}/init.h

    ${SOFAHTTP_SRC}/VersionChecker.h
)

set(SOURCE_FILES
    ${SOFAHTTP_SRC}/init.cpp

    ${SOFAHTTP_SRC}/VersionChecker.cpp
)

find_package(Sofa.Helper REQUIRED)

find_package(httplib QUIET)
if(NOT httplib_FOUND AND SOFA_ALLOW_FETCH_DEPENDENCIES)
    message("${PROJECT_NAME}: DEPENDENCY httplib NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is ON, fetching httplib...")

    include(FetchContent)
    FetchContent_Declare(httplib
            GIT_REPOSITORY https://github.com/yhirose/cpp-httplib
            GIT_TAG        v0.15.3
            GIT_SHALLOW    TRUE
    )

    set(HTTPLIB_REQUIRE_OPENSSL ON CACHE INTERNAL "") #This requires OpenSSL >=3.0.0
    FetchContent_MakeAvailable(httplib)

elseif (NOT httplib_FOUND)
    message(FATAL_ERROR "${PROJECT_NAME}: DEPENDENCY httplib NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is OFF and thus cannot be fetched. Install httplib, or enable SOFA_ALLOW_FETCH_DEPENDENCIES to fix this issue.")
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Helper)
target_link_libraries(${PROJECT_NAME} PUBLIC httplib)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
