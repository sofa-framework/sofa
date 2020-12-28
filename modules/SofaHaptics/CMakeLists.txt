cmake_minimum_required(VERSION 3.12)
project(SofaHaptics LANGUAGES CXX)

find_package(SofaConstraint REQUIRED)

set(HEADER_FILES
    src/SofaHaptics/config.h
    src/SofaHaptics/initHaptics.h
    src/SofaHaptics/ForceFeedback.h
    src/SofaHaptics/LCPForceFeedback.h
    src/SofaHaptics/LCPForceFeedback.inl
    src/SofaHaptics/MechanicalStateForceFeedback.h
    src/SofaHaptics/NullForceFeedback.h
    src/SofaHaptics/NullForceFeedbackT.h
    )

set(SOURCE_FILES
    src/SofaHaptics/initHaptics.cpp
    src/SofaHaptics/ForceFeedback.cpp
    src/SofaHaptics/LCPForceFeedback.cpp
    src/SofaHaptics/NullForceFeedback.cpp
    src/SofaHaptics/NullForceFeedbackT.cpp
    )

set(EXTRA_FILES
    README.md
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaConstraint)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )
    
# Tests
cmake_dependent_option(SOFAHAPTICS_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAHAPTICS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaHaptics_test)
endif()
