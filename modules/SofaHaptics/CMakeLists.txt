cmake_minimum_required(VERSION 3.1)
project(SofaHaptics VERSION 1.0 LANGUAGES CXX)

find_package(SofaGeneral REQUIRED) # To have SofaConstraint

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
target_compile_definitions(${PROJECT_NAME} PRIVATE COMPILE_FLAGS "-DSOFA_BUILD_SOFAHAPTICS")
target_compile_definitions(${PROJECT_NAME} PUBLIC COMPILE_FLAGS "-DSOFA_HAVE_SOFAHAPTICS")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

sofa_generate_package(NAME ${PROJECT_NAME}
                      VERSION ${PROJECT_VERSION}
                      TARGETS ${PROJECT_NAME}
                      INCLUDE_ROOT_DIR ${PROJECT_NAME})
