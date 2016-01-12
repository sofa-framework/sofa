cmake_minimum_required(VERSION 2.8.12)
project(SofaHaptics)

set(HEADER_FILES
    # EnslavementForceFeedback.h
    ForceFeedback.h
    LCPForceFeedback.h
    LCPForceFeedback.inl
    MechanicalStateForceFeedback.h
    NullForceFeedback.h
    NullForceFeedbackT.h
    config.h
    initHaptics.h
)

set(SOURCE_FILES
    # EnslavementForceFeedback.cpp
    LCPForceFeedback.cpp
    NullForceFeedback.cpp
    NullForceFeedbackT.cpp
    initHaptics.cpp
)

find_package(Boost QUIET REQUIRED COMPONENTS thread)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES} SofaConstraint)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_HAPTICS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
