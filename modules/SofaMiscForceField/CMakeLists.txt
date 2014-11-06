cmake_minimum_required(VERSION 2.8)
project(SofaMiscForceField)

set(HEADER_FILES

    initMiscForcefield.h
    MatrixMass.h
    MatrixMass.inl
    MeshMatrixMass.h
    MeshMatrixMass.inl
    LennardJonesForceField.h
    LennardJonesForceField.inl
    WashingMachineForceField.h
    WashingMachineForceField.inl
    GearSpringForceField.h
    GearSpringForceField.inl
    LineBendingSprings.h
    LineBendingSprings.inl
    )

set(SOURCE_FILES

    initMiscForcefield.cpp
    MatrixMass.cpp
    MeshMatrixMass.cpp
    LennardJonesForceField.cpp
    WashingMachineForceField.cpp
    GearSpringForceField.cpp
    LineBendingSprings.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaDeformable SofaBoundaryCondition SofaMiscTopology)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_DEPENDENCY_DIR}>")

set( SOFAMISCFORCEFIELD_COMPILER_FLAGS "-DSOFA_BUILD_MISC_FORCEFIELD -DPOINT_DATA_VECTOR_ACCESS")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAMISC_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAMISCFORCEFIELD_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaMiscForceField_libraries
        EXPORT    SofaMiscTargets
        RUNTIME DESTINATION "${SOFAMISC_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAMISC_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
)
