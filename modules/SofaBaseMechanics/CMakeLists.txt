cmake_minimum_required(VERSION 2.8.11)
project(SofaBaseMechanics)

set(HEADER_FILES
    initBaseMechanics.h
    MappedObject.h
    MappedObject.inl
    MechanicalObject.h
    MechanicalObject.inl
    AddMToMatrixFunctor.h
    DiagonalMass.h
    DiagonalMass.inl
    UniformMass.h
    UniformMass.inl
    BarycentricMapping.h
    BarycentricMapping.inl
    IdentityMapping.h
    IdentityMapping.inl
    SubsetMapping.h
    SubsetMapping.inl
    )

set(SOURCE_FILES
    initBaseMechanics.cpp
    MappedObject.cpp
    MechanicalObject.cpp
    DiagonalMass.cpp
    UniformMass.cpp
    BarycentricMapping.cpp
    IdentityMapping.cpp
    SubsetMapping.cpp
    )


if(SOFA-MISC_SMP)
    list(APPEND HEADER_FILES "MechanicalObjectTasks.inl")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationCommon SofaBaseTopology)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFABASE_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFABASE_DEPENDENCY_INCLUDE_DIR}>")

set( SOFABASEMECHANICS_COMPILER_FLAGS "-DSOFA_BUILD_BASE_MECHANICS")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFABASE_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFABASEMECHANICS_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

if(TARGET gtest)
    add_subdirectory(SofaBaseMechanics_test)
endif()


install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaBaseTargets
        RUNTIME DESTINATION "${SOFABASE_INSTALL_BIN_DIR}" COMPONENT SofaBase_libraries
        LIBRARY DESTINATION "${SOFABASE_INSTALL_LIB_DIR}" COMPONENT SofaBase_libraries
        ARCHIVE DESTINATION "${SOFABASE_INSTALL_LIB_DIR}" COMPONENT SofaBase_libraries
        PUBLIC_HEADER DESTINATION "${SOFABASE_INSTALL_INCLUDE_DIR}/SofaBaseMechanics" COMPONENT SofaBase_headers
        )
