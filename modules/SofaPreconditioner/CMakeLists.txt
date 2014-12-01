cmake_minimum_required(VERSION 2.8)
project(SofaPreconditioner)

set(HEADER_FILES

    initPreconditioner.h
    ShewchukPCGLinearSolver.h
    JacobiPreconditioner.h
    JacobiPreconditioner.inl
    BlockJacobiPreconditioner.h
    BlockJacobiPreconditioner.inl
    SSORPreconditioner.h
    SSORPreconditioner.inl
    WarpPreconditioner.h
    WarpPreconditioner.inl
    PrecomputedWarpPreconditioner.h
    PrecomputedWarpPreconditioner.inl
    )

set(SOURCE_FILES

    initPreconditioner.cpp
    ShewchukPCGLinearSolver.cpp
    JacobiPreconditioner.cpp
    BlockJacobiPreconditioner.cpp
    SSORPreconditioner.cpp
    WarpPreconditioner.cpp
    PrecomputedWarpPreconditioner.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaImplicitOdeSolver SofaSimpleFem SofaBaseLinearSolver)
#if(SOFA_HAVE_CSPARSE)
#    target_link_libraries( ${PROJECT_NAME} SofaSparseSolver )
#endif()
#if(SOFA_HAVE_NEWMAT)
#    target_link_libraries( ${PROJECT_NAME} SofaDenseSolver)
#endif()
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAPRECONDITIONER_COMPILER_FLAGS "-DSOFA_BUILD_PRECONDITIONER")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAPRECONDITIONER_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION "${SOFAGENERAL_INSTALL_BIN_DIR}" COMPONENT SofaGeneral_libraries
        LIBRARY DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}" COMPONENT SofaGeneral_libraries
        ARCHIVE DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}" COMPONENT SofaGeneral_libraries
        PUBLIC_HEADER DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" COMPONENT SofaGeneral_headers
)
