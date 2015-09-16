cmake_minimum_required(VERSION 2.8.12)
project(SofaPreconditioner)

set(HEADER_FILES
    BlockJacobiPreconditioner.h
    BlockJacobiPreconditioner.inl
    JacobiPreconditioner.h
    JacobiPreconditioner.inl
    PrecomputedWarpPreconditioner.h
    PrecomputedWarpPreconditioner.inl
    SSORPreconditioner.h
    SSORPreconditioner.inl
    ShewchukPCGLinearSolver.h
    WarpPreconditioner.h
    WarpPreconditioner.inl
    config.h
    initPreconditioner.h
)

set(SOURCE_FILES
    BlockJacobiPreconditioner.cpp
    JacobiPreconditioner.cpp
    PrecomputedWarpPreconditioner.cpp
    SSORPreconditioner.cpp
    ShewchukPCGLinearSolver.cpp
    WarpPreconditioner.cpp
    initPreconditioner.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaImplicitOdeSolver SofaSimpleFem SofaBaseLinearSolver)
if(SOFA_HAVE_CSPARSE)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaSparseSolver)
endif()
if(SOFA_HAVE_NEWMAT)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaDenseSolver)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_PRECONDITIONER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
