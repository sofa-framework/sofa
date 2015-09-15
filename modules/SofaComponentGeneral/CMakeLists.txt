cmake_minimum_required(VERSION 2.8.12)
project(SofaComponentGeneral)

set(HEADER_FILES
    config.h
    initComponentGeneral.h
)

set(SOURCE_FILES
    initComponentGeneral.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBoundaryCondition SofaConstraint SofaEngine SofaExporter SofaGraphComponent SofaHaptics SofaPreconditioner SofaSparseSolver SofaTopologyMapping SofaUserInteraction SofaValidation)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_COMPONENT_GENERAL")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

if(SOFA_HAVE_NEWMAT)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaDenseSolver)
endif()
if(NOT SOFA_NO_OPENGL)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaOpenglVisual)
endif()
if(SOFA_HAVE_TAUCS)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaTaucsSolver)
endif()
if(SOFA_HAVE_EIGEN2)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaEigen2Solver)
endif()
if(SOFAGENERAL_PARDISOSOLVER)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaPardisoSolver)
endif()

SofaGeneral_install(${PROJECT_NAME})
