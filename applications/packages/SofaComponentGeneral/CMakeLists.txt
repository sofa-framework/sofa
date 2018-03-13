cmake_minimum_required(VERSION 3.1)
project(SofaComponentGeneral)

set(HEADER_FILES
    config.h
    initComponentGeneral.h
    )

set(SOURCE_FILES
    initComponentGeneral.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC
    SofaGeneralAnimationLoop SofaGeneralDeformable SofaGeneralExplicitOdeSolver SofaGeneralImplicitOdeSolver SofaGeneralLinearSolver SofaGeneralLoader
    SofaGeneralMeshCollision SofaGeneralObjectInteraction SofaGeneralRigid SofaGeneralSimpleFem SofaGeneralTopology  SofaGeneralVisual SofaBoundaryCondition
    SofaConstraint SofaGeneralEngine SofaExporter SofaGraphComponent SofaHaptics SofaPreconditioner SofaSparseSolver SofaTopologyMapping SofaUserInteraction SofaValidation)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_COMPONENT_GENERAL")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")

if(SOFA_HAVE_NEWMAT)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaDenseSolver)
endif()
if(NOT SOFA_NO_OPENGL)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaOpenglVisual)
endif()

sofa_create_package(${PROJECT_NAME} "1.0" ${PROJECT_NAME} ${PROJECT_NAME})
