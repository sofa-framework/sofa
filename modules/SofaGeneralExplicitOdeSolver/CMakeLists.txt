cmake_minimum_required(VERSION 3.1)
project(SofaGeneralExplicitOdeSolver)

message(STATUS "SofaGeneralExplicitOdeSolver:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initGeneralExplicitODESolver.h
)

set(SOURCE_FILES
    initGeneralExplicitODESolver.cpp
)


############################## COMPONENTS HERE ARE THE LIGHT-SET ###################################
if(SOFA_BUILD_COMPONENTSET_LIGHT)
    list(APPEND HEADER_FILES

        )
    list(APPEND SOURCE_FILES

        )
endif()

############################## COMPONENTS HERE ARE THE STANDARD-SET ################################
if(SOFA_BUILD_COMPONENTSET_STANDARD)
    list(APPEND HEADER_FILES
        CentralDifferenceSolver.h
        RungeKutta2Solver.h
        RungeKutta4Solver.h
        )
    list(APPEND SOURCE_FILES
        CentralDifferenceSolver.cpp
        RungeKutta2Solver.cpp
        RungeKutta4Solver.cpp
        )
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_EXPLICIT_ODE_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
