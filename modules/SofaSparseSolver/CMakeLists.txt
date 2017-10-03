cmake_minimum_required(VERSION 3.1)
project(SofaSparseSolver)

message(STATUS "SofaSparseSolver:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initSparseSolver.h
)

set(SOURCE_FILES
    initSparseSolver.cpp
)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
        PrecomputedLinearSolver.h
        PrecomputedLinearSolver.inl
        SparseCholeskySolver.h
        SparseLUSolver.h
        )
    list(APPEND SOURCE_FILES
        PrecomputedLinearSolver.cpp
        SparseCholeskySolver.cpp
        SparseLUSolver.cpp
        )

    message(STATUS "   With all maintained sofa components.")

    if(SOFA_HAVE_METIS)
        list(APPEND HEADER_FILES SparseLDLSolver.h SparseLDLSolver.inl SparseLDLSolverImpl.h)
        list(APPEND SOURCE_FILES SparseLDLSolver.h SparseLDLSolver.cpp)
        message(STATUS "   Metis was found, SparseLDLSolver will be built")
    else()
        message(STATUS "   Metis was not found, SparseLDLSolver won't be built")
    endif()

    if(SOFA_HAVE_CSPARSE)
        list(APPEND HEADER_FILES SparseCholeskySolver.h)
        list(APPEND HEADER_FILES SparseLUSolver.h)
        list(APPEND SOURCE_FILES SparseCholeskySolver.cpp)
        list(APPEND SOURCE_FILES SparseLUSolver.cpp)
        message(STATUS "   Csparse was found, SparseLUSolver and SparseCholeskySolver will be built")
    else()
        message(STATUS "   Csparse was not found, SparseLUSolver and SparseCholeskySolver won't be built")
    endif()
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "   With deprecated components.")
else()
    message(STATUS "   Without deprecated components.")
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver SofaImplicitOdeSolver SofaSimpleFem)
if(Metis_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC metis)
endif()
if(SOFA_HAVE_CSPARSE)
    target_link_libraries(${PROJECT_NAME} PUBLIC csparse)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SPARSE_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
