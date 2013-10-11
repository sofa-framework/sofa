include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initSparseSolver.h 
    ../linearsolver/PrecomputedLinearSolver.h 
    ../linearsolver/PrecomputedLinearSolver.inl 
    ../linearsolver/SparseCholeskySolver.h 
    ../linearsolver/SparseLUSolver.h 
    ../linearsolver/SparseLDLSolverImpl.h 
    ../linearsolver/SparseLDLSolver.h 
    ../linearsolver/SparseLDLSolver.inl
    )
    
set(SOURCE_FILES

    ../initSparseSolver.cpp 
    ../linearsolver/PrecomputedLinearSolver.cpp 
    ../linearsolver/SparseCholeskySolver.cpp 
    ../linearsolver/SparseLUSolver.cpp 
    ../linearsolver/SparseLDLSolver.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SPARSE_SOLVER" )
AddLinkerDependencies(SofaBaseLinearSolver SofaImplicitOdeSolver csparse newmat)

if(SOFA-EXTERNAL_METIS)
        AddLinkerDependencies(${METIS_LIBRARIES})
        include_directories(${METIS_INCLUDE_DIR})
        find_package(MPI REQUIRED)
        include_directories(${MPI_INCLUDE_DIR})
        AddLinkerDependencies(${MPI_LIBRARY})
        AddLinkerDependencies(${MPI_CXX_LIBRARY})
endif()
    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
