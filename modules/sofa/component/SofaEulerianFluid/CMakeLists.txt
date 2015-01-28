include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initEulerianFluid.h 
    # ../behaviormodel/eulerianfluid/Fluid2D.h 
    # ../behaviormodel/eulerianfluid/Fluid3D.h 
    # ../behaviormodel/eulerianfluid/Grid2D.h 
    # ../behaviormodel/eulerianfluid/Grid3D.h

    )
    
set(SOURCE_FILES

    ../initEulerianFluid.cpp 
    # ../behaviormodel/eulerianfluid/Fluid2D.cpp 
    # ../behaviormodel/eulerianfluid/Fluid3D.cpp 
    # ../behaviormodel/eulerianfluid/Grid2D.cpp 
    # ../behaviormodel/eulerianfluid/Grid3D.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_EULERIAN_FLUID")
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
