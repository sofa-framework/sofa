include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    chainHybrid.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(SofaSimulationTree SofaComponentMain SofaGuiMain ObjectCreator)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

