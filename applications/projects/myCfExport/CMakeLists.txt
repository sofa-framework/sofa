include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    myCfExport.cpp
    )


add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddLinkerDependencies(SofaSimulationGraph SofaComponentMain )
AddCompilerDefinitions("MyCfExport_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")


include(${SOFA_CMAKE_DIR}/postProject.cmake)

