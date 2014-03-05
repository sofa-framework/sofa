include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    oneParticleWithSofaTypedefs.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set_source_files_properties(${SOURCE_FILES} PROPERTIES OBJECT_DEPENDS SofaTypedefs)

AddLinkerDependencies(SofaSimulationTree SofaComponentMain SofaGuiMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

