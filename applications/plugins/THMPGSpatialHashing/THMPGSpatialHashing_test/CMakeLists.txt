include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    THMPGSpatialHashingTest.cpp
)

include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")
link_directories("${SOFA_EXTLIBS_DIR}/gtest/lib")

#add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddCompilerDefinitions("")
AddLinkerDependencies(THMPGSpatialHashing SofaGTestMain)
#AddLinkerDependencies(SofaTest_test)
AddLinkerDependencies(SofaSimulationGraph)
AddLinkerDependencies(SceneCreator)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
