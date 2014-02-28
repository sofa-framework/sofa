include(${SOFA_CMAKE_DIR}/preProject.cmake)

include_directories("${SOFA_EXTLIBS_DIR}/PML" "${SOFA_EXTLIBS_DIR}/LML")
AddLinkerDependencies(SofaSimulationCommon SofaDeformable xml2 PML LML)

file(GLOB HEADER_FILES *.h)
file(GLOB SOURCE_FILES *.cpp)
AddCompilerDefinitions("SOFA_BUILD_FILEMANAGER_PML")
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)
