include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initOpenCTMPlugin.h
    OpenCTMLoader.h
    OpenCTMExporter.h
	)

set(SOURCE_FILES
    initOpenCTMPlugin.cpp
    OpenCTMLoader.cpp
    OpenCTMExporter.cpp
	)

set(README_FILES
    OpenCTMPlugin.txt
	)

find_package(OpenCTM REQUIRED)
include_directories("${OPENCTM_INCLUDE_PATH}")

AddCompilerDefinitions("SOFA_BUILD_OPENCTMPLUGIN")
AddLinkerDependencies(SofaBaseTopology ${OPENCTM_LIBRARY})
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})


include(${SOFA_CMAKE_DIR}/postProject.cmake)
