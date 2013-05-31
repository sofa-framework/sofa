include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
find_package( SofaPimGUIQtLibs REQUIRED)
include_directories(${SOFAPIMSGUIQT_INCLUDE_DIRS})

set(HEADER_FILES
    
    QMouseOperations.h

	)

set(SOURCE_FILES

    QMouseOperations.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_PIMGUIQT")
AddLinkerDependencies(SofaPimGui SofaComponentCollision SofaPim)

include(${SOFA_CMAKE_DIR}/postProject.cmake)