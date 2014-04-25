include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
	qtOnePick.cpp
	QtViewer.cpp
)
set(HEADER_FILES
	QtViewer.h
)

UseQt()
	
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

AddLinkerDependencies(SofaSimpleGui)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

