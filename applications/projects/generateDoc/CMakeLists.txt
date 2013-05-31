include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    
    generateDoc.h
    
    )

set(SOURCE_FILES

    generateDoc.cpp
	Main.cpp

	)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

AddLinkerDependencies(SofaComponentMain )

include(${SOFA_CMAKE_DIR}/postProject.cmake)

