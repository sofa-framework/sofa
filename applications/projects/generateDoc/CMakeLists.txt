cmake_minimum_required(VERSION 2.8)

project("GenerateDoc")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES
    
    generateDoc.h
    
    )

set(SOURCE_FILES

    generateDoc.cpp
	Main.cpp

	)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaComponent )

include(${SOFA_CMAKE_DIR}/post.cmake)

