cmake_minimum_required(VERSION 2.8)

project("GenerateRigid")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES


    )

set(SOURCE_FILES

    GenerateRigid.cpp
	Main.cpp

	)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaComponent )

include(${SOFA_CMAKE_DIR}/post.cmake)

