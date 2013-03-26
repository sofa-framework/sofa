cmake_minimum_required(VERSION 2.8)

project("RunSofa")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(SOURCE_FILES

	Main.cpp
	)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_SOFAPYTHON")
set(LINKER_DEPENDENCIES SofaGuiMain SofaComponent)

include(${SOFA_CMAKE_DIR}/post.cmake)

