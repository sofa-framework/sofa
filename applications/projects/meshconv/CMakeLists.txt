cmake_minimum_required(VERSION 3.1)
project(meshconv)

find_package(SofaGeneral)
find_package(SofaAdvanced)
find_package(SofaMisc)

if(SOFA_HAVE_MINIFLOWVR)
	add_executable(${PROJECT_NAME} meshconv.cpp tesselate.cpp)
	target_link_libraries(${PROJECT_NAME} SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)
else()
	message("WARNING: meshconv needs miniFlowVR, enable SOFA_BUILD_MINIFLOWVR if you want to build it. meshconv won't be built.")
endif()