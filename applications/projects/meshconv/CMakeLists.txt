cmake_minimum_required(VERSION 3.1)
project(meshconv)

find_package(SofaGeneral)
find_package(SofaMisc)

find_package(MiniFlowVR QUIET)
if(MiniFlowVR_FOUND)
    add_executable(${PROJECT_NAME} meshconv.cpp tesselate.cpp)
    target_link_libraries(${PROJECT_NAME} SofaGeneral SofaMisc miniFlowVR)
else()
    message("WARNING: meshconv needs miniFlowVR, enable SOFA_BUILD_MINIFLOWVR if you want to build it. meshconv won't be built.")
endif()

