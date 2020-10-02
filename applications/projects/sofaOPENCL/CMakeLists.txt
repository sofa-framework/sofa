cmake_minimum_required(VERSION 3.1)
project(sofaOPENCL)

find_package(SofaGui REQUIRED)
find_package(SofaOpenCL REQUIRED)

set(SOURCE_FILES Main.cpp)
set(RC_FILES sofa.rc)

include_directories(../plugins)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})
target_link_libraries(${PROJECT_NAME} SofaGuiMain SofaOpenCL)
