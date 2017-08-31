cmake_minimum_required(VERSION 3.1)
project(json C CXX)

set(HEADER_FILES
    nlohmann_json.h
    json.h
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION 0)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)


include(../../SofaFramework/SofaMacros.cmake)
sofa_create_package(json 0 ${PROJECT_NAME} "")
