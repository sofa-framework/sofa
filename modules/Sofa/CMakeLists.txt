cmake_minimum_required(VERSION 3.1)

##########################################

add_subdirectory(Component)

##########################################

project(SofaNg)

find_package(Sofa.Component REQUIRED)

set(HEADER_FILES
    config/${PROJECT_NAME}.h
)

set(SOURCE_FILES
    config/${PROJECT_NAME}.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DBUILD_SOFA")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/config>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>")

sofa_create_package(${PROJECT_NAME} "1.0" ${PROJECT_NAME} ${PROJECT_NAME})
