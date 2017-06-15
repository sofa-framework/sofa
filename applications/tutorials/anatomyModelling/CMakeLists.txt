cmake_minimum_required(VERSION 3.1)
project(anatomyModelling)

find_package(SofaGui)
find_package(SceneCreator)
find_package(Flexible QUIET)
find_package(image QUIET)

add_executable(${PROJECT_NAME} anatomyModelling.cpp)

if(Flexible_FOUND)
	include_directories("${CMAKE_SOURCE_DIR}/applications/plugins/Flexible")
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_HAVE_PLUGIN_Flexible")
	target_link_libraries(${PROJECT_NAME} Flexible)
endif()

if(image_FOUND)
	include_directories("${CMAKE_SOURCE_DIR}/applications/plugins/image")
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_HAVE_IMAGE")
	target_link_libraries(${PROJECT_NAME} image)
endif()

target_link_libraries(${PROJECT_NAME}  SofaGuiMain SceneCreator   )
