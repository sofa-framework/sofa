include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

	Main.cpp
	)
	
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

add_custom_command(	TARGET ${PROJECT_NAME} POST_BUILD
					COMMAND ${PROJECT_NAME})
					
AddLinkerDependencies(SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)