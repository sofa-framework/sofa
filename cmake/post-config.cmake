cmake_minimum_required(VERSION 2.8)

# compile flags
if(WIN32)
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-wd4250 -wd4251 -wd4275 -wd4675 -wd4996 /bigobj")
endif()