cmake_minimum_required(VERSION 2.8)

# compile flags
if(CMAKE_HOST_WIN32)
	set_target_properties(${name} PROPERTIES COMPILE_FLAGS "/bigobj")
endif()

message("> ${PROJECT_NAME} : Done\n")