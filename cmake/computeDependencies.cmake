cmake_minimum_required(VERSION 2.8)

# retrieve dependencies and include directories
message(STATUS "Compute dependencies :")
set(projectNames ${GLOBAL_DEPENDENCIES})
foreach(projectName ${projectNames})
	ComputeDependencies(${projectName} false "")
endforeach()