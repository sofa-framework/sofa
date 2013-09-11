cmake_minimum_required(VERSION 2.8.8)

set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

# MSVC: Use solution folders.
if(MSVC)
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

include(cmake/preMainSolution.cmake)

if(PRECONFIGURE_DONE)
	
	include(cmake/preSolution.cmake)

	include(cmake/dependencies.cmake)
	
	include(cmake/postSolution.cmake)

endif()

include(cmake/postMainSolution.cmake)
