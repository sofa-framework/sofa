cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

include(cmake/preMainSolution.cmake)

if(PRECONFIGURE_DONE)
	
	include(cmake/preSolution.cmake)

	include(sofa-dependencies.cmake)
	
	include(cmake/postSolution.cmake)

endif()

include(cmake/postMainSolution.cmake)