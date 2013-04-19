cmake_minimum_required(VERSION 2.8)

if(PRECONFIGURE_DONE)
	# print report
	if(SOFA_ERROR_MESSAGE)
		message(AUTHOR_WARNING "Final report : ${SOFA_ERROR_MESSAGE}")
	endif()
	
	message(STATUS "--------------------------------------------")
	message(STATUS "----- DONE CONFIGURING SOFA FRAMEWORK ------")
	message(STATUS "--------------------------------------------")
	message(STATUS "")
endif()

set(PRECONFIGURE_DONE 1 CACHE INTERNAL "Configure does not set projects up, it just displays options")