cmake_minimum_required(VERSION 2.8)

if(FIRST_CONFIGURE_DONE)
	# print report
	if(SOFA_ERROR_MESSAGE)
		message(SEND_ERROR "Final report : ${SOFA_ERROR_MESSAGE}")
	endif()
	
	message(STATUS "--------------------------------------------")
	message(STATUS "----- DONE CONFIGURING SOFA FRAMEWORK ------")
	message(STATUS "--------------------------------------------")
	message(STATUS "")
endif()

set(FIRST_CONFIGURE_DONE 1 CACHE INTERNAL "First configure does not set every projects up, it just displays options")