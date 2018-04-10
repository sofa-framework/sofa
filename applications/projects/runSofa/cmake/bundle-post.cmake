
message("POST BUNDLE SCRIPT")


file(GLOB_RECURSE VERSIONED_LIBS "${CMAKE_INSTALL_PREFIX}/runSofa.app/Contents/MacOS/*.*.dylib")
foreach(LIB ${VERSIONED_LIBS})
    get_filename_component(LIB_NAME "${LIB}" NAME_WE)
    if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/runSofa.app/Contents/MacOS/${LIB_NAME}.dylib")
    	message("create_symlink ${CMAKE_INSTALL_PREFIX}/runSofa.app/Contents/MacOS/${LIB_NAME}.dylib")
		execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${LIB} ${CMAKE_INSTALL_PREFIX}/runSofa.app/Contents/MacOS/${LIB_NAME}.dylib)
	endif()
endforeach()

