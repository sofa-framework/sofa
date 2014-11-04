cmake_minimum_required(VERSION 2.8.12)

if(WIN32)
    set(SOFA_INSTALL_SHARE_DIR share CACHE PATH "Installation directory for shares")
else()
    set(SOFA_INSTALL_SHARE_DIR share/sofa CACHE PATH "Installation directory for shares")
endif()

if(WIN32)
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
else()
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}
                                                               ${CMAKE_BINARY_DIR}/sofa/share )
endif()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION ${SOFA_INSTALL_SHARE_DIR}
        PATTERN "CVS" EXCLUDE
        PATTERN "SVN" EXCLUDE
        PATTERN "GIT" EXCLUDE
        PATTERN "CMakeLists.txt" EXCLUDE
        )
