message("CPACK_GENERATOR= ${CPACK_GENERATOR}")

execute_process(COMMAND bash "@CMAKE_SOURCE_DIR@/tools/postinstall-fixup/generate-stubfiles.sh" "@CMAKE_BINARY_DIR@" "${CPACK_TEMPORARY_INSTALL_DIRECTORY}" "${CMAKE_SYSTEM_NAME}:${CPACK_GENERATOR}")
