if(TARGET ${PROJECT_NAME})
    # group files (headers, sources, etc.)
    get_directory_property(properties VARIABLES)
    foreach(property ${properties})
        if(NOT property STREQUAL "")
            string(REGEX MATCH "^.+_FILES$" fileGroup ${property})
            if(fileGroup)
                if(NOT fileGroup MATCHES "CUDA_ADDITIONAL_CLEAN_FILES")
                    set(fileTopGroup)
                    if(${fileGroup}_GROUP)
                        set(fileTopGroup "${${fileGroup}_GROUP}")
                    endif()
                    GroupFiles("${fileGroup}" "${fileTopGroup}" "${GROUP_BASE_DIR}")
                endif()
            endif()
        endif()
    endforeach()

    # if this project is a test, add the necessary include and lib
    if(PROJECT_NAME MATCHES ".*_test.*")
        if(SOFA-MISC_BUILD_GTEST OR WIN32)
            include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")
        endif()

        link_directories("${SOFA_EXTLIBS_DIR}/gtest/lib")

        if(WIN32)
            # MSVC2012 has some troubles with the way gtest use the STL, this preprocessor macro fix this issue
            AddCompilerDefinitions("_VARIADIC_MAX=10")
        endif()

        AddLinkerDependencies(gtest gtest_main)
    endif()

    # include directories
    include_directories(${GLOBAL_INCLUDE_DIRECTORIES} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

    get_directory_property(includeDirs INCLUDE_DIRECTORIES)
    ## put includes inside a CACHE variable for further uses
    set(${PROJECT_NAME}_INCLUDE_DIR ${${PROJECT_NAME}_INCLUDE_DIR} ${includeDirs} CACHE INTERNAL "${PROJECT_NAME} include path" FORCE)

    # compile definitions
    set(ADDITIONAL_COMPILER_DEFINES ${ADDITIONAL_COMPILER_DEFINES} "SOFA_TARGET=${PROJECT_NAME}")
    set(${PROJECT_NAME}_COMPILER_DEFINES "${GLOBAL_COMPILER_DEFINES};${ADDITIONAL_COMPILER_DEFINES};${COMPILER_DEFINES}" CACHE INTERNAL "${PROJECT_NAME} compiler definitions" FORCE)
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${${PROJECT_NAME}_COMPILER_DEFINES}")

    # compile flags
    set(COMPILER_FLAGS_OS_SPECIFIC "")
    if(WIN32)
        set(COMPILER_FLAGS_OS_SPECIFIC "-wd4250 -wd4251 -wd4275 -wd4675 -wd4996 /bigobj")
    endif()
    if(XBOX)
        set(COMPILER_FLAGS_OS_SPECIFIC "-wd4250 -wd4231 /GR /EHsc /bigobj")
    endif()
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS_OS_SPECIFIC} ${COMPILER_FLAGS}")

    # link dependencies
    target_link_libraries(${PROJECT_NAME} ${ADDITIONAL_LINKER_DEPENDENCIES} ${LINKER_DEPENDENCIES})

    # store dependencies for further uses
    RegisterProjectDependencies(${PROJECT_NAME} ${SOURCE_DEPENDENCIES} ${ADDITIONAL_LINKER_DEPENDENCIES} ${LINKER_DEPENDENCIES})

    #link flags
    set(LINKER_FLAGS_OS_SPECIFIC "")
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${LINKER_FLAGS_OS_SPECIFIC} ${LINKER_FLAGS}")

    # output name suffixes
    set(LIBSUFFIX)
    if(WIN32)
        set(LIBSUFFIX "_${SOFA_VERSION_NUM}")
    endif()
    set(BUILDSUFFIX)
    get_target_property(BUILDTYPE ${PROJECT_NAME} TYPE)
    if(NOT "${BUILDTYPE}" STREQUAL "EXECUTABLE")
        set(BUILDSUFFIX ${LIBSUFFIX})
    endif()
    set_target_properties(${PROJECT_NAME} PROPERTIES
        DEBUG_POSTFIX "${BUILDSUFFIX}d"
        RELEASE_POSTFIX "${BUILDSUFFIX}"
        RELWITHDEBINFO_POSTFIX "${BUILDSUFFIX}rd"
        MINSIZEREL_POSTFIX "${BUILDSUFFIX}"
        COMPILE_DEFINITIONS_DEBUG "SOFA_LIBSUFFIX=${LIBSUFFIX}d"
		COMPILE_DEFINITIONS_RELWITHDEBINFO "SOFA_LIBSUFFIX=${LIBSUFFIX}rd"
        )
	if(NOT "${LIBSUFFIX}" STREQUAL "")
		set_target_properties(${PROJECT_NAME} PROPERTIES
			COMPILE_DEFINITIONS_RELEASE "SOFA_LIBSUFFIX=${LIBSUFFIX}"
			COMPILE_DEFINITIONS_MINSIZEREL "SOFA_LIBSUFFIX=${LIBSUFFIX}"
			)
	endif()

    # if this project is a test, add it in the test group
    if(PROJECT_NAME MATCHES ".*_test.*")
        add_test(NAME "${PROJECT_NAME}" WORKING_DIRECTORY "${SOFA_BIN_DIR}" COMMAND ${PROJECT_NAME})
    endif()

    # set IDE project filter
    if(MSVC AND NOT "${GLOBAL_PROJECT_OPTION_FOLDER_${PROJECT_NAME}}" STREQUAL "")
        #message("${PROJECT_NAME} in ${GLOBAL_PROJECT_OPTION_FOLDER_${PROJECT_NAME}}")
        set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${GLOBAL_PROJECT_OPTION_FOLDER_${PROJECT_NAME}}")
    endif()
endif()
