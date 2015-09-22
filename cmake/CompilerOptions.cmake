#### Compiler options

## GCC-specific
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    ## Find out the version of g++ (and save it in GCXX_VERSION)
    if(CMAKE_CXX_COMPILER_ARG1) # CXX="ccache g++"
        string(STRIP ${CMAKE_CXX_COMPILER_ARG1} CMAKE_CXX_COMPILER_ARG1_stripped)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER_ARG1_stripped} -dumpversion OUTPUT_VARIABLE GCXX_VERSION)
    else()
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCXX_VERSION)
    endif()

    ## Disable partial inlining under gcc 4.6 (Why?)
    if("${GCXX_VERSION}" VERSION_EQUAL 4.6)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-partial-inlining")
    endif()

    # stack-protector
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fstack-protector --param=ssp-buffer-size=4")
    # _FORTIFY_SOURCE
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")

    if(CMAKE_BUILD_TYPE MATCHES "Release")
        # ???
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined -lc ${CMAKE_SHARED_LINKER_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "-Wl,--no-undefined -lc ${CMAKE_MODULE_LINKER_FLAGS}")
    endif()
endif()

## GCC/Clang-specific
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    # Warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W")
endif()

## Windows-specific
if(WIN32)
    add_definitions("-DUNICODE")
    add_definitions("-D_USE_MATH_DEFINES") # just to access M_PI with cmath
    add_definitions("-wd4250 -wd4251 -wd4275 -wd4675 -wd4996")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

## XBox-specific
if(XBOX)
    add_definitions("-wd4250 -wd4231 /GR /EHsc")
endif()

## SOFA_DEBUG preprocessor macro
if(WIN32 OR APPLE)
    # Reminder: multi-configuration generators like Visual Studio and XCode do
    # not use CMAKE_BUILD_TYPE, as they generate all configurations in the
    # project, not just one at a time!
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DSOFA_DEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSOFA_DEBUG")
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions("-DSOFA_DEBUG")
endif()
