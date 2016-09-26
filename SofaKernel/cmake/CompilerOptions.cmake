#### Compiler options

## GCC-specific
if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    ## Find out the version of g++ (and save it in GCXX_VERSION)
    if(CMAKE_CXX_COMPILER_ARG1) # CXX="ccache g++"
        string(STRIP ${CMAKE_CXX_COMPILER_ARG1} CMAKE_CXX_COMPILER_ARG1_stripped)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER_ARG1_stripped} -dumpversion OUTPUT_VARIABLE GCXX_VERSION)
    else()
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCXX_VERSION)
    endif()

    ## Disable partial inlining under gcc 4.6 (Why? -> the memory was exploding)
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
if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU" OR ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # Warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W")
endif()

## Windows-specific
if(WIN32)
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



## OpenMP
option(SOFA_OPENMP "Compile Sofa with OpenMP multithreading." OFF)
if(SOFA_OPENMP)
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
    #    target_compile_options(SofaHelper PUBLIC "${OpenMP_CXX_FLAGS}") # it is not enough to declare it for SofaHelper, because a link flag is also expected
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        message("WARNING: Your compiler does not implement OpenMP.")
    endif()
endif()



# C++11 is now mandatory
# TODO how to propagate such properties to dependents?
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# An important C++11 feature may be not enabled due to
# the compiler being built without the --enable-libstdcxx-time option.
if(CMAKE_COMPILER_IS_GNUCXX)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_NANOSLEEP -D_GLIBCXX_USE_SCHED_YIELD")
    endif()
endif()

# hack for clang on old macosx (version < 10.9, such as the dashboard servers)
# that is using, by default at that time, a libstdc++ that did not fully implement c++11
if(APPLE AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND CMAKE_SYSTEM_VERSION VERSION_LESS "10.9" AND ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif()

