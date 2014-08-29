include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES Compliant_run.cpp)

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddLinkerDependencies(Compliant SofaComponentMain SofaGuiMain)

if(UNIX)
    AddLinkerDependencies("dl")
    # Compliant uses Boost.Chrono as a header-only library, in such a way that it
    # should *not* depend on Boost.System (see Compliant/numericalsolver/Benchmark.h).
    # But for some obscure reason, on linux and OS X, Compliant_run contains
    # references to symbols of Boost.System anyway, so we link against it.
    find_package(Boost COMPONENTS system REQUIRED)
    AddLinkerDependencies(${Boost_SYSTEM_LIBRARY})
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
