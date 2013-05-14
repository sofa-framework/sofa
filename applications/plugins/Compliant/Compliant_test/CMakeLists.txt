include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    Compliant_test.cpp
	)

if(APPLE)
    set(RC_FILES "runSOFA.icns")
endif()

include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")
link_directories("${SOFA_EXTLIBS_DIR}/gtest/lib")

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

set(COMPILER_DEFINES "")
set(LINKER_DEPENDENCIES Compliant_lib SofaComponentMain SofaSimulationGraph gtest gtest_main)

if(WIN32)
	set(COMPILER_FLAGS ${COMPILER_FLAGS} "/MT")
endif()

if(UNIX)
    list(APPEND LINKER_DEPENDENCIES "dl")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)