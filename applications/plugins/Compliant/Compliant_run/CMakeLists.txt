include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    Compliant_run.cpp
	)

if(UNIX)
    set(RC_FILES sofa.rc)
endif()

if(APPLE)
    set(RC_FILES "runSOFA.incs")
endif()

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

set(COMPILER_DEFINES "")
set(LINKER_DEPENDENCIES Compliant_lib SofaComponentMain SofaGuiMain SofaGuiQt)

if(UNIX)
    list(APPEND LINKER_DEPENDECIES "dl")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)