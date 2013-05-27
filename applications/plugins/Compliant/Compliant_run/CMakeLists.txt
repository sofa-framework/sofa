include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    Compliant_run.cpp
	)

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

set(COMPILER_DEFINES "")
set(LINKER_DEPENDENCIES Compliant_lib SofaComponentMain SofaGuiMain SofaGuiQt)

if(UNIX)
    list(APPEND LINKER_DEPENDENCIES "dl")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)