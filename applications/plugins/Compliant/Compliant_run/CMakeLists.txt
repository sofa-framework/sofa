include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    Compliant_run.cpp
	)

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddLinkerDependencies(Compliant_lib SofaComponentMain SofaGuiMain SofaGuiQt)

if(UNIX)
    AddLinkerDependencies("dl")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)