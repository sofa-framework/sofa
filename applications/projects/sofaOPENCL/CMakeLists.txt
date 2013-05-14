include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    Main.cpp
    )
    
set(RC_FILES

    sofa.rc
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

set(LINKER_DEPENDENCIES SofaGuiMain SofaGpuOpenCL )

include(${SOFA_CMAKE_DIR}/postProject.cmake)

