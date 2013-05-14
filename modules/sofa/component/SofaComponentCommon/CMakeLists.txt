include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initComponentCommon.h

    )
    
set(SOURCE_FILES

    ../initComponentCommon.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_COMPONENT_COMMON")
set(LINKER_DEPENDENCIES SofaLoader SofaRigid SofaDeformable SofaSimpleFem SofaObjectInteraction SofaMeshCollision SofaExplicitOdeSolver SofaImplicitOdeSolver)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
