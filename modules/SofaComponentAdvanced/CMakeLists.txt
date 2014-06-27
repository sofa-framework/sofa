cmake_minimum_required(VERSION 2.8.11)
project(SofaComponentAdvanced)

set(HEADER_FILES

    initComponentAdvanced.h

    )

set(SOURCE_FILES

    initComponentAdvanced.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaEulerianFluid SofaSphFluid SofaVolumetricData SofaNonUniformFem)

if(SOFA_HAVE_EIGEN2)
    target_link_libraries(${PROJECT_NAME} SofaEigen2Solver)
endif()

set( COMPILER_FLAGS "-DSOFA_BUILD_COMPONENT_ADVANCED")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAADVANCED_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaAdvanced_libraries
        EXPORT  SofaAdvancedTargets
        RUNTIME DESTINATION "${SOFAADVANCED_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAADVANCED_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAADVANCED_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAADVANCED_INSTALL_INCLUDE_DIR}/SofaComponentAdvanced"
        )
