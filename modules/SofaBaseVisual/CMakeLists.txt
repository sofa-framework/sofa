cmake_minimum_required(VERSION 2.8.11)
project(SofaBaseVisual)

set(HEADER_FILES

#    BaseVisual.h
    BaseCamera.h
    InteractiveCamera.h
    RecordedCamera.h
    VisualModelImpl.h
    VisualStyle.h
    VisualTransform.h

    )

set(SOURCE_FILES

#    BaseVisual.cpp
    BaseCamera.cpp
    InteractiveCamera.cpp
    RecordedCamera.cpp
    VisualModelImpl.cpp
    VisualStyle.cpp
    VisualTransform.cpp

    )


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaCore SofaBaseTopology)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFABASE_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFABASE_DEPENDENCY_INCLUDE_DIR}>")

set( COMPILER_FLAGS "-DSOFA_BUILD_BASE_VISUAL")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFABASE_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaBaseTargets
        RUNTIME DESTINATION "${SOFABASE_INSTALL_BIN_DIR}" COMPONENT SofaBase_libraries
        LIBRARY DESTINATION "${SOFABASE_INSTALL_LIB_DIR}" COMPONENT SofaBase_libraries
        ARCHIVE DESTINATION "${SOFABASE_INSTALL_LIB_DIR}" COMPONENT SofaBase_libraries
        PUBLIC_HEADER DESTINATION "${SOFABASE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" COMPONENT SofaBase_headers
        )
