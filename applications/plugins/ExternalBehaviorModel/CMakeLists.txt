cmake_minimum_required(VERSION 2.8)
project(ExternalBehaviorModel)


set(HEADER_FILES
    FEMGridBehaviorModel.h
    FEMGridBehaviorModel.inl
    config.h
)

set(SOURCE_FILES
    FEMGridBehaviorModel.cpp
    initExternalBehaviorModel.cpp
)


find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_ExternalBehaviorModel")
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral)

install(TARGETS ${PROJECT_NAME}
    COMPONENT ExternalBehaviorModel_libraries
    EXPORT ExternalBehaviorModelTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY example/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
