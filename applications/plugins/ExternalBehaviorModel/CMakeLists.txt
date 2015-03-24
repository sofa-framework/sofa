cmake_minimum_required(VERSION 2.8)
project(ExternalBehaviorModel)

set(EXTERNALBEHAVIORMODEL_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(EXTERNALBEHAVIORMODEL_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(EXTERNALBEHAVIORMODEL_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    initExternalBehaviorModel.h
    FEMGridBehaviorModel.h
    FEMGridBehaviorModel.inl)

set(SOURCE_FILES
    initExternalBehaviorModel.cpp
    FEMGridBehaviorModel.cpp)


find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_ExternalBehaviorModel")
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral)

install(TARGETS ${PROJECT_NAME}
    COMPONENT ExternalBehaviorModel_libraries
    EXPORT ExternalBehaviorModelTargets
    RUNTIME DESTINATION "${EXTERNALBEHAVIORMODEL_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${EXTERNALBEHAVIORMODEL_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${EXTERNALBEHAVIORMODEL_INSTALL_LIB_DIR}")
