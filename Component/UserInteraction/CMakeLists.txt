cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.UserInteraction LANGUAGES CXX)

set(SOFACOMPONENTUSERINTERACTION_SOURCE_DIR "src/sofa/component/userinteraction")

set(SOFACOMPONENTUSERINTERACTION_DIRS ConfigurationSetting Controller Performer)
set(SOFACOMPONENTUSERINTERACTION_TARGETS)
foreach(userinteraction_dir ${SOFACOMPONENTUSERINTERACTION_DIRS})
    sofang_add_component_subdirectory(${userinteraction_dir} ${PROJECT_NAME}.${userinteraction_dir})
    list(APPEND SOFACOMPONENTUSERINTERACTION_TARGETS ${PROJECT_NAME}.${userinteraction_dir})
endforeach()


set(HEADER_FILES
    ${SOFACOMPONENTUSERINTERACTION_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTUSERINTERACTION_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTUSERINTERACTION_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTUSERINTERACTION_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
