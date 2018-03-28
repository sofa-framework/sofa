cmake_minimum_required(VERSION 3.1)
project(Sofa.Component.Utils)

find_package(SofaFramework REQUIRED)

set(HEADER_FILES config/${PROJECT_NAME}.h)
set(SOURCE_FILES config/${PROJECT_NAME}.cpp)

list(APPEND HEADER_FILES
    deprecated_layout/SofaComponentBase/MakeAliasComponent.h
    deprecated_layout/SofaComponentBase/MakeDataAliasComponent.h
    deprecated_layout/SofaComponentBase/MessageHandlerComponent.h
    deprecated_layout/SofaComponentBase/InfoComponent.h
    )

list(APPEND HEADER_FILES
    src/sofa/component/utils/MakeAliasComponent.h
    src/sofa/component/utils/MakeDataAliasComponent.h
    src/sofa/component/utils/MessageHandlerComponent.h
    src/sofa/component/utils/InfoComponent.h
    )
list(APPEND SOURCE_FILES
    src/sofa/component/utils/MakeAliasComponent.cpp
    src/sofa/component/utils/MakeDataAliasComponent.cpp
    src/sofa/component/utils/MessageHandlerComponent.cpp
    src/sofa/component/utils/InfoComponent.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaHelper)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DBUILD_SOFA_COMPONENT_UTILS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/config>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/deprecated_layout>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>")

sofa_create_package(${PROJECT_NAME} "1.0" ${PROJECT_NAME} ${PROJECT_NAME})
