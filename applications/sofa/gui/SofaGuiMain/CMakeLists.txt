cmake_minimum_required(VERSION 3.1)
project(SofaGuiMain)

set(HEADER_FILES
    ../Main.h
)

set(SOURCE_FILES
    ../Main.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set(SOFAGUIMAIN_DEPS ${SOFAGUI_TARGETS})
list(REMOVE_ITEM SOFAGUIMAIN_DEPS SofaGuiMain)
target_link_libraries(${PROJECT_NAME} ${SOFAGUIMAIN_DEPS})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GUIMAIN")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGui SofaGuiMain "sofa/gui")
