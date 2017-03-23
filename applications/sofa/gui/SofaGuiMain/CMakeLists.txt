cmake_minimum_required(VERSION 3.1)
project(SofaGuiMain)

add_library(${PROJECT_NAME} SHARED ../Main.h ../Main.cpp)
set(SOFAGUIMAIN_DEPS ${SOFAGUI_TARGETS})
list(REMOVE_ITEM SOFAGUIMAIN_DEPS SofaGuiMain)
target_link_libraries(${PROJECT_NAME} ${SOFAGUIMAIN_DEPS})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GUIMAIN")

sofa_install_targets(SofaGui SofaGuiMain "")
install(FILES ${HEADER_FILES} DESTINATION "include/sofa/gui")
