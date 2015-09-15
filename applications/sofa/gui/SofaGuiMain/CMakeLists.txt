cmake_minimum_required(VERSION 2.8.12)
project(SofaGuiMain)

add_library(${PROJECT_NAME} SHARED ../Main.h ../Main.cpp)
set(SOFAGUIMAIN_DEPS ${SOFAGUI_TARGETS})
list(REMOVE_ITEM SOFAGUIMAIN_DEPS SofaGuiMain)
target_link_libraries(${PROJECT_NAME} ${SOFAGUIMAIN_DEPS})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GUIMAIN")

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGui_libraries
        EXPORT SofaGuiTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES ${HEADER_FILES} DESTINATION "include/sofa/gui" COMPONENT SofaGui_headers)
