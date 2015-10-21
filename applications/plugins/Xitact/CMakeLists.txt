cmake_minimum_required(VERSION 2.8)
project(Xitact)

set(XITACT_INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(XITACT_INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(XITACT_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    XiTrocarInterface.h
    initXitact.h
    PaceMaker.h
    IHPDriver.h
    ITPDriver.h)

set(SOURCE_FILES
    initXitact.cpp
    PaceMaker.cpp
    IHPDriver.cpp
    ITPDriver.cpp)

set(README_FILES
	PluginXitact.txt)

find_package(SofaGui REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_XITACTPLUGIN")
target_link_libraries(${PROJECT_NAME} SofaComponentMain SofaGui SofaGuiQt XiRobot)

install(TARGETS ${PROJECT_NAME}
    COMPONENT Xitact_libraries
    EXPORT XitactTargets
    RUNTIME DESTINATION "${XITACT_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${XITACT_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${XITACT_INSTALL_LIB_DIR}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
