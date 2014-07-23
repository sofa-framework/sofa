cmake_minimum_required(VERSION 2.8)
project(runSofa)

set(SOURCE_FILES

	Main.cpp
	)

#UseQt()

if(APPLE)
    set(RC_FILES "runSOFA.icns")
else()
    set(RC_FILES "sofa.rc")
endif()
	
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

# since "SofaComponentMain" does not exist any more we link to every components collection
# this is temporary, until every module is switched to plugins
target_link_libraries( ${PROJECT_NAME} SofaGuiMain SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc ) 

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGUI_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGui_libraries
        EXPORT  SofaGuiTargets
        RUNTIME DESTINATION "${SOFAGUI_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGUI_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGUI_INSTALL_LIB_DIR}"
        )
        
# note: pas de headers à installer ici