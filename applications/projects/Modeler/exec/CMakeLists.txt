cmake_minimum_required(VERSION 2.8)
project(Modeler)

set(SOURCE_FILES

    Main.cpp
    )
    
#UseQt()

if(APPLE)

    set(RC_FILES "../Modeler.icns")
    
else()
    
    set(RC_FILES "sofa.rc")

endif()

#temporarily set SOFA_SHARE_DIR
set(SOFA_SHARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../share")

QT4_ADD_RESOURCES(RESOURCE_FILES "${SOFA_SHARE_DIR}/textures/modeler/icons.qrc")

# make resource files path from absolute to relative
#ToRelativePath(RESOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${RESOURCE_FILES}")

set(RESOURCE_FILES_GROUP "Generated")

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES} ${RESOURCE_FILES})

# since "SofaComponentMain" does not exist any more we link to every components collection
# this is temporary, until every module is switched to plugins
target_link_libraries( ${PROJECT_NAME} SofaModeler SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc ) 

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGUI_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

# make the name of debug applications end in _d.
set_target_properties( ${PROJECT_NAME}  PROPERTIES DEBUG_POSTFIX "_d" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGui_applications
        EXPORT  SofaGuiTargets
        RUNTIME DESTINATION "${SOFAGUI_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGUI_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGUI_INSTALL_LIB_DIR}"
        )
        
# note: pas de headers à installer ici