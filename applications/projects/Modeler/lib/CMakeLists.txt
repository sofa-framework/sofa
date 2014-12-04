cmake_minimum_required(VERSION 2.8)
project(SofaModeler)

set(HEADER_FILES

    SofaModeler.h 
    GraphModeler.h 
    SofaTutorialManager.h 
    TutorialSelector.h 
    AddPreset.h 
    FilterLibrary.h 
    GlobalModification.h 
	LinkComponent.h 
    GraphHistoryManager.h 
    ModifierCondition.h 
    ../../../sofa/gui/qt/SofaPluginManager.h 

    )

set(SOURCE_FILES

    SofaModeler.cpp 
    GraphModeler.cpp 
	LinkComponent.cpp 
    SofaTutorialManager.cpp 
    TutorialSelector.cpp 
    AddPreset.cpp 
    FilterLibrary.cpp 
    GlobalModification.cpp 
    GraphHistoryManager.cpp 
    ModifierCondition.cpp 
    ../../../sofa/gui/qt/SofaPluginManager.cpp 

    )
	
set(UI_FILES ../../../sofa/gui/qt/PluginManager.ui)

#UseQt()

list(APPEND HEADER_FILES "QSofaTreeLibrary.h")
list(APPEND HEADER_FILES "QCategoryTreeLibrary.h")
list(APPEND HEADER_FILES "QComponentTreeLibrary.h")

list(APPEND SOURCE_FILES "QSofaTreeLibrary.cpp")
list(APPEND SOURCE_FILES "QCategoryTreeLibrary.cpp")
list(APPEND SOURCE_FILES "QComponentTreeLibrary.cpp")

QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})
QT4_WRAP_UI(FORM_FILES ${UI_FILES})

# make mocced header / form files path from absolute to relative
#ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")
#ToRelativePath(FORM_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${FORM_FILES}")

set(MOC_FILES_GROUP "Generated")
set(FORM_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${MOC_FILES} ${FORM_FILES} ${SOURCE_FILES})

# since "SofaComponentMain" does not exist any more we link to every components collection
# this is temporary, until every module is switched to plugins
target_link_libraries( ${PROJECT_NAME} SofaGuiMain SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc ) 
# SofaGuiCommon Qwt SofaBaseVisual SofaExporter SofaLoader SofaOpenglVisual SofaMiscForceField

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFABASE_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGui_libraries
        EXPORT  SofaGuiTargets
        RUNTIME DESTINATION "${SOFABASE_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFABASE_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFABASE_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFABASE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
        )


