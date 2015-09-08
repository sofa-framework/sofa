cmake_minimum_required(VERSION 2.8)
project(SofaModeler)

set(HEADER_FILES

    ../../../sofa/gui/qt/SofaPluginManager.h
    AddPreset.h
    FilterLibrary.h
    GlobalModification.h
    GraphHistoryManager.h
    GraphModeler.h
    LinkComponent.h
    ModifierCondition.h
    QCategoryTreeLibrary.h
    QComponentTreeLibrary.h
    QSofaTreeLibrary.h
    SofaModeler.h
    SofaTutorialManager.h
    TutorialSelector.h
)

set(SOURCE_FILES

    ../../../sofa/gui/qt/SofaPluginManager.cpp
    AddPreset.cpp
    FilterLibrary.cpp
    GlobalModification.cpp
    GraphHistoryManager.cpp
    GraphModeler.cpp
    LinkComponent.cpp
    ModifierCondition.cpp
    QCategoryTreeLibrary.cpp
    QComponentTreeLibrary.cpp
    QSofaTreeLibrary.cpp
    SofaModeler.cpp
    SofaTutorialManager.cpp
    TutorialSelector.cpp
)

set(UI_FILES ../../../sofa/gui/qt/PluginManager.ui)

qt4_wrap_cpp(MOC_FILES ${HEADER_FILES})
set(MOC_FILES_GROUP "Generated")

qt4_wrap_ui(FORM_FILES ${UI_FILES})
set(FORM_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${MOC_FILES} ${FORM_FILES} ${SOURCE_FILES})

# since "SofaComponentMain" does not exist any more we link to every components collection
# this is temporary, until every module is switched to plugins
target_link_libraries(${PROJECT_NAME} SofaGuiMain SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)
# SofaGuiCommon Qwt SofaBaseVisual SofaExporter SofaLoader SofaOpenglVisual SofaMiscForceField

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFABASE_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGui_libraries
        EXPORT SofaGuiTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}")
