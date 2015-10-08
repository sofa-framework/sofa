cmake_minimum_required(VERSION 2.8.12)
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

find_package(Qt5 COMPONENTS Core QUIET)

if(Qt5Core_FOUND)
    message("Modeler: Using Qt5")
    find_package(Qt5 COMPONENTS Gui OpenGL REQUIRED)

    set(EXT_QT_INCLUDES ${Qt5Core_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS} ${Qt5Xml_INCLUDE_DIRS})
    set(EXT_QT_LIBRARIES ${Qt5Core_LIBRARIES} ${Qt5Widgets_LIBRARIES} ${Qt5OpenGL_LIBRARIES})

    qt5_wrap_cpp(MOC_FILES ${HEADER_FILES})
    set(MOC_FILES_GROUP "Generated")

    qt5_wrap_ui(FORM_FILES ${UI_FILES})
    set(FORM_FILES_GROUP "Generated")

else()
    message("Modeler: Using Qt4")

    find_package(Qt4 COMPONENTS qtcore qtgui qtopengl qtxml)
    set (EXT_QT_INCLUDES ${QT_INCLUDES})
    set(EXT_QT_LIBRARIES ${QT_LIBRARIES})

    qt4_wrap_cpp(MOC_FILES ${HEADER_FILES})
    set(MOC_FILES_GROUP "Generated")

    qt4_wrap_ui(FORM_FILES ${UI_FILES})
    set(FORM_FILES_GROUP "Generated")
endif()

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${MOC_FILES} ${FORM_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGuiMain SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc ${EXT_QT_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${EXT_QT_INCLUDES}>")

sofa_install_targets(SofaGui SofaModeler "SofaModeler")
