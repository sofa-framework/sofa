cmake_minimum_required(VERSION 3.12)
project(SofaModeler VERSION 0.1)

find_package(TinyXML2 REQUIRED)

set(HEADER_FILES
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

find_package(Qt5 COMPONENTS Core Gui OpenGL Widgets REQUIRED)

qt5_wrap_cpp(MOC_FILES ${HEADER_FILES})
set(MOC_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGui SofaBase SofaCommon SofaGeneral SofaMisc)
target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::Widgets)
target_link_libraries(${PROJECT_NAME} PUBLIC tinyxml2::tinyxml2) # Public because needed by Modeler (exec)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "."
    INCLUDE_INSTALL_DIR "SofaModeler"
    RELOCATABLE "projects"
    )
