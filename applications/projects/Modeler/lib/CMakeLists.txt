cmake_minimum_required(VERSION 2.8)

project("SofaModeler")

include(${SOFA_CMAKE_DIR}/pre.cmake)

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

if(EXTERNAL_USE_QT4)
    list(APPEND HEADER_FILES "QSofaTreeLibrary.h")
    list(APPEND HEADER_FILES "QCategoryTreeLibrary.h")
    list(APPEND HEADER_FILES "QComponentTreeLibrary.h")

    list(APPEND SOURCE_FILES "QSofaTreeLibrary.cpp")
    list(APPEND SOURCE_FILES "QCategoryTreeLibrary.cpp")
    list(APPEND SOURCE_FILES "QComponentTreeLibrary.cpp")
else()
    list(APPEND HEADER_FILES "QSofaLibrary.h")
    list(APPEND HEADER_FILES "QCategoryLibrary.h")
    list(APPEND HEADER_FILES "QComponentLibrary.h")

    list(APPEND SOURCE_FILES "QSofaLibrary.cpp")
    list(APPEND SOURCE_FILES "QCategoryLibrary.cpp")
    list(APPEND SOURCE_FILES "QComponentLibrary.cpp")
endif()

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaGuiMain SofaComponentMain )

include(${SOFA_CMAKE_DIR}/post.cmake)

