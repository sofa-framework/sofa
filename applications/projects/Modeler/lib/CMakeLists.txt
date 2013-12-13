include(${SOFA_CMAKE_DIR}/preProject.cmake)

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
	
set(UI_FILES

	Modeler.ui
	DialogAddPreset.ui
	../../../sofa/gui/qt/PluginManager.ui
	)

UseQt()

list(APPEND HEADER_FILES "QSofaTreeLibrary.h")
list(APPEND HEADER_FILES "QCategoryTreeLibrary.h")
list(APPEND HEADER_FILES "QComponentTreeLibrary.h")

list(APPEND SOURCE_FILES "QSofaTreeLibrary.cpp")
list(APPEND SOURCE_FILES "QCategoryTreeLibrary.cpp")
list(APPEND SOURCE_FILES "QComponentTreeLibrary.cpp")

SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})
SOFA_QT4_WRAP_UI(FORM_FILES ${UI_FILES})
QT4_ADD_RESOURCES(RESOURCE_FILES "${SOFA_SHARE_DIR}/textures/modeler/icons.qrc")

# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")
ToRelativePath(FORM_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${FORM_FILES}")
ToRelativePath(RESOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${RESOURCE_FILES}")

set(MOC_FILES_GROUP "Generated")
set(FORM_FILES_GROUP "Generated")
set(RESOURCE_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${MOC_FILES} ${FORM_FILES} ${RESOURCE_FILES} ${SOURCE_FILES})

AddLinkerDependencies(SofaGuiMain SofaComponentMain Qwt SofaGuiQt)

# SofaGuiCommon Qwt SofaBaseVisual SofaExporter SofaLoader SofaOpenglVisual SofaMiscForceField

include(${SOFA_CMAKE_DIR}/postProject.cmake)

