include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
find_package( SofaQtOgreViewerLibs REQUIRED)
include_directories(${SOFAQTOGREVIEWER_INCLUDE_DIRS})

set(HEADER_FILES

    DotSceneLoader.h 
    HelperLogics.h 
    QtOgreViewer.h 
    OgreVisualModel.h 
    OgreMeshLoader.h 
    OgreShaderEntryPoint.h 
    OgreShaderParameter.h 
    OgreShaderTextureUnit.h 
    QOgreLightWidget.h 
    OgrePlanarReflectionMaterial.h 
    OgreSofaViewer.h 
    DrawToolOGRE.h 
    OgreViewerSetting.h 
    OgreSceneObject.h 
    PropagateOgreSceneManager.h 
    initQtOgreViewer.h 
    SubMesh.h

	)

set(SOURCE_FILES

    DotSceneLoader.cpp 
    HelperLogics.cpp 
    QtOgreViewer.cpp
    QtOgreViewer_slots.cpp
    OgreVisualModel.cpp 
    OgreMeshLoader.cpp 
    OgreShaderParameter.cpp 
    OgreShaderTextureUnit.cpp 
    QOgreLightWidget.cpp 
    OgrePlanarReflectionMaterial.cpp
    DrawToolOGRE.cpp 
    OgreViewerSetting.cpp 
    OgreSceneObject.cpp 
    PropagateOgreSceneManager.cpp 
    initQtOgreViewer.cpp 
    SubMesh.cpp 

	)

set(README_FILES

	QtOgreViewer.txt
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_QTOGREVIEWERPLUGIN SOFA_GUI_QT")
AddLinkerDependencies()

if(APPLE)
    AddLinkerDependencies("Ogre")
    AddLinkerDependencies("CoreFoundation")
endif()

if(WIN32)
    AddLinkerDependencies("OgreMain")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)