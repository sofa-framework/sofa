include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    Main.cpp
    )
    
UseQt()
if(APPLE)

    set(RC_FILES "../Modeler.icns")
    
else()
    
    set(RC_FILES "sofa.rc")

endif()

UseQt()

QT4_ADD_RESOURCES(RESOURCE_FILES "${SOFA_SHARE_DIR}/textures/modeler/icons.qrc")

# make resource files path from absolute to relative
ToRelativePath(RESOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${RESOURCE_FILES}")

set(RESOURCE_FILES_GROUP "Generated")

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES} ${RESOURCE_FILES})

AddLinkerDependencies(SofaModeler SofaComponentMain tinyxml)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

