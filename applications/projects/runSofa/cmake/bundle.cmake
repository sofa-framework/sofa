
#--------------------------------------------------------------------------------
# Now the installation stuff below
#--------------------------------------------------------------------------------
SET(qtconf_dest_dir bin)
SET(qtplugins_dest_dir bin/plugins/imageformats)
SET(qtplatforms_dest_dir bin/plugins/platforms)
SET(APPS "\${CMAKE_INSTALL_PREFIX}/bin/${PROJECT_NAME}")

IF(APPLE)
    SET(qtconf_dest_dir runSofa.app/Contents/Resources)
    SET(qtplugins_dest_dir runSofa.app/Contents/plugins/imageformats)
    SET(qtplatforms_dest_dir runSofa.app/Contents/MacOS/platforms)
    SET(APPS "\${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}.app")
ENDIF()

#--------------------------------------------------------------------------------
# Install the  application, on Apple, the bundle is at the root of the
# install tree, and on other platforms it'll go into the bin directory.
INSTALL(TARGETS ${PROJECT_NAME} 
    BUNDLE DESTINATION . COMPONENT BundlePack
    RUNTIME DESTINATION bin COMPONENT BundlePack
    LIBRARY DESTINATION lib COMPONENT BundlePack)

## Install resource files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/share/ DESTINATION runSofa.app/Contents/MacOS/share/sofa COMPONENT BundlePack )
install(DIRECTORY ${CMAKE_SOURCE_DIR}/examples/ DESTINATION runSofa.app/Contents/MacOS/share/sofa/examples COMPONENT BundlePack )
install(FILES "${_defaultConfigPluginFilePath}" DESTINATION runSofa.app/Contents/MacOS/ COMPONENT BundlePack)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/applications/projects/runSofa/resources/ DESTINATION runSofa.app/Contents/MacOS/share/sofa/gui/runSofa/resources COMPONENT BundlePack)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/applications/sofa/gui/qt/resources/ DESTINATION runSofa.app/Contents/MacOS/share/sofa/gui/qt/resources COMPONENT BundlePack)
install(FILES "${CMAKE_BINARY_DIR}/etc/installedSofa.ini" DESTINATION runSofa.app/Contents/MacOS/etc RENAME sofa.ini COMPONENT BundlePack)

macro(sofa_set_python_bundle plugin_name directory)
    ## Install python scripts, preserving the file tree
    file(GLOB_RECURSE PYTHON_FILES "${directory}/*.py")
    file(GLOB_RECURSE JSON_FILES   "${directory}/*.json")
    LIST(APPEND ALL_FILES ${PYTHON_FILES} ${JSON_FILES})
    foreach(python_file ${ALL_FILES})
        file(RELATIVE_PATH script "${directory}" "${python_file}")
        get_filename_component(path ${script} DIRECTORY)
        install(FILES ${directory}/${script}
                DESTINATION "runSofa.app/Contents/MacOS/python2.7/site-packages/${path}"
                COMPONENT BundlePack)
    endforeach()

    ## Python configuration file (install tree)
     file(WRITE "${CMAKE_BINARY_DIR}/bundle-SofaPython-config"
         "python2.7/site-packages")
     install(FILES "${CMAKE_BINARY_DIR}/bundle-SofaPython-config"
             DESTINATION "runSofa.app/Contents/MacOS/etc/sofa/python.d"
             RENAME "${plugin_name}"
             COMPONENT BundlePack)
endmacro()
sofa_set_python_bundle(SofaPython ${CMAKE_SOURCE_DIR}/applications/plugins/SofaPython/python)

### TODO: split examples/resources between the ones in the package and the ones outside the package
install(DIRECTORY ${CMAKE_SOURCE_DIR}/share/ DESTINATION share/sofa COMPONENT BundlePack)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/examples/ DESTINATION share/sofa/examples COMPONENT BundlePack)

# Own way to get plugins dir
find_package(Qt5 COMPONENTS Core Gui Widgets) # to get SOFA_HAVE_GLUT
get_target_property(QtJpegLocation Qt5::QJpegPlugin LOCATION_release)
get_filename_component(QT_PLUGINS_IMAGES_DIR ${QtJpegLocation} DIRECTORY)
get_target_property(QtCocoaLocation Qt5::QCocoaIntegrationPlugin LOCATION_release)
get_filename_component(QT_PLUGINS_PLATFORMS_DIR ${QtCocoaLocation} DIRECTORY)

#--------------------------------------------------------------------------------
# Install needed Qt plugins by copying directories from the qt installation
# One can cull what gets copied by using 'REGEX "..." EXCLUDE'
INSTALL(DIRECTORY "${QT_PLUGINS_IMAGES_DIR}/" DESTINATION ${qtplugins_dest_dir} COMPONENT BundlePack)
INSTALL(DIRECTORY "${QT_PLUGINS_PLATFORMS_DIR}/" DESTINATION ${qtplatforms_dest_dir} COMPONENT BundlePack)
#--------------------------------------------------------------------------------
# install a qt.conf file
# this inserts some cmake code into the install script to write the file
INSTALL(CODE "
    file(WRITE \"\${CMAKE_INSTALL_PREFIX}/${qtconf_dest_dir}/qt.conf\" \"[Paths]\n Plugins = ../plugins \")
" COMPONENT BundlePack)


#--------------------------------------------------------------------------------
# Use BundleUtilities to get all other dependencies for the application to work.
# It takes a bundle or executable along with possible plugins and inspects it
# for dependencies.  If they are not system dependencies, they are copied.

# directories to look for dependencies
get_target_property(QtCoreLocation Qt5::Core LOCATION_release)
get_filename_component(QT_CORE_LIB_DIR ${QtCoreLocation} DIRECTORY)
get_filename_component(QT_LIB_DIR ${QT_CORE_LIB_DIR} DIRECTORY)
SET(DIRS ${QT_LIB_DIR}) # ${CMAKE_INSTALL_PREFIX}/lib

# Now the work of copying dependencies into the bundle/package
# The quotes are escaped and variables to use at install time have their $ escaped
# An alternative is the do a configure_file() on a script and use install(SCRIPT  ...).
# Note that the image plugins depend on QtSvg and QtXml, and it got those copied
# over.

INSTALL(CODE "
    file(GLOB_RECURSE LIBS
        \"${CMAKE_INSTALL_PREFIX}/${qtplugins_dest_dir}/*${CMAKE_SHARED_LIBRARY_SUFFIX}\"
        \"${CMAKE_INSTALL_PREFIX}/${qtplatforms_dest_dir}/*${CMAKE_SHARED_LIBRARY_SUFFIX}\")
    set(BU_CHMOD_BUNDLE_ITEMS ON)
    include(BundleUtilities)
    fixup_bundle(\"${APPS}\" \"\${LIBS}\" \"${DIRS}\")
" COMPONENT BundlePack)

install(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postInstall.cmake" COMPONENT BundlePack)

# To Create a package, one can run "cpack -G DragNDrop CPackConfig.cmake" on Mac OS X
# where CPackConfig.cmake is created by including CPack
# And then there's ways to customize this as well
SET(CPACK_PACKAGE_ICON ${CMAKE_CURRENT_SOURCE_DIR}/runSOFA.icns)
SET(CPACK_BINARY_DRAGNDROP ON)

# include(CPack)
