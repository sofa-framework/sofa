
#--------------------------------------------------------------------------------
# Now the installation stuff below
#--------------------------------------------------------------------------------
SET(qtconf_dest_dir bin)
SET(qtplugins_dest_dir bin/plugins)
SET(qtplatforms_dest_dir bin/plugins/platforms)
SET(APPS "\${CMAKE_INSTALL_PREFIX}/bin/${PROJECT_NAME}")

IF(APPLE)
    SET(qtconf_dest_dir runSofa.app/Contents/Resources)
    SET(qtplugins_dest_dir runSofa.app/Contents/plugins)
    SET(qtplatforms_dest_dir runSofa.app/Contents/MacOS)
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
install(FILES "${CMAKE_BINARY_DIR}/etc/installedSofa.ini" DESTINATION runSofa.app/Contents/MacOS/etc RENAME sofa.ini COMPONENT BundlePack)

### TODO: split examples/resources between the ones in the package and the ones outside the package
install(DIRECTORY ${CMAKE_SOURCE_DIR}/share/ DESTINATION share/sofa COMPONENT BundlePack)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/examples/ DESTINATION share/sofa/examples COMPONENT BundlePack)

# Own way to get plugins dir
find_package(Qt5 COMPONENTS Core Gui Widgets) # to get SOFA_HAVE_GLUT
get_target_property(QtJpegLocation Qt5::QJpegPlugin LOCATION_release)
get_filename_component(QT_PLUGINS_IMAGES_DIR ${QtJpegLocation} DIRECTORY)
get_target_property(QtCocoaLocation Qt5::QCocoaIntegrationPlugin LOCATION_release)
get_filename_component(QT_PLUGINS_PLATFORMS_DIR ${QtCocoaLocation} DIRECTORY)

#set(QT_PLUGINS_DIR /Volumes/Data/Dependencies/Qt/5.5/clang_64/plugins)

#--------------------------------------------------------------------------------
# Install needed Qt plugins by copying directories from the qt installation
# One can cull what gets copied by using 'REGEX "..." EXCLUDE'
INSTALL(DIRECTORY "${QT_PLUGINS_IMAGES_DIR}" DESTINATION ${qtplugins_dest_dir} COMPONENT BundlePack)
INSTALL(DIRECTORY "${QT_PLUGINS_PLATFORMS_DIR}" DESTINATION ${qtplatforms_dest_dir} COMPONENT BundlePack)

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
set(QT_LIB_DIR_TEMP ${QT_CORE_LIB_DIR}/..)
SET(DIRS ${QT_LIB_DIR_TEMP} ${CMAKE_INSTALL_PREFIX}/lib)

# Now the work of copying dependencies into the bundle/package
# The quotes are escaped and variables to use at install time have their $ escaped
# An alternative is the do a configure_file() on a script and use install(SCRIPT  ...).
# Note that the image plugins depend on QtSvg and QtXml, and it got those copied
# over.

INSTALL(CODE "
    file(GLOB_RECURSE QTPLUGINS_QTPLATFORMS
        \"\${CMAKE_INSTALL_PREFIX}/${qtplugins_dest_dir}/*${CMAKE_SHARED_LIBRARY_SUFFIX}\"
        \"\${CMAKE_INSTALL_PREFIX}/${qtplatforms_dest_dir}/*${CMAKE_SHARED_LIBRARY_SUFFIX}\")
    set(BU_CHMOD_BUNDLE_ITEMS ON)
    include(BundleUtilities)
    fixup_bundle(\"${APPS}\" \"\${QTPLUGINS_QTPLATFORMS}\" \"${DIRS}\")
" COMPONENT BundlePack)

# To Create a package, one can run "cpack -G DragNDrop CPackConfig.cmake" on Mac OS X
# where CPackConfig.cmake is created by including CPack
# And then there's ways to customize this as well
SET(CPACK_PACKAGE_ICON ${CMAKE_CURRENT_SOURCE_DIR}/runSOFA.icns)
SET(CPACK_BINARY_DRAGNDROP ON)

# include(CPack)
