# Post-install scripts
if(NOT SOFA_BUILD_RELEASE_PACKAGE)
    return()
endif()

find_package(Qt5 COMPONENTS Gui REQUIRED)
get_target_property(qt5gui_loc Qt5::Gui LOCATION_RELEASE)
get_filename_component(QT_LIB_DIR "${qt5gui_loc}" DIRECTORY)
if(QT_LIB_DIR MATCHES "\.framework$")
    get_filename_component(QT_LIB_DIR "${QT_LIB_DIR}" DIRECTORY)
endif()
get_target_property(qjpeg_loc Qt5::QJpegPlugin LOCATION_RELEASE)
get_filename_component(imageformats_dir "${qjpeg_loc}" DIRECTORY)
get_filename_component(plugins_dir "${imageformats_dir}" DIRECTORY)
get_filename_component(QT_DATA_DIR "${plugins_dir}" DIRECTORY)

message("QT_LIB_DIR = ${QT_LIB_DIR}")
message("QT_DATA_DIR = ${QT_DATA_DIR}")

if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    install(CODE "file(WRITE
        \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/qt.conf\"
        \"[Paths]\\n  Prefix = ..\\n  Plugins = bin\\n  LibraryExecutables = bin\"
        )"
        COMPONENT resources)
    install(CODE "execute_process(COMMAND /bin/bash
        \"${CMAKE_CURRENT_LIST_DIR}/linux-postinstall-fixup.sh\"
        \"${CMAKE_BINARY_DIR}\"
        \"\$ENV{DESTDIR\}\${CMAKE_INSTALL_PREFIX\}\"
        \"${QT_LIB_DIR}\"
        \"${QT_DATA_DIR}\"
        )"
        COMPONENT resources)
elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    if(SOFA_BUILD_APP_BUNDLE)
        include(macdeployqt) # set MACDEPLOYQT_EXECUTABLE
        install(CODE "execute_process(COMMAND /bin/bash
            \"${CMAKE_CURRENT_LIST_DIR}/macos-postinstall-fixup.sh\"
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/../..\"
            \"${QT_LIB_DIR}\"
            \"${QT_DATA_DIR}\"
            \"${MACDEPLOYQT_EXECUTABLE}\"
            )"
            COMPONENT resources)
    else()
        install(CODE "execute_process(COMMAND /bin/bash
            \"${CMAKE_CURRENT_LIST_DIR}/macos-postinstall-fixup.sh\"
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\"
            \"${QT_LIB_DIR}\"
            \"${QT_DATA_DIR}\"
            )"
            COMPONENT resources)
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL Windows)
    install(CODE "file(WRITE
        \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/qt.conf\"
        \"[Paths]\\n  Prefix = ..\\n  Plugins = bin\\n  LibraryExecutables = bin\"
        )"
        COMPONENT resources)
    install(CODE "execute_process(COMMAND bash
        \"${CMAKE_CURRENT_LIST_DIR}/windows-postinstall-fixup.sh\"
        \"${CMAKE_BINARY_DIR}\"
        \"\$ENV{DESTDIR\}\${CMAKE_INSTALL_PREFIX\}\"
        )"
        COMPONENT resources)
endif()
