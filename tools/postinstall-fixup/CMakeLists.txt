# Post-install scripts
if(SOFA_BUILD_RELEASE_PACKAGE)
    if(CMAKE_SYSTEM_NAME STREQUAL Linux)
        install(CODE "file(WRITE
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/qt.conf\"
            \"[Paths]\\n  Prefix = ..\\n  LibraryExecutables = bin\"
            )"
            COMPONENT applications)
        install(CODE "execute_process(COMMAND /bin/bash
            \"${CMAKE_CURRENT_LIST_DIR}/linux-postinstall-fixup.sh\"
            \"${CMAKE_BINARY_DIR}\"
            \"\$ENV{DESTDIR\}\${CMAKE_INSTALL_PREFIX\}\"
            \"${Qt5_DIR}/../../..\"
            )"
            COMPONENT applications)
    elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
        if(SOFA_BUILD_APP_BUNDLE)
            include(macdeployqt) # set MACDEPLOYQT_EXECUTABLE
            install(CODE "execute_process(COMMAND /bin/bash
                \"${CMAKE_CURRENT_LIST_DIR}/macos-postinstall-fixup.sh\"
                \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/../..\"
                \"${Qt5_DIR}/../../..\"
                \"${MACDEPLOYQT_EXECUTABLE}\"
                )"
                COMPONENT applications)
        else()
            install(CODE "execute_process(COMMAND /bin/bash
                \"${CMAKE_CURRENT_LIST_DIR}/macos-postinstall-fixup.sh\"
                \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\"
                \"${Qt5_DIR}/../../..\"
                )"
                COMPONENT applications)
        endif()
    endif()
endif()
