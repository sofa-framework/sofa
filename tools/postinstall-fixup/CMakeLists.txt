# Post-install scripts
if(NOT SOFA_BUILD_RELEASE_PACKAGE AND NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
    return()
endif()


if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    install(CODE "execute_process(COMMAND /bin/bash
        \"${CMAKE_CURRENT_LIST_DIR}/linux-postinstall-fixup.sh\"
        \"${CMAKE_CURRENT_LIST_DIR}\"
        \"${CMAKE_BINARY_DIR}\"
        \"\$ENV{DESTDIR\}\${CMAKE_INSTALL_PREFIX\}\"
        )"
        COMPONENT resources)
elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    if(SOFA_BUILD_APP_BUNDLE)
        install(CODE "execute_process(COMMAND /bin/bash
            \"${CMAKE_CURRENT_LIST_DIR}/macos-postinstall-fixup.sh\"
            \"${CMAKE_CURRENT_LIST_DIR}\"
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/../..\"
            )"
            COMPONENT resources)
    else()
        install(CODE "execute_process(COMMAND /bin/bash
            \"${CMAKE_CURRENT_LIST_DIR}/macos-postinstall-fixup.sh\"
            \"${CMAKE_CURRENT_LIST_DIR}\"
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\"
            )"
            COMPONENT resources)
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL Windows)
    install(CODE "execute_process(COMMAND bash
        \"${CMAKE_CURRENT_LIST_DIR}/windows-postinstall-fixup.sh\"
        \"${CMAKE_CURRENT_LIST_DIR}\"
        \"${CMAKE_BINARY_DIR}\"
        \"\$ENV{DESTDIR\}\${CMAKE_INSTALL_PREFIX\}\"
        )"
        COMPONENT resources)
endif()
