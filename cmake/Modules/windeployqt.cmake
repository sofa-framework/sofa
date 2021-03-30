# The MIT License (MIT)
#
# Copyright (c) 2017 Nathan Osman
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

find_package(Qt5Core REQUIRED)

# Retrieve the absolute path to qmake and then use that path to find
# the windeployqt binary
get_target_property(_qmake_executable Qt5::qmake IMPORTED_LOCATION)
get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)
find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${_qt_bin_dir}")

# Running this with MSVC 2015 requires CMake 3.6+
if((MSVC_VERSION VERSION_EQUAL 1900 OR MSVC_VERSION VERSION_GREATER 1900)
        AND CMAKE_VERSION VERSION_LESS "3.6")
    message(WARNING "Deploying with MSVC 2015+ requires CMake 3.6+")
endif()

# Add commands that copy the Qt runtime to the target's output directory after
# build and install the Qt runtime to the specified directory
function(windeployqt target build_dir install_dir)

    # execute windeployqt in a tmp directory after build
    add_custom_command(TARGET ${target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_BINARY_DIR}/windeployqt"
        COMMAND set PATH="${_qt_bin_dir}"
        COMMAND "${WINDEPLOYQT_EXECUTABLE}" --dir "${CMAKE_CURRENT_BINARY_DIR}/windeployqt" --verbose 0 --no-compiler-runtime --no-translations --no-angle --release --no-opengl-sw "$<TARGET_FILE:${target}>"
        COMMAND if exist "${build_dir}/$<CONFIG>/" (${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_BINARY_DIR}/windeployqt" "${build_dir}/$<CONFIG>") else (${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_BINARY_DIR}/windeployqt" "${build_dir}")
        )

    # copy deployment directory during installation
    if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (MSVC)
        foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
            install(
                DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/windeployqt/"
                DESTINATION bin/${CONFIG}
                OPTIONAL
                COMPONENT applications
                PATTERN "resources" EXCLUDE
                PATTERN "translations" EXCLUDE
                )
        endforeach()
    else()
        install(
            DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/windeployqt/"
            DESTINATION bin
            COMPONENT applications
            PATTERN "resources" EXCLUDE
            PATTERN "translations" EXCLUDE
            )
    endif()
    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/windeployqt/resources/"
        DESTINATION resources
        OPTIONAL
        COMPONENT applications
        )
    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/windeployqt/translations/"
        DESTINATION translations
        OPTIONAL
        COMPONENT applications
        )

    # windeployqt doesn't work correctly with the system runtime libraries,
    # so we fall back to one of CMake's own modules for copying them over
    set(CMAKE_INSTALL_UCRT_LIBRARIES TRUE)

    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)

    include(InstallRequiredSystemLibraries)

    if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (MSVC)
        foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
            install(
                PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
                DESTINATION bin/${CONFIG}
                OPTIONAL
                COMPONENT applications
            )
        endforeach()
    else()
        install(
            PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
            DESTINATION bin
            COMPONENT applications
        )
    endif()

    # foreach(lib ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
    #     get_filename_component(filename "${lib}" NAME)
    #     add_custom_command(TARGET ${target} POST_BUILD
    #         COMMAND "${CMAKE_COMMAND}" -E
    #             copy_if_different "${lib}" \"$<TARGET_FILE_DIR:${target}>\"
    #     )
    # endforeach()

    # install DLL dependencies
    # install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Release/" DESTINATION ${install_dir})

endfunction()

mark_as_advanced(WINDEPLOYQT_EXECUTABLE)
