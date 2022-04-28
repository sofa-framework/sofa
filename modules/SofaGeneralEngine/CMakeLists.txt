cmake_minimum_required(VERSION 3.12)
project(SofaGeneralEngine LANGUAGES CXX)


set(SOFAGENERALENGINE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/config.h.in
    ${SOFAGENERALENGINE_SRC}/initSofaGeneralEngine.h
    )

set(SOURCE_FILES
    ${SOFAGENERALENGINE_SRC}/initSofaGeneralEngine.cpp
    )


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Engine.Analyze, Sofa.Component.Engine.Generate, Sofa.Component.Engine.Select and Sofa.Component.Engine.Transform instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Engine.Analyze QUIET)
sofa_find_package(Sofa.Component.Engine.Generate QUIET)
sofa_find_package(Sofa.Component.Engine.Select QUIET)
sofa_find_package(Sofa.Component.Engine.Transform QUIET)
sofa_find_package(Sofa.GL QUIET)

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Analyze)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Generate)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Select)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Transform)


if(Sofa.GL_FOUND)
    message(WARNING "Moreover, use Sofa.GL.Component.Engine if you need TextureInterpolation.")
    sofa_find_package(Sofa.GL.Component.Engine QUIET)
    if(Sofa.GL.Component.Engine_FOUND)
        target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL.Component.Engine) 
    endif()
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAGENERALENGINE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAGENERALENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()

