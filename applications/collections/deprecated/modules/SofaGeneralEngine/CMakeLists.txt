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

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Engine.Analyze, Sofa.Component.Engine.Generate, Sofa.Component.Engine.Select and Sofa.Component.Engine.Transform instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Engine.Analyze REQUIRED)
sofa_find_package(Sofa.Component.Engine.Generate REQUIRED)
sofa_find_package(Sofa.Component.Engine.Select REQUIRED)
sofa_find_package(Sofa.Component.Engine.Transform REQUIRED)

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Analyze)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Generate)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Select)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Transform)

sofa_find_package(Sofa.GL.Component.Engine QUIET)
if(Sofa.GL.Component.Engine_FOUND)
#    message(WARNING "Moreover, use Sofa.GL.Component.Engine if you need TextureInterpolation.")
    target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL.Component.Engine) 
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
