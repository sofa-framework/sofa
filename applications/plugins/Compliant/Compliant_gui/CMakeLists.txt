cmake_minimum_required(VERSION 2.8.12)
project(Compliant_gui)

set(MOC_HEADER_FILES
    QCompliantMouseOperations.h
)

set(HEADER_FILES
    config.h
    CompliantAttachPerformer.h
    CompliantAttachPerformer.inl
)

set(SOURCE_FILES
    initCompliant_gui.cpp
    CompliantAttachPerformer.cpp
    QCompliantMouseOperations.cpp
)

find_package(SofaGui QUIET)

if(SofaGui_FOUND)
    qt4_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
    add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${MOC_FILES})
    add_definitions("-DSOFA_BUILD_COMPLIANT_GUI")
    target_link_libraries(${PROJECT_NAME} SofaGuiMain Compliant)
    set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
endif()
