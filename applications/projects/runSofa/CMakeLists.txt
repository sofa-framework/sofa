cmake_minimum_required(VERSION 2.8.12)
project(runSofa)

set(SOURCE_FILES
	Main.cpp)

if(APPLE)
    set(RC_FILES "runSOFA.icns")
else()
    set(RC_FILES "sofa.rc")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})
target_link_libraries(${PROJECT_NAME} SofaGuiMain SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGui_applications
        EXPORT SofaGuiTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
