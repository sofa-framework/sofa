cmake_minimum_required(VERSION 3.1)

project(SofaExporter_test)


set(SOURCE_FILES ../../empty.cpp)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND SOURCE_FILES
        OBJExporter_test.cpp
        STLExporter_test.cpp
        MeshExporter_test.cpp)
    message(STATUS "SofaExporter_test: build all maintained sofa components.")
else()
    message(STATUS "SofaExporter_test: build with only the minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "SofaExporter_test: build with deprecated components.")
else()
    message(STATUS "SofaExporter_test: build without deprecated components.")
endif()


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
