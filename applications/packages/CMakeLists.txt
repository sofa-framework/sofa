cmake_minimum_required(VERSION 3.1)

foreach(TARGET SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentMisc SofaComponentAdvanced)
    add_subdirectory(${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/SofaKernel/modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
    # set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    # set_target_properties(${TARGET} PROPERTIES VERSION  ${SofaFramework_VERSION})
endforeach()

sofa_add_plugin(SofaComponentAll SofaComponentAll ON)


