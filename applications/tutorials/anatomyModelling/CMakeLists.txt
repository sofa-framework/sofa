cmake_minimum_required(VERSION 3.1)
project(anatomyModelling)

find_package(SofaGui)
find_package(SceneCreator)
find_package(Flexible REQUIRED)
find_package(image REQUIRED)


set(SOFACOMMON_TARGETS
    SofaComponentCommon
    SofaGeneralLoader

)

include_directories("${SOFA_KERNEL_SOURCE_DIR}/../applications/plugins/image")
include_directories("${SOFA_KERNEL_SOURCE_DIR}/../applications/plugins/Flexible")


add_executable(${PROJECT_NAME} anatomyModelling.cpp)
#target_link_libraries(${PROJECT_NAME} Flexible image SofaGuiMain SceneCreator SofaCore SofaComponentBase SofaGeneralLoader SofaBaseVisual SofaOpenglVisual SofaBaseTopology SofaBaseMechanics SofaHelper)
target_link_libraries(${PROJECT_NAME} Flexible  SofaGuiMain SceneCreator SofaCore SofaComponentBase SofaGeneralLoader   )
