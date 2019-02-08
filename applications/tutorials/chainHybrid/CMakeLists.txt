cmake_minimum_required(VERSION 3.1)
project(chainHybrid)

find_package(SceneCreator)
find_package(SofaGui)
find_package(SofaGeneral)
find_package(SofaCommon)
find_package(SofaBase)
find_package(SofaSimulation)
find_package(SofaFramework)

add_executable(${PROJECT_NAME} chainHybrid.cpp)
target_link_libraries(${PROJECT_NAME}
    SceneCreator
    SofaGuiMain SofaGuiCommon # SofaGui
    SofaGeneralDeformable SofaGeneralLoader # SofaGeneral
    SofaDeformable SofaSimpleFem # SofaCommon
    SofaBaseMechanics SofaBaseTopology # SofaBase
    SofaSimulationTree # SofaSimulation
    SofaCore SofaHelper SofaDefaultType # SofaFramework
    )
