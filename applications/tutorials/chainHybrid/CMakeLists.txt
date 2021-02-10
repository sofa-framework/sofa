cmake_minimum_required(VERSION 3.12)
project(chainHybrid)

find_package(SceneCreator REQUIRED)
find_package(SofaGui REQUIRED)
find_package(SofaGeneral REQUIRED)
find_package(SofaCommon REQUIRED)
find_package(SofaBase REQUIRED)
find_package(SofaSimulation REQUIRED)
find_package(SofaFramework REQUIRED)

add_executable(${PROJECT_NAME} chainHybrid.cpp)
target_link_libraries(${PROJECT_NAME}
    SceneCreator
    SofaGui
    SofaGeneralDeformable SofaGeneralLoader # SofaGeneral
    SofaDeformable SofaSimpleFem # SofaCommon
    SofaBaseMechanics SofaBaseTopology # SofaBase
    SofaSimulationGraph # SofaSimulation
    SofaCore SofaHelper SofaDefaultType # SofaFramework
    )
