cmake_minimum_required(VERSION 3.12)
project(chainHybrid)

sofa_find_package(SceneCreator REQUIRED)
sofa_find_package(SofaGui REQUIRED)
sofa_find_package(SofaGeneralDeformable REQUIRED)
sofa_find_package(SofaGeneralLoader REQUIRED)
sofa_find_package(SofaDeformable REQUIRED)
sofa_find_package(SofaSimpleFem REQUIRED)
sofa_find_package(SofaBaseMechanics REQUIRED)
sofa_find_package(SofaBaseTopology REQUIRED)
sofa_find_package(SofaSimulationGraph REQUIRED)
sofa_find_package(SofaFramework REQUIRED)

add_executable(${PROJECT_NAME} chainHybrid.cpp)
target_link_libraries(${PROJECT_NAME}
    SceneCreator
    SofaGui
    SofaGeneralDeformable SofaGeneralLoader
    SofaDeformable SofaSimpleFem
    SofaBaseMechanics SofaBaseTopology
    SofaSimulationGraph
    SofaCore SofaHelper SofaDefaultType # SofaFramework
    )
