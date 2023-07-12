cmake_minimum_required(VERSION 3.12)

project(Sofa.Core_test)

set(SOURCE_FILES
    accessor/ReadAccessor.cpp
    accessor/WriteAccessor.cpp
    collision/NarrowPhaseDetection_test.cpp
    loader/MeshLoader_test.cpp
    objectmodel/AspectPool_test.cpp
    objectmodel/BaseClass_test.cpp
    objectmodel/BaseData_test.cpp
    objectmodel/BaseLink_test.cpp
    objectmodel/BaseObjectDescription_test.cpp
    objectmodel/Data_test.cpp
    objectmodel/DataLink_test.cpp
    objectmodel/DataFileName_test.cpp
    objectmodel/DataFileNameVector_test.cpp
    objectmodel/DataCallback_test.cpp
    objectmodel/DDGNode_test.cpp
    objectmodel/MultiLink_test.cpp
    objectmodel/RemovedData_test.cpp
    objectmodel/SingleLink_test.cpp
    objectmodel/VectorData_test.cpp
    topology/BaseMeshTopology_test.cpp
    DataEngine_test.cpp
    Engine_test.cpp
    MatrixAccumulator_test.cpp
    ObjectFactory_test.cpp
    PathResolver_test.cpp
    TestEngine.cpp
    TrackedData_test.cpp
    VecId_test.cpp
    visual/DisplayFlags_test.cpp
)

set(HEADER_FILES
    objectmodel/BaseLink_test.h
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Core)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
