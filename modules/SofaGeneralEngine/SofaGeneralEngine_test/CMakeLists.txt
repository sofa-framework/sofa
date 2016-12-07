cmake_minimum_required(VERSION 3.1)

project(SofaGeneralEngine_test)


set(HEADER_FILES)

set(SOURCE_FILES

        AverageCoord_test.cpp
        ClusteringEngine_test.cpp
        ComplementaryROI_test.cpp
        DifferenceEngine_test.cpp
        DilateEngine_test.cpp
        ExtrudeEdgesAndGenerateQuads_test.cpp
        MeshROI_test.cpp
        SphereROI_test.cpp
        PlaneROI_test.cpp
        TransformEngine_test.cpp

        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
