cmake_minimum_required(VERSION 2.8.12)

project(RigidScale_test)

set(HEADER_FILES

)

set(SOURCE_FILES
    RigidScaleToAffineMultiMapping_test.cpp
    #RigidScaleToRigidMultiMapping_test.cpp # this line is comment for the moment because this test does nit work yet.
)

set(PYTHON_FILES

)

if(WIN32)
    list(APPEND HEADER_FILES "stdafx.h")
    list(APPEND SOURCE_FILES "stdafx.cpp")
endif()

find_package(SofaPython QUIET)
find_package(SofaFlexible QUIET)
find_package(Compliant QUIET)

if(SofaPython_FOUND)
    list(APPEND SOURCE_FILES python_test_list.cpp)
    add_definitions("-DRIGIDSCALE_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes/python\"")
endif()

add_definitions("-DRIGIDSCALE_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})

target_link_libraries(${PROJECT_NAME} RigidScale SofaTest SofaGTestMain)

if(Flexible_FOUND)
    target_link_libraries(${PROJECT_NAME} Flexible)
endif()

if(Compliant_FOUND)
    target_link_libraries(${PROJECT_NAME} Compliant)
endif()

# #Needed by python test
# AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
