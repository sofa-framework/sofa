cmake_minimum_required(VERSION 3.1)

project(Sofa.Component.Mass VERSION 1.0)

find_package(SofaFramework REQUIRED)

set(HEADER_FILES config/${PROJECT_NAME}.h)
set(SOURCE_FILES config/${PROJECT_NAME}.cpp)

list(APPEND HEADER_FILES
	config/Sofa.Component.Mass.h
    src/sofa/component/mass/BeamMassMatrix.h
    src/sofa/component/mass/BeamMassMatrix.inl
)

list(APPEND SOURCE_FILES
	config/Sofa.Component.Mass.cpp
    src/sofa/component/mass/BeamMassMatrix.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaHelper SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DBUILD_SOFA_MASS_API")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/config>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>")

sofa_create_package(${PROJECT_NAME} ${PROJECT_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})
