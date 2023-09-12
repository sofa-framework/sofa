cmake_minimum_required(VERSION 3.12)
project(Sofa.DefaultType LANGUAGES CXX)

set(SRC_ROOT "src/sofa/defaulttype")

set(HEADER_FILES
    ${SRC_ROOT}/AbstractTypeInfo.h
    ${SRC_ROOT}/DataTypeInfo.h
    ${SRC_ROOT}/DataTypeOperations.h
    ${SRC_ROOT}/MapMapSparseMatrix.h
    ${SRC_ROOT}/MapMapSparseMatrixEigenUtils.h
    ${SRC_ROOT}/MatrixExporter.h
    ${SRC_ROOT}/RigidCoord.h
    ${SRC_ROOT}/RigidDeriv.h
    ${SRC_ROOT}/RigidMass.h
    ${SRC_ROOT}/RigidTypes.h
    ${SRC_ROOT}/RigidVec6Types.h
    ${SRC_ROOT}/SolidTypes.h
    ${SRC_ROOT}/SolidTypes.inl
    ${SRC_ROOT}/TemplatesAliases.h
    ${SRC_ROOT}/TypeInfoID.h
    ${SRC_ROOT}/TypeInfoRegistry.h
    ${SRC_ROOT}/TypeInfoRegistryTools.h
    ${SRC_ROOT}/VecTypes.h
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/init.h
    ${SRC_ROOT}/fwd.h
    ${SRC_ROOT}/typeinfo/DataTypeInfoDynamicWrapper.h
    ${SRC_ROOT}/typeinfo/DataTypeInfo.h
    ${SRC_ROOT}/typeinfo/NoTypeInfo.h
    ${SRC_ROOT}/typeinfo/NameOnlyTypeInfo.h
    ${SRC_ROOT}/typeinfo/TypeInfo_Bool.h
    ${SRC_ROOT}/typeinfo/TypeInfo_BoundingBox.h
    ${SRC_ROOT}/typeinfo/TypeInfo_FixedArray.h
    ${SRC_ROOT}/typeinfo/TypeInfo_Integer.h
    ${SRC_ROOT}/typeinfo/TypeInfo_Mat.h
    ${SRC_ROOT}/typeinfo/TypeInfo_Quat.h
    ${SRC_ROOT}/typeinfo/TypeInfo_RigidTypes.h
    ${SRC_ROOT}/typeinfo/TypeInfo_RGBAColor.h
    ${SRC_ROOT}/typeinfo/TypeInfo_Set.h
    ${SRC_ROOT}/typeinfo/TypeInfo_Scalar.h
    ${SRC_ROOT}/typeinfo/TypeInfo_Text.h
    ${SRC_ROOT}/typeinfo/TypeInfo_Vector.h
    ${SRC_ROOT}/typeinfo/TypeInfo_Vec.h
    ${SRC_ROOT}/typeinfo/TypeInfo_VecTypes.h
    ${SRC_ROOT}/typeinfo/TypeInfo_Topology.h

    ${SRC_ROOT}/typeinfo/models/IncompleteTypeInfo.h
    ${SRC_ROOT}/typeinfo/models/BoolTypeInfo.h
    ${SRC_ROOT}/typeinfo/models/IntegerTypeInfo.h
    ${SRC_ROOT}/typeinfo/models/FixedArrayTypeInfo.h
    ${SRC_ROOT}/typeinfo/models/SetTypeInfo.h
    ${SRC_ROOT}/typeinfo/models/ScalarTypeInfo.h
    ${SRC_ROOT}/typeinfo/models/TextTypeInfo.h
    ${SRC_ROOT}/typeinfo/models/VectorTypeInfo.h
)

set(SOURCE_FILES
    ${SRC_ROOT}/AbstractTypeInfo.cpp
    ${SRC_ROOT}/MatrixExporter.cpp
    ${SRC_ROOT}/SolidTypes.cpp
    ${SRC_ROOT}/TemplatesAliases.cpp
    ${SRC_ROOT}/TypeInfoID.cpp
    ${SRC_ROOT}/TypeInfoRegistry.cpp
    ${SRC_ROOT}/TypeInfoRegistryTools.cpp
    ${SRC_ROOT}/init.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_Bool.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_BoundingBox.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_FixedArray.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_Integer.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_Mat.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_Quat.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_RGBAColor.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_RigidTypes.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_Set.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_Scalar.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_Text.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_Vector.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_Vec.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_VecTypes.cpp
    ${SRC_ROOT}/typeinfo/TypeInfo_Topology.cpp
)

sofa_find_package(Sofa.Helper REQUIRED)
sofa_find_package(Sofa.LinearAlgebra REQUIRED)
sofa_find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Helper Sofa.Type Sofa.LinearAlgebra)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_DEFAULTTYPE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_DEFAULTTYPE_BUILD_TESTS)
    add_subdirectory(test)
    add_subdirectory(simutest)
endif()
