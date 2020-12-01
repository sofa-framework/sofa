cmake_minimum_required(VERSION 3.12)
project(SofaDefaultType)

set(SRC_ROOT "src/sofa/defaulttype")

set(HEADER_FILES
    ${SRC_ROOT}/AbstractTypeInfo.h
    ${SRC_ROOT}/BaseMatrix.h
    ${SRC_ROOT}/BaseVector.h
    ${SRC_ROOT}/BoundingBox.h
    ${SRC_ROOT}/DataTypeInfo.h
    ${SRC_ROOT}/Frame.h
    ${SRC_ROOT}/MapMapSparseMatrix.h
    ${SRC_ROOT}/MapMapSparseMatrixEigenUtils.h
    ${SRC_ROOT}/Mat.h
    ${SRC_ROOT}/MatSym.h
    ${SRC_ROOT}/Mat_solve_Cholesky.h
    ${SRC_ROOT}/Mat_solve_LU.h
    ${SRC_ROOT}/Mat_solve_SVD.h
    ${SRC_ROOT}/Quat.h
    ${SRC_ROOT}/Quat.inl
    ${SRC_ROOT}/RGBAColor.h
    ${SRC_ROOT}/RigidTypes.h
    ${SRC_ROOT}/RigidVec6Types.h
    ${SRC_ROOT}/SolidTypes.h
    ${SRC_ROOT}/SolidTypes.inl
    ${SRC_ROOT}/TemplatesAliases.h
    ${SRC_ROOT}/TopologyTypes.h
    ${SRC_ROOT}/TypeInfoID.h
    ${SRC_ROOT}/TypeInfoRegistry.h
    ${SRC_ROOT}/TypeInfoRegistryTools.h
    ${SRC_ROOT}/TopologyTypes.h
    ${SRC_ROOT}/Vec.h
    ${SRC_ROOT}/Ray.h
    ${SRC_ROOT}/Vec3Types.h
    ${SRC_ROOT}/VecTypes.h
    ${SRC_ROOT}/defaulttype.h
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/init.h
    ${SRC_ROOT}/Color.h

    ${SRC_ROOT}/typeinfo/DataTypeInfoDynamicWrapper.h
    ${SRC_ROOT}/typeinfo/DataTypeInfo.h
    ${SRC_ROOT}/typeinfo/NoTypeInfo.h
    ${SRC_ROOT}/typeinfo/NameOnlyTypeInfo.h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[bool].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[BoundingBox].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[fixed_array].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[Integer].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[Mat].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[Quat].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[RigidTypes].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[RGBAColor].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[set].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[Scalar].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[string].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[vector].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[Vec].h
    ${SRC_ROOT}/typeinfo/DataTypeInfo[VecTypes].h

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
    ${SRC_ROOT}/BaseMatrix.cpp
    ${SRC_ROOT}/BoundingBox.cpp
    ${SRC_ROOT}/Frame.cpp
    ${SRC_ROOT}/SolidTypes.cpp
    ${SRC_ROOT}/TemplatesAliases.cpp
    ${SRC_ROOT}/TypeInfoID.cpp
    ${SRC_ROOT}/TypeInfoRegistry.cpp
    ${SRC_ROOT}/TypeInfoRegistryTools.cpp
    ${SRC_ROOT}/init.cpp

    ${SRC_ROOT}/typeinfo/DataTypeInfo[bool].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[BoundingBox].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[fixed_array].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[Integer].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[Mat].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[Quat].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[RGBAColor].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[RigidTypes].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[set].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[Scalar].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[string].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[vector].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[Vec].cpp
    ${SRC_ROOT}/typeinfo/DataTypeInfo[VecTypes].cpp
)

find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_DEFAULTTYPE")

sofa_add_targets_to_package(
    PACKAGE_NAME SofaFramework
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaFramework"
    )

if(SOFA_BUILD_TESTS)
    add_subdirectory(SofaDefaultType_test)
endif()
