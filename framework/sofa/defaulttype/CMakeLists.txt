include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    BaseMatrix.h
    BaseVector.h
    BoundingBox.h
    DataTypeInfo.h
    Frame.h
    LaparoscopicRigidTypes.h
    MapMapSparseMatrix.h
    Mat.h
    MatSym.h
    Mat_solve_Cholesky.h
    Mat_solve_LU.h
    Mat_solve_SVD.h
    Quat.h
    Quat.inl
    # RigidInertia.h
    # RigidInertia.inl
    RigidTypes.h
    RigidVec6Types.h
    SolidTypes.h
    SolidTypes.inl
    # SparseConstraintTypes.h
    TemplatesAliases.h
    Vec.h
    Vec3Types.h
    VecTypes.h
    defaulttype.h
    init.h
)

set(SOURCE_FILES
    BaseMatrix.cpp
    BoundingBox.cpp
    Frame.cpp
    # RigidInertia.cpp
    SolidTypes.cpp
    TemplatesAliases.cpp
    init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_DEFAULTTYPE")
AddLinkerDependencies(SofaHelper)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
