cmake_minimum_required(VERSION 2.8.8)
project(SofaDefaultType)

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
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/../../")
set(SOFADEFAULTTYPE_COMPILER_FLAGS "-DSOFA_BUILD_DEFAULTTYPE")

# make the name of debug libraries end in _d.
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAFRAMEWORK_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFADEFAULTTYPE_COMPILER_FLAGS}")

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaFramework_libraries
        EXPORT SofaFrameworkTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY ${SofaDefaultType_SOURCE_DIR}/
        DESTINATION "include/sofa/defaulttype"
        COMPONENT SofaFramework_headers
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")
