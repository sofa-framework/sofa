cmake_minimum_required(VERSION 2.8.12)
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
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/../..")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_DEFAULTTYPE")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION "${SOFAFRAMEWORK_VERSION}")

sofa_install_targets(SofaFramework SofaDefaultType "")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "include/sofa/defaulttype"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")

#
# python wrapping using swig
#
if(SWIG_FOUND AND PYTHONLIBS_FOUND)
    set(PYTHON_PACKAGE_NAME "defaulttype")
    message("Add sofa.${PYTHON_PACKAGE_NAME} python package generation using SWIG")
    include_directories(${PYTHON_INCLUDE_PATH})
    include(${SWIG_USE_FILE})

    set_source_files_properties(${PYTHON_PACKAGE_NAME}.i PROPERTIES CPLUSPLUS ON)
    # TODO optimization, explore the -builtin swig option
    swig_add_module(${PYTHON_PACKAGE_NAME} python ${PYTHON_PACKAGE_NAME}.i)
    swig_link_libraries(${PYTHON_PACKAGE_NAME} ${PYTHON_LIBRARIES} ${PROJECT_NAME})
    # TODO: ${SWIG_MODULE_${PYTHON_PACKAGE_NAME}_REAL_NAME}
    set_target_properties(${SWIG_MODULE_defaulttype_REAL_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SWIG_OUTDIR})
    install(TARGETS ${SWIG_MODULE_defaulttype_REAL_NAME} DESTINATION lib/python2.7/site-packages/sofa)
    install(FILES ${CMAKE_SWIG_OUTDIR}/${PYTHON_PACKAGE_NAME}.py DESTINATION lib/python2.7/site-packages/sofa)
endif()
