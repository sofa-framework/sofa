cmake_minimum_required(VERSION 3.12)
project(Sofa.Compat LANGUAGES CXX)

set(SOFACOMPATSRC_ROOT "src/")

set(HEADER_FILES
    # ${SOFACOMPATSRC_ROOT}/config.h.in 
    ${SOFACOMPATSRC_ROOT}/sofa/helper/vectorData.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/vectorLinks.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/vector.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/vector_device.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/SVector.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/stable_vector.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/vector_algebra.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/vector_algorithm.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/fixed_array.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/fixed_array_algorithms.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/Quater.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/Quater.inl
    ${SOFACOMPATSRC_ROOT}/sofa/helper/DualQuat.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/DualQuat.inl
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/Trackball.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/template.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/Texture.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/FrameBufferObject.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/GLSLShader.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/Transformation.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/RAII.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/BasicShapes.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/Capture.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/VideoRecorderFFMPEG.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/Axis.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/glText.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/glText.inl
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/Color.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/gl/Cylinder.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/types/RGBAColor.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/types/RGBAColor_fwd.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/types/Material.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/types/PrimitiveGroup.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/system/gl.h
    ${SOFACOMPATSRC_ROOT}/sofa/helper/system/glu.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Vec.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Mat.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/MatSym.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Mat_solve_Cholesky.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Mat_solve_LU.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Mat_solve_SVD.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Quat.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Quat.inl
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/BoundingBox.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Color.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Frame.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/Ray.h
    ${SOFACOMPATSRC_ROOT}/sofa/defaulttype/RGBAColor.h
    ${SOFACOMPATSRC_ROOT}/sofa/core/visual/DrawTool.h
    ${SOFACOMPATSRC_ROOT}/sofa/core/visual/DrawToolGL.h
)

set(SOURCE_FILES
    ${SOFACOMPATSRC_ROOT}/sofa/initSofaCompat.cpp
)

sofa_find_package(Sofa.Config REQUIRED)
# sofa_find_package(SofaFramework REQUIRED)
# sofa_find_package(Sofa.Type REQUIRED)
# sofa_find_package(Sofa.Geometry REQUIRED)
# sofa_find_package(Sofa.Topology REQUIRED)
# sofa_find_package(SofaHelper REQUIRED)
# sofa_find_package(SofaDefaulttype REQUIRED)
# sofa_find_package(SofaCore REQUIRED)
# sofa_find_package(SofaSimulationCore REQUIRED)
# sofa_find_package(Sofa.GL QUIET)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config)
# target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Type Sofa.Geometry Sofa.Topology)
# target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaDefaulttype SofaCore SofaSimulationCore)
# if(Sofa.GL_FOUND)
#     target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL)
# endif()

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER SofaFramework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
