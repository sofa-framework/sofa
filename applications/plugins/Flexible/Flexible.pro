load(sofa/pre)
defineAsPlugin(Flexible)

TARGET = Flexible

DEFINES += SOFA_BUILD_Flexible
DEFINES *= EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

SOURCES = initFlexible.cpp \
    types/DeformationGradientTypes.cpp \
    types/StrainTypes.cpp \
    types/AffineComponents.cpp \
    #types/AffineMass.cpp \
    types/QuadraticTypes.cpp \
    quadrature/BaseGaussPointSampler.cpp \
    quadrature/TopologyGaussPointSampler.cpp \
    shapeFunction/BaseShapeFunction.cpp \
    shapeFunction/ShepardShapeFunction.cpp \
    shapeFunction/BarycentricShapeFunction.cpp \
    deformationMapping/DistanceMapping.cpp \
    deformationMapping/ExtensionMapping.cpp \
    deformationMapping/LinearMapping.cpp \
    deformationMapping/TriangleDeformationMapping.cpp \
    deformationMapping/TriangleStrainAverageMapping.cpp \
    strainMapping/GreenStrainMapping.cpp \
    strainMapping/CorotationalStrainMapping.cpp \
    strainMapping/DiagonalStrainMapping.cpp \
    strainMapping/PrincipalStretchesMapping.cpp \
    strainMapping/InvariantMapping.cpp \
    strainMapping/PlasticStrainMapping.cpp \
    strainMapping/RelativeStrainMapping.cpp \
    material/HookeForceField.cpp \
    material/MooneyRivlinForceField.cpp \
    #material/OgdenForceField.cpp \
    material/VolumePreservationForceField.cpp \
    #forceField/FlexibleTetrahedronFEMForceField.cpp \
	forceField/FlexibleForceField.cpp



HEADERS = initFlexible.h \
    BaseJacobian.h \
    helper.h \
    types/DeformationGradientTypes.h \
    types/StrainTypes.h \
    types/AffineTypes.h \
    types/AffineComponents.h \
    #types/AffineMass.h \
    types/QuadraticTypes.h \
    types/PolynomialBasis.h \
    quadrature/BaseGaussPointSampler.h \
    quadrature/TopologyGaussPointSampler.h \
    shapeFunction/BaseShapeFunction.h \
    shapeFunction/ShepardShapeFunction.h \
    shapeFunction/BarycentricShapeFunction.h \
    deformationMapping/BaseDeformationMapping.h \
    deformationMapping/BaseDeformationMapping.inl \
    deformationMapping/DistanceMapping.h \
    deformationMapping/DistanceMapping.inl \
    deformationMapping/ExtensionMapping.h \
    deformationMapping/ExtensionMapping.inl \
    deformationMapping/LinearMapping.h \
    deformationMapping/LinearJacobianBlock.h \
    deformationMapping/LinearJacobianBlock.inl \
    deformationMapping/TriangleDeformationMapping.h \
    deformationMapping/TriangleDeformationMapping.inl \
    deformationMapping/TriangleStrainAverageMapping.h \
    deformationMapping/TriangleStrainAverageMapping.inl \
    strainMapping/BaseStrainMapping.h \
    strainMapping/GreenStrainMapping.h \
    strainMapping/GreenStrainJacobianBlock.h \
    strainMapping/CorotationalStrainMapping.h \
    strainMapping/CorotationalStrainJacobianBlock.h \
    strainMapping/CorotationalStrainJacobianBlock.inl \
    strainMapping/DiagonalStrainMapping.h \
    strainMapping/DiagonalStrainJacobianBlock.h \
    strainMapping/PrincipalStretchesMapping.h \
    strainMapping/PrincipalStretchesJacobianBlock.h \
    strainMapping/InvariantMapping.h \
    strainMapping/InvariantJacobianBlock.h \
    strainMapping/InvariantJacobianBlock.inl \
    strainMapping/PlasticStrainMapping.h \
    strainMapping/PlasticStrainJacobianBlock.h \
    strainMapping/RelativeStrainMapping.h \
    strainMapping/RelativeStrainJacobianBlock.h \
    material/BaseMaterial.h \
    material/BaseMaterialForceField.h \
    material/HookeForceField.h \
    material/HookeMaterialBlock.h \
    material/HookeMaterialBlock.inl \
    material/MooneyRivlinForceField.h \
    material/MooneyRivlinMaterialBlock.h \
    material/MooneyRivlinMaterialBlock.inl \
    #material/OgdenForceField.h \
    #material/OgdenMaterialBlock.h \
    material/VolumePreservationForceField.h \
    material/VolumePreservationMaterialBlock.h \
    material/VolumePreservationMaterialBlock.inl \
    #forceField/FlexibleTetrahedronFEMForceField.h \
    shapeFunction/BaseShepardShapeFunction.h \
    forceField/FlexibleForceField.h 



contains(DEFINES, SOFA_HAVE_IMAGE) {

    contains(DEFINES, SOFA_IMAGE_HAVE_OPENCV) { # should be "SOFA_HAVE_OPENCV" -> use "SOFA_IMAGE_HAVE_OPENCV" until the opencv plugin is fixed..
            INCLUDEPATH += $$SOFA_OPENCV_PATH
            LIBS += -lml  -lcvaux -lhighgui -lcv -lcxcore
            }

    INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs \
                   $$SOFA_INSTALL_INC_DIR/applications/plugins

    HEADERS +=  quadrature/ImageGaussPointSampler.h \
                shapeFunction/BaseImageShapeFunction.h \
                shapeFunction/VoronoiShapeFunction.h \
                shapeFunction/DiffusionShapeFunction.h \
                deformationMapping/ImageDeformation.h \
                mass/ImageDensityMass.h \
                mass/ImageDensityMass.inl \

    SOURCES += quadrature/ImageGaussPointSampler.cpp \
               shapeFunction/VoronoiShapeFunction.cpp \
               shapeFunction/DiffusionShapeFunction.cpp \
               deformationMapping/ImageDeformation.cpp \
               mass/ImageDensityMass.cpp \

    }

README_FILE = Flexible.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

win32 {
	QMAKE_CXXFLAGS += /bigobj
	INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs/SuiteSparse/cholmod/Include
	DEFINES += EIGEN_DONT_ALIGN
}
unix {
    INCLUDEPATH += /usr/include/suitesparse/
}

load(sofa/post)
