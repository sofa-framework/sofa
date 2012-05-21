load(sofa/pre)
defineAsPlugin(Flexible)

TARGET = Flexible

DEFINES += SOFA_BUILD_Flexible
DEFINES *= EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

SOURCES = initFlexible.cpp\
    types/DeformationGradientTypes.cpp \
    types/StrainTypes.cpp \
    types/AffineTypes.cpp \
    types/QuadraticTypes.cpp \
    quadrature/BaseGaussPointSampler.cpp \
    quadrature/TopologyGaussPointSampler.cpp \
    shapeFunction/BaseShapeFunction.cpp \
    shapeFunction/ShepardShapeFunction.cpp \
    shapeFunction/BarycentricShapeFunction.cpp \
    deformationMapping/DistanceMapping.cpp \
    deformationMapping/ExtensionMapping.cpp \
    deformationMapping/LinearMapping.cpp \
    deformationMapping/LogRigidMapping.cpp \
    deformationMapping/RelativeRigidMapping.cpp \
    deformationMapping/TriangleDeformationMapping.cpp \
    deformationMapping/TriangleStrainAverageMapping.cpp \
    strainMapping/GreenStrainMapping.cpp \
    strainMapping/CorotationalStrainMapping.cpp \
    strainMapping/InvariantMapping.cpp \
    material/HookeForceField.cpp \
    material/MooneyRivlinForceField.cpp \
    material/VolumePreservationForceField.cpp \

HEADERS = initFlexible.h \
    BaseJacobian.h \
    helper.h \
    types/DeformationGradientTypes.h \
    types/StrainTypes.h \
    types/AffineTypes.h \
    types/QuadraticTypes.h \
    types/PolynomialBasis.h \
    quadrature/BaseGaussPointSampler.h \
    quadrature/TopologyGaussPointSampler.h \
    shapeFunction/BaseShapeFunction.h \
    shapeFunction/ShepardShapeFunction.h \
    shapeFunction/BarycentricShapeFunction.h \
    deformationMapping/BaseDeformationMapping.h \
    deformationMapping/DistanceMapping.h \
    deformationMapping/DistanceMapping.inl \
    deformationMapping/ExtensionMapping.h \
    deformationMapping/ExtensionMapping.inl \
    deformationMapping/LinearMapping.h \
    deformationMapping/LinearJacobianBlock.h \
    deformationMapping/LinearJacobianBlock.inl \
    deformationMapping/LogRigidMapping.h \
    deformationMapping/LogRigidMapping.inl \
    deformationMapping/RelativeRigidMapping.h \
    deformationMapping/RelativeRigidMapping.inl \
    deformationMapping/TriangleDeformationMapping.h \
    deformationMapping/TriangleDeformationMapping.inl \
    deformationMapping/TriangleStrainAverageMapping.h \
    deformationMapping/TriangleStrainAverageMapping.inl \
    strainMapping/BaseStrainMapping.h \
    strainMapping/GreenStrainMapping.h \
    strainMapping/GreenStrainJacobianBlock.h \
    strainMapping/GreenStrainJacobianBlock.inl \
    strainMapping/CorotationalStrainMapping.h \
    strainMapping/CorotationalStrainJacobianBlock.h \
    strainMapping/CorotationalStrainJacobianBlock.inl \
    strainMapping/InvariantMapping.h \
    strainMapping/InvariantJacobianBlock.h \
    strainMapping/InvariantJacobianBlock.inl \
    material/BaseMaterial.h \
    material/BaseMaterialForceField.h \
    material/HookeForceField.h \
    material/HookeMaterialBlock.h \
    material/HookeMaterialBlock.inl \
    material/MooneyRivlinForceField.h \
    material/MooneyRivlinMaterialBlock.h \
    material/MooneyRivlinMaterialBlock.inl \
    material/VolumePreservationForceField.h \
    material/VolumePreservationMaterialBlock.h \
    material/VolumePreservationMaterialBlock.inl \

contains(DEFINES, SOFA_HAVE_IMAGE) {

    contains(DEFINES, SOFA_IMAGE_HAVE_OPENCV) { # should be "SOFA_HAVE_OPENCV" -> use "SOFA_IMAGE_HAVE_OPENCV" until the opencv plugin is fixed..
            INCLUDEPATH += $$SOFA_OPENCV_PATH
            LIBS += -lml  -lcvaux -lhighgui -lcv -lcxcore
            }

    INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs/CImg \
                   $$SOFA_INSTALL_INC_DIR/applications/plugins/image

    HEADERS +=  quadrature/ImageGaussPointSampler.h \
                shapeFunction/VoronoiShapeFunction.h \

    SOURCES += quadrature/ImageGaussPointSampler.cpp \
               shapeFunction/VoronoiShapeFunction.cpp \
    }

README_FILE = Flexible.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
