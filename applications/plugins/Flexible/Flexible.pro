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
    strainMapping/GreenStrainMapping.cpp \
    strainMapping/CorotationalStrainMapping.cpp \
    strainMapping/InvariantMapping.cpp \
    material/HookeForceField.cpp \
    material/MooneyRivlinForceField.cpp \
    material/VolumePreservationForceField.cpp \

HEADERS = initFlexible.h \
    BaseJacobian.h \
    types/DeformationGradientTypes.h \
    types/StrainTypes.h \
    types/AffineTypes.h \
    types/QuadraticTypes.h \
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

README_FILE = Flexible.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
