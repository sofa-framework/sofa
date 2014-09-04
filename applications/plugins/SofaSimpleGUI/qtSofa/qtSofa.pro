################################################################
# An application which uses Sofa, built as an independent project linked to Sofa

QT += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = myQGLSofa
TEMPLATE = app

SOURCES += QSofaMainWindow.cpp QSofaScene.cpp QSofaViewer.cpp qtSofa.cpp

HEADERS  +=  oneTetra.h QSofaMainWindow.h QSofaScene.h QSofaViewer.h 

LIBS += -lglut -lGLU -lGLEW


###### SOFA  ####################################################
#### customize the following paths:
#QMAKE_CXX = clang++
#QMAKE_CXXFLAGS += -Wno-overloaded-virtual -Wno-unused-parameter
#QMAKE_CC = clang
SOFA_SRC=/home/faure/scm/sofa-git/sofa/
DEFINES += QTSOFA_SRC_DIR=\\\"/home/faure/scm/sofa-git/sofa/applications/plugins/SofaSimpleGUI/qtSofa\\\"
SOFA_BUILD_DEBUG=/home/faure/scm/sofa-git/clang-debug              # required for a debug version
SOFA_BUILD_RELEASE=/home/evasion/ffaure/scm/sofa-git/sofa-build-clang34-release # required for a release version
################################################################

INCLUDEPATH += $${SOFA_SRC}/applications/plugins
INCLUDEPATH += $${SOFA_SRC}/framework
INCLUDEPATH += $${SOFA_SRC}/modules
INCLUDEPATH += $${SOFA_SRC}/extlibs/tinyxml
INCLUDEPATH += $${SOFA_SRC}/extlibs/newmat
INCLUDEPATH += $${SOFA_SRC}/extlibs/eigen-3.2.1

DEFINES += SOFA_HAVE_GLEW

CONFIG(debug, debug|release) {

INCLUDEPATH += $${SOFA_BUILD_DEBUG}/misc/include

LIBS += -L$${SOFA_BUILD_DEBUG}/lib
LIBS += -lSofaSimpleGUId
LIBS += -lSofaSimulationGraphd -lSofaComponentMaind -lSofaComponentBased -lSofaBaseAnimationLoopd -lSofaComponentCommond -lSofaComponentGenerald -lSofaValidationd -lSofaLoaderd
LIBS += -lSofaExporterd -lSofaEngined -lSofaHapticsd -lSofaPreconditionerd -lSofaComponentAdvancedd -lSofaEulerianFluidd -lSofaComponentMiscd -lSofaMiscMappingd -lSofaMiscForceFieldd -lSofaMiscFemd
LIBS += -lSofaMiscEngined -lSofaNonUniformFemd -lSofaMiscCollisiond -lSofaExplicitOdeSolverd -lSofaConstraintd -lSofaSimpleFemd -lSofaOpenglVisuald -lSofaImplicitOdeSolverd -lSofaVolumetricDatad
LIBS += -lSofaMiscSolverd -lSofaMiscd -lSofaMiscTopologyd -lSofaUserInteractiond -lSofaBaseVisuald -lSofaMeshCollisiond -lSofaBaseCollisiond -lSofaRigidd -lSofaSphFluidd -lSofaBaseMechanicsd
LIBS += -lSofaObjectInteractiond -lSofaDeformabled -lSofaGraphComponentd -lSofaTopologyMappingd -lSofaBoundaryConditiond -lSofaBaseTopologyd -lSofaEigen2Solverd -lSofaDenseSolverd
LIBS += -lSofaBaseLinearSolverd -lSofaSimulationTreed -lSofaSimulationCommond -lSofaCored -lSofaDefaultTyped -lSofaHelperd
LIBS += -lpng -lz -ltinyxmld
LIBS += -L$${SOFA_BUILD_DEBUG}/extlibs/newmat  -lnewmatd

}
else {

INCLUDEPATH += $${SOFA_BUILD_RELEASE}/misc/include

LIBS += -L$${SOFA_BUILD_RELEASE}/lib
LIBS += -lSofaSimpleGUI
LIBS += -lSofaSimulationGraph -lSofaComponentMain -lSofaComponentBase -lSofaBaseAnimationLoop -lSofaComponentCommon -lSofaComponentGeneral -lSofaValidation -lSofaLoader
LIBS += -lSofaExporter -lSofaEngine -lSofaHaptics -lSofaPreconditioner -lSofaComponentAdvanced -lSofaEulerianFluid -lSofaComponentMisc -lSofaMiscMapping -lSofaMiscForceField -lSofaMiscFem
LIBS += -lSofaMiscEngine -lSofaNonUniformFem -lSofaMiscCollision -lSofaExplicitOdeSolver -lSofaConstraint -lSofaSimpleFem -lSofaOpenglVisual -lSofaImplicitOdeSolver -lSofaVolumetricData
LIBS += -lSofaMiscSolver -lSofaMisc -lSofaMiscTopology -lSofaUserInteraction -lSofaBaseVisual -lSofaMeshCollision -lSofaBaseCollision -lSofaRigid -lSofaSphFluid -lSofaBaseMechanics
LIBS += -lSofaObjectInteraction -lSofaDeformable -lSofaGraphComponent -lSofaTopologyMapping -lSofaBoundaryCondition -lSofaBaseTopology -lSofaEigen2Solver -lSofaDenseSolver
LIBS += -lSofaBaseLinearSolver -lSofaSimulationTree -lSofaSimulationCommon -lSofaCore -lSofaDefaultType -lSofaHelper
LIBS += -lpng -lz -ltinyxml
LIBS += -L$${SOFA_BUILD_RELEASE}/extlibs/newmat  -lnewmat
}
