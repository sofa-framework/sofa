/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <iostream>
#include <sstream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/UnitTest.h>
#include <sofa/helper/vector_algebra.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/PluginManager.h>

#include <SofaSimulationGraph/graph.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseTopology/CubeTopology.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaLoader/MeshObjLoader.h>

// Include of sofaImage classes
#ifdef SOFA_HAVE_IMAGE
#include <ImageTypes.h>
#include <MeshToImageEngine.h>
#include <ImageContainer.h>
#include <ImageSampler.h>
#endif

// Include of sofaFlexible classes
#ifdef SOFA_HAVE_PLUGIN_Flexible
#include <shapeFunction/VoronoiShapeFunction.h>
#include <shapeFunction/ShepardShapeFunction.h>
#include <quadrature/ImageGaussPointSampler.h>
#include <material/HookeForceField.h>
#include <deformationMapping/LinearMapping.h>
#include <strainMapping/CorotationalStrainMapping.h>
#include <strainMapping/GreenStrainMapping.h>
#endif

#ifdef SOFA_HAVE_SOHUSIM
#include <forcefield\StiffSpringLink.h>
#endif


using namespace sofa;
using namespace sofa::helper;
using helper::vector;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using namespace sofa::component::container;
using namespace sofa::component::topology;
using namespace sofa::component::collision;
using namespace sofa::component::visualmodel;
using namespace sofa::component::mapping;
using namespace sofa::component::forcefield;
using namespace sofa::defaulttype;

typedef SReal Scalar;
typedef Vec<3,SReal> Vec3;
typedef Vec<1,SReal> Vec1;
typedef component::odesolver::EulerImplicitSolver EulerImplicitSolver;
typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;

// TypeDef pour les type du plugin image
#ifdef SOFA_HAVE_IMAGE
typedef sofa::component::engine::MeshToImageEngine<ImageUC> MeshToImageEngine_ImageUC;
typedef sofa::component::container::ImageContainer<ImageUC> ImageContainer_ImageUC;
typedef sofa::component::engine::ImageSampler<ImageUC> ImageSampler_ImageUC;
#endif

// TypeDef pour les type du plugin Flexible
#ifdef SOFA_HAVE_PLUGIN_Flexible
//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define V3(type) StdVectorTypes<Vec<3,type>,Vec<3,type>,type>
#define EV3(type) ExtVectorTypes<Vec<3,type>,Vec<3,type>,type>

#define Rigid3(type)  StdRigidTypes<3,type>
#define Affine3(type)  StdAffineTypes<3,type>
#define Quadratic3(type)  StdQuadraticTypes<3,type>

#define F331(type)  DefGradientTypes<3,3,0,type>
#define F321(type)  DefGradientTypes<3,2,0,type>
#define F311(type)  DefGradientTypes<3,1,0,type>

#define F332(type)  DefGradientTypes<3,3,1,type>

#define E321(type)  StrainTypes<3,2,0,type>
#define E311(type)  StrainTypes<3,1,0,type>
#define E332(type)  StrainTypes<3,3,1,type>

#define I331(type)  InvariantStrainTypes<3,3,0,type>

#define U331(type)  PrincipalStretchesStrainTypes<3,3,0,type>
#define U321(type)  PrincipalStretchesStrainTypes<3,2,0,type>
//////////////////////////////////////////////////////////////////////////////////

// container
typedef sofa::component::container::MechanicalObject< E332(double) > MechanicalObjectE332d;
typedef sofa::component::container::MechanicalObject< F332(double) > MechanicalObjectF332d;
typedef sofa::component::container::MechanicalObject< Affine3(double) > MechanicalObjectAffine3d;

// mapping
typedef sofa::component::mapping::LinearMapping< Affine3(double) , V3(double) > LinearMapping_Affine_Vec3d;
typedef sofa::component::mapping::LinearMapping< Affine3(double) , EV3(float) > LinearMapping_Affine_ExtVec3f;
typedef sofa::component::mapping::LinearMapping< Affine3(double) , F332(double) > LinearMapping_Affine_F332;
typedef sofa::component::mapping::LinearMapping< Rigid3(double), Affine3(double)  > LinearMapping_Rigid_Affine;

typedef sofa::component::mapping::SubsetMultiMapping< Affine3(double), Affine3(double) > SubsetMultiMapping_Affine_Affine;
typedef sofa::component::mapping::CorotationalStrainMapping< F332(double), E332(double) > CorotationalStrainMapping_F332_E332;
typedef sofa::component::mapping::GreenStrainMapping< F332(double), E332(double) > GreenStrainMapping_F332_E332;

// sampler
typedef sofa::component::engine::ImageGaussPointSampler<ImageD> ImageGaussPointSampler_ImageD;

// material
typedef sofa::component::forcefield::HookeForceField< E332(double) > HookeForceField_E332;

// shape function
typedef sofa::component::shapefunction::VoronoiShapeFunction< ShapeFunctionTypes<3,double>, ImageUC > VoronoiShapeFunction;
typedef sofa::component::shapefunction::ShepardShapeFunction< ShapeFunctionTypes<3,double> > ShepardShapeFunction;

// Uniform Mass
typedef sofa::component::mass::UniformMass< Affine3(double), double > UniformMass_Affine;
#endif

bool startAnim = true;
bool verbose = false;
SReal complianceValue = 0.1;
SReal dampingRatio = 0.1;
Vec3 gravity(0,-1,0);
SReal dt = 0.01;

/// helper for more compact component creation
template<class Component>
typename Component::SPtr addNew( Node::SPtr parentNode, std::string name="" )
{
    typename Component::SPtr component = New<Component>();
    parentNode->addObject(component);
    component->setName(name);
    return component;
}

/// Create musculoskeletic system
simulation::Node::SPtr createScene()
{
	using helper::vector;// The graph root node
    Node::SPtr  root = simulation::getSimulation()->createNewGraph("root");
    root->setGravity( Coord3(0,-9.81,0) );
    root->setAnimate(false);
    root->setDt(0.001);
    addVisualStyle(root)->setShowVisual(true).setShowCollision(false).setShowMapping(false).setShowBehavior(false);
	
	// Solver
    EulerImplicitSolver::SPtr eulerImplicitSolver = New<EulerImplicitSolver>();
	eulerImplicitSolver->f_rayleighMass.setValue(0.000);
	eulerImplicitSolver->f_rayleighStiffness.setValue(0.005);

    CGLinearSolver::SPtr cgLinearSolver = New<CGLinearSolver>();
	cgLinearSolver->f_maxIter.setValue(1000);
	cgLinearSolver->f_tolerance.setValue(1E-3);
	cgLinearSolver->f_smallDenominatorThreshold.setValue(1E-3);

    root->addObject(eulerImplicitSolver);
    root->addObject(cgLinearSolver);

	// *********************************************************************************
	// ************************************* main  *************************************
	// *********************************************************************************
    Node::SPtr mainScene = root->createChild("main");
	
	/**********************************************************************************/
	/********************************** Rigid Node  ***********************************/
	/**********************************************************************************/
	// Bones gravity center - rigid node which contains bones, articuated system and ...
    Node::SPtr rigidNode = mainScene->createChild("rigidNode");
    MechanicalObjectRigid3d::SPtr rigid_dof = addNew<MechanicalObjectRigid3d>(rigidNode, "dof");
	// write position of dof
	unsigned int numRigid = 5;
    rigid_dof->resize(numRigid);	// number of degree of freedom
    MechanicalObjectRigid3d::WriteVecCoord xrigid = rigid_dof->writePositions();
    xrigid[0].getCenter()=Vec3d( 0, 0, 0);
    xrigid[1].getCenter()=Vec3d(-0.20438,  0.11337, -0.04110);
    xrigid[2].getCenter()=Vec3d(-0.24487, -0.17229, -0.01618);
    xrigid[3].getCenter()=Vec3d(-0.22625, -0.13082, -0.04326);
    xrigid[4].getCenter()=Vec3d(-0.24653, -0.32916,  0.02977);
	
    DiagonalMassRigid3d::SPtr rigid_mass = addNew<DiagonalMassRigid3d>(rigidNode,"mass");
	rigid_mass->showAxisSize.setValue(0.025);
	// set mass
    float in = 0.1f;	
    DiagonalMassRigid3d::MassType::Mat3x3 inertia; inertia.fill(0.0);
    inertia[0][0] = in; inertia[1][1] = in; inertia[2][2] = in;
    DiagonalMassRigid3d::MassType m0, m1, m2, m3, m4;
	// m0
	m0.mass=1.701798355; m0.volume=1;
    m0.inertiaMatrix = inertia; m0.recalc();
	// m1
    m1.mass=0.685277744; m1.volume=0.685277744;
    m1.inertiaMatrix = inertia; m1.recalc();
	// m2
    m2.mass=0.618109080; m2.volume=0.618109080;
    m2.inertiaMatrix = inertia; m2.recalc();
	// m3
    m3.mass=0.640758879; m3.volume=0.640758879;
    m3.inertiaMatrix = inertia; m3.recalc();
	// m4
    m4.mass=0.700385585; m4.volume=0.700385585;
    m4.inertiaMatrix = inertia; m4.recalc();

    rigid_mass->addMass(m0);
    rigid_mass->addMass(m1);
    rigid_mass->addMass(m2);
    rigid_mass->addMass(m3);
    rigid_mass->addMass(m4);
	
    FixedConstraintRigid3d::SPtr rigid_fixedConstraint = addNew<FixedConstraintRigid3d>(rigidNode,"fixedConstraint");
	
	// ================================== Joint Node  ==================================
	Node::SPtr jointNode = rigidNode->createChild("joint");
	
	// mapped dof
    MechanicalObjectRigid3d::SPtr mapped_rigid_dof = addNew<MechanicalObjectRigid3d>(jointNode, "dof");
	unsigned int numMappedRigid = 10;
    mapped_rigid_dof->resize(numMappedRigid);	// number of degree of freedom
    MechanicalObjectRigid3d::WriteVecCoord xmrigid = mapped_rigid_dof->writePositions();
    xmrigid[0].getCenter()=Vec3d(-0.18367, 0.22972, -0.02958);
    xmrigid[1].getCenter()=Vec3d(0.0207088, 0.1163452, 0.0115178);
    xmrigid[2].getCenter()=Vec3d(-0.0176212, -0.1628648, -0.0126722);
    xmrigid[3].getCenter()=Vec3d(0.0133631, 0.0962597, -0.0353593);	xmrigid[3].getOrientation()=Quat(0, 0.609994, -0.121999,  0.782958);
    xmrigid[4].getCenter()=Vec3d(0.0057731, -0.0952803, 0.0233047); xmrigid[4].getOrientation()=Quat(0, 0.609994, -0.121999,  0.782958);
    xmrigid[5].getCenter()=Vec3d(0.0059931, -0.0999703, 0.0231617);
    xmrigid[6].getCenter()=Vec3d(0.0042448, 0.0813336, -0.0105070); 
    xmrigid[7].getCenter()=Vec3d(-0.0052652, 0.0547936, -0.0082770); xmrigid[7].getOrientation()=Quat(0, 0.609994, -0.121999,  0.782958);
    xmrigid[8].getCenter()=Vec3d(-0.0128552, -0.1367464, 0.0503870); xmrigid[8].getOrientation()=Quat(0, 0.609994, -0.121999,  0.782958);
    xmrigid[9].getCenter()=Vec3d(0.00765, 0.0569, -0.0227938);
	
	// Mapping between bones and joint
    RigidRigidMappingRigid3d_to_Rigid3d::SPtr mapping = addNew<RigidRigidMappingRigid3d_to_Rigid3d>(jointNode,"mapping");
    mapping->setModels( rigid_dof.get(), mapped_rigid_dof.get() );
	sofa::helper::vector<unsigned int> repartition;
	repartition.resize(5);
	repartition[0] = 1;
	repartition[1] = 2;
	repartition[2] = 3;
	repartition[3] = 3;
	repartition[4] = 1;
	mapping->setRepartition(repartition);

	// joint spring force field
	double kd = 1.E-1;
	double softKst = 1.E7;
	double hardKst = 1.E8;
	double softKsr = 1.;
	double hardKsr = 1.E7;
	double blocKsr = 3.E5;
	JointSpringForceFieldRigid3d::SPtr joint = addNew<JointSpringForceFieldRigid3d>(jointNode,"joint");

	// s1
	JointSpringForceFieldRigid3d::Spring s1(0, 1, softKst, hardKst, softKsr, hardKsr, blocKsr, -3.14159, 1.57080, -1.73790,  1.47379, -2.18160, 0.00010, kd);
	s1.setFreeAxis(true, true, true, true, true, true);
	Vec<3,double> t1; t1[0]=1.E-4; t1[1]=1.E-4; t1[2]=1.E-4;
	s1.setInitLength(t1);

	// s2
	JointSpringForceFieldRigid3d::Spring s2(2, 6, softKst, hardKst, softKsr, hardKsr, blocKsr, -2.26893, 0.00001, -0.00001, 0.00001, -0.10000, 0.10000, kd);
	s2.setFreeAxis(true, true, true, true, false, false);
	Vec<3,double> t2; t2[0]=1.E-4; t2[1]=1.E-4; t2[2]=1.E-4;
	s2.setInitLength(t2);

	// s3
	JointSpringForceFieldRigid3d::Spring s3(3, 7, softKst, hardKst, softKsr, hardKsr, blocKsr, -0.00001, 0.00001, -2.26893, 0.00001, -0.00001, 0.00001, kd);
	//s3.setFreeAxis(true, true, true, false, false, false);	//s3.setFreeAxis(true, true, true, false, true, false);
	Vec<3,double> t3; t3[0]=1.E-4; t3[1]=1.E-4; t3[2]=1.E-4;
	s3.setInitLength(t3);

	// s4
	JointSpringForceFieldRigid3d::Spring s4(4, 8, softKst, hardKst, softKsr, hardKsr, blocKsr, -0.00001, 0.00001, -2.26893, 0.00001, -0.00001, 0.00001, kd);
	s4.setFreeAxis(true, true, true, false, false, false);	//s4.setFreeAxis(true, true, true, false, true, false);
	Vec<3,double> t4; t4[0]=1.E-4; t4[1]=1.E-4; t4[2]=1.E-4;
	s4.setInitLength(t4);

	// s5
	JointSpringForceFieldRigid3d::Spring s5(5, 9, softKst, hardKst, softKsr, hardKsr, blocKsr, -0.95993, 1.134465, -0.00001, 0.00001, -0.04906, 0.34906, kd);
	s5.setFreeAxis(true, true, true, false, false, false);	//s5.setFreeAxis(true, true, true, true, false, true);
	Vec<3,double> t5; t5[0]=1.E-4; t5[1]=1.E-4; t5[2]=1.E-4;
	s5.setInitLength(t5);

	// add those springs
	joint->addSpring(s1);
	joint->addSpring(s2);
	joint->addSpring(s3);
	joint->addSpring(s4);
	joint->addSpring(s5);
	
	// ================================= Visual Node  ==================================
	Node::SPtr visuNode = rigidNode->createChild("bones");

	// Torso
	Node::SPtr torsoNode = visuNode->createChild("torso");	
    component::visualmodel::OglModel::SPtr ribs = addNew< component::visualmodel::OglModel >(torsoNode,"ribs");
    ribs->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/ribs.obj") );

    component::visualmodel::OglModel::SPtr l_scapula = addNew< component::visualmodel::OglModel >(torsoNode,"l_scapula");
    l_scapula->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/l_scapula.obj") );
	
    component::visualmodel::OglModel::SPtr l_clavicle = addNew< component::visualmodel::OglModel >(torsoNode,"l_clavicle");
    l_clavicle->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/l_clavicle.obj") );
	
    component::visualmodel::OglModel::SPtr spine = addNew< component::visualmodel::OglModel >(torsoNode,"spine");
    spine->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/spine.obj") );
	
    component::visualmodel::OglModel::SPtr r_scapula = addNew< component::visualmodel::OglModel >(torsoNode,"r_scapula");
    r_scapula->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_scapula.obj") );
	
    component::visualmodel::OglModel::SPtr r_clavicle = addNew< component::visualmodel::OglModel >(torsoNode,"r_clavicle");
    r_clavicle->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_clavicle.obj") );
	
	// Humerus
	Node::SPtr r_humerusNode = visuNode->createChild("r_humerus");	
    component::visualmodel::OglModel::SPtr r_humerus = addNew< component::visualmodel::OglModel >(r_humerusNode,"r_humerus");
    r_humerus->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_humerus.obj") );
    RigidMappingRigid3d_to_Ext3f::SPtr r_humerusMapping = addNew<RigidMappingRigid3d_to_Ext3f>(r_humerusNode,"mapping");
    r_humerusMapping->setModels( rigid_dof.get(), r_humerus.get() );
	r_humerusMapping->index.setValue(1);

	// Raduis
	Node::SPtr r_radiusNode = visuNode->createChild("r_radius");	
    component::visualmodel::OglModel::SPtr r_radius = addNew< component::visualmodel::OglModel >(r_radiusNode,"r_radius");
    r_radius->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_radius.obj") );
    RigidMappingRigid3d_to_Ext3f::SPtr r_radiusMapping = addNew<RigidMappingRigid3d_to_Ext3f>(r_radiusNode,"mapping");
    r_radiusMapping->setModels( rigid_dof.get(), r_radius.get() );
	r_radiusMapping->index.setValue(2);

	// Ulna
	Node::SPtr r_ulnaNode = visuNode->createChild("r_ulna");	
    component::visualmodel::OglModel::SPtr r_ulna = addNew< component::visualmodel::OglModel >(r_ulnaNode,"r_ulna");
    r_ulna->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_ulna.obj") );
    RigidMappingRigid3d_to_Ext3f::SPtr r_ulnaMapping = addNew<RigidMappingRigid3d_to_Ext3f>(r_ulnaNode,"mapping");
    r_ulnaMapping->setModels( rigid_dof.get(), r_ulna.get() );
	r_ulnaMapping->index.setValue(3);

	// Hand
	Node::SPtr r_handNode = visuNode->createChild("r_hand");	
    component::visualmodel::OglModel::SPtr r_hand = addNew< component::visualmodel::OglModel >(r_handNode,"r_hand");
    r_hand->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_hand.obj") );
    RigidMappingRigid3d_to_Ext3f::SPtr r_handMapping = addNew<RigidMappingRigid3d_to_Ext3f>(r_handNode,"mapping");
    r_handMapping->setModels( rigid_dof.get(), r_hand.get() );
	r_handMapping->index.setValue(4);

	////////////////////////////////////////////////////////////////////////////////////
	////////////////////////// Muscles attach in bones (Node) //////////////////////////
	////////////////////////////////////////////////////////////////////////////////////
    Node::SPtr attachNode = mainScene->createChild("attach");

	///////////////////////////////////////////
	//r_bicep_med origin on scapula
	///////////////////////////////////////////
	Node::SPtr originNode = attachNode->createChild("r_bicep_med_origin");

	//Add mesh obj loader
	sofa::component::loader::MeshObjLoader::SPtr originLoader = addNew< sofa::component::loader::MeshObjLoader >(originNode,"loader");
	originLoader->setFilename(sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_scapula.obj"));
	originLoader->triangulate.setValue(true);
    originLoader->load();

	//Bones gravity center - rigid node which contains bones, articuated system and ...
    MechanicalObjectRigid3d::SPtr originRigidDof = addNew<MechanicalObjectRigid3d>(originNode, "dof");
	// write position of dof
    originRigidDof->resize(1);	// number of degree of freedom
	originRigidDof->showObject.setValue(true);
	originRigidDof->showObjectScale.setValue(0.05);
    MechanicalObjectRigid3d::WriteVecCoord xoriginrigid = originRigidDof->writePositions();
    xoriginrigid[0].getCenter()=Vec3d(-0.15882, 0.22436, -0.009336);

	// Shepard shape function
	ShepardShapeFunction::SPtr originShapeFunction = addNew<ShepardShapeFunction>(originNode, "shapeFunction");
	originShapeFunction->f_nbRef.setValue(1);
	
	// Mapping between bones and 
    RigidRigidMappingRigid3d_to_Rigid3d::SPtr originMapping = addNew<RigidRigidMappingRigid3d_to_Rigid3d>(originNode,"mapping");
    originMapping->setModels( rigid_dof.get(), originRigidDof.get() );
	sofa::helper::vector<unsigned int> originRepartition;
	originRepartition.resize(5);
	originRepartition[0] = 1;
	originRepartition[1] = 0;
	originRepartition[2] = 0;
	originRepartition[3] = 0;
	originRepartition[4] = 0;
	originMapping->setRepartition(originRepartition);

	// **** affine node ****
	Node::SPtr frameOriginAttachNode = originNode->createChild("frame_attach");
	
	// mstate
	MechanicalObjectAffine3d::SPtr originFrameDof = addNew<MechanicalObjectAffine3d>(frameOriginAttachNode, "dof");
	originFrameDof->showObject.setValue(true);
	originFrameDof->showObjectScale.setValue(0.05);
    originFrameDof->resize(1);	// number of degree of freedom
    MechanicalObjectAffine3d::WriteVecCoord xoriginaffine = originFrameDof->writePositions();
    xoriginaffine[0].getCenter()=Vec3d(-0.15882, 0.22436, -0.009336);
	
	// Linear mapping between rigid and affine
	LinearMapping_Rigid_Affine::SPtr originLinearMapping = addNew<LinearMapping_Rigid_Affine>(frameOriginAttachNode, "Mapping");
	originLinearMapping->setModels( originRigidDof.get(), originFrameDof.get() );
	
	///////////////////////////////////////////
	//r_bicep_med insertion on radius
	///////////////////////////////////////////
	Node::SPtr insertionNode = attachNode->createChild("r_bicep_med_insertion");

	//Add mesh obj loader
	sofa::component::loader::MeshObjLoader::SPtr insertionLoader = addNew< sofa::component::loader::MeshObjLoader >(insertionNode,"loader");
	insertionLoader->setFilename(sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_scapula.obj"));
	insertionLoader->triangulate.setValue(true);
    insertionLoader->load();

	//Bones gravity center - rigid node which contains bones, articuated system and ...
    MechanicalObjectRigid3d::SPtr insertionRigidDof = addNew<MechanicalObjectRigid3d>(insertionNode, "dof");
	// write position of dof
    insertionRigidDof->resize(1);	// number of degree of freedom
	insertionRigidDof->showObject.setValue(true);
	insertionRigidDof->showObjectScale.setValue(0.05);
    MechanicalObjectRigid3d::WriteVecCoord xinsertionrigid = insertionRigidDof->writePositions();
    xinsertionrigid[0].getCenter()=Vec3d(0.0175631, 0.0783997, -0.0253493);

	// Shepard shape function
	ShepardShapeFunction::SPtr insertionShapeFunction = addNew<ShepardShapeFunction>(insertionNode, "shapeFunction");
	insertionShapeFunction->f_nbRef.setValue(1);
	
	// Mapping between bones and 
    RigidRigidMappingRigid3d_to_Rigid3d::SPtr insertionMapping = addNew<RigidRigidMappingRigid3d_to_Rigid3d>(insertionNode,"mapping");
    insertionMapping->setModels( rigid_dof.get(), insertionRigidDof.get() );
	sofa::helper::vector<unsigned int> insertionRepartition;
	insertionRepartition.resize(5);
	insertionRepartition[0] = 0;
	insertionRepartition[1] = 0;
	insertionRepartition[2] = 1;
	insertionRepartition[3] = 0;
	insertionRepartition[4] = 0;
	insertionMapping->setRepartition(insertionRepartition);

	// //// affine node ////
	Node::SPtr frameInsertionAttachNode = insertionNode->createChild("frame_attach");
	
	// mstate
	MechanicalObjectAffine3d::SPtr insertionFrameDof = addNew<MechanicalObjectAffine3d>(frameInsertionAttachNode, "dof");
	insertionFrameDof->showObject.setValue(true);
	insertionFrameDof->showObjectScale.setValue(0.05);
    insertionFrameDof->resize(1);	// number of degree of freedom
    MechanicalObjectAffine3d::WriteVecCoord xinsertionaffine = insertionFrameDof->writePositions();
    xinsertionaffine[0].getCenter()=Vec3d(0.0175631, 0.0783997, -0.0253493);
	
	// Linear mapping between rigid and affine
	LinearMapping_Rigid_Affine::SPtr insertionLinearMapping = addNew<LinearMapping_Rigid_Affine>(frameInsertionAttachNode, "Mapping");
	insertionLinearMapping->setModels( insertionRigidDof.get(), insertionFrameDof.get() );
	
	////////////////////////////////////////////////////////////////////////////////////
	////////////////////////// Independents particles Node /////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////
    Node::SPtr independentParticlesNode = mainScene->createChild("independentParticles");
    MechanicalObjectAffine3d::SPtr independentParticlesDof = addNew< MechanicalObjectAffine3d>(independentParticlesNode,"dof");
	independentParticlesDof->showObject.setValue(true);
	independentParticlesDof->showObjectScale.setValue(0.05);
    independentParticlesDof->resize(5);	// number of degree of freedom
    MechanicalObjectAffine3d::WriteVecCoord xindependentaffine = independentParticlesDof->writePositions();
    xindependentaffine[0].getCenter()=Vec3d(-0.19065, 0.058865, -0.02014);
    xindependentaffine[1].getCenter()=Vec3d(-0.18365, 0.140865, -0.01514);
    xindependentaffine[2].getCenter()=Vec3d(-0.20765, -0.021135, -0.02314);
    xindependentaffine[3].getCenter()=Vec3d(-0.19465, 0.020865, -0.02114);
    xindependentaffine[4].getCenter()=Vec3d(-0.18765, 0.096865, -0.01914);
	
	////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////// Deformable Structure Node /////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////
    Node::SPtr musclesNode = mainScene->createChild("muscles");
	
	// ==================================r_bicep_med  ==================================
	Node::SPtr rbicepmedNode = independentParticlesNode->createChild("r_bicep_med");

	// Add mesh obj loader
	sofa::component::loader::MeshObjLoader::SPtr loader = addNew< sofa::component::loader::MeshObjLoader >(rbicepmedNode,"loader");
	loader->setFilename(sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/muscles/r_bicep_med.obj"));
	loader->triangulate.setValue(true);
    loader->load();
	
	// add rasterizer
	sofa::helper::vector<double> vSize; vSize.push_back(0.001);
	MeshToImageEngine_ImageUC::SPtr rasterizer = addNew< MeshToImageEngine_ImageUC >(rbicepmedNode, "rasterizer");
	rasterizer->setSrc("",loader.get());
	rasterizer->voxelSize.setValue(vSize);
	rasterizer->padSize.setValue(1);
	rasterizer->rotateImage.setValue(0);

	// volumetric image container
	ImageContainer_ImageUC::SPtr image = addNew< ImageContainer_ImageUC >(rbicepmedNode, "image");
	image->setSrc("", rasterizer.get());
	image->drawBB.setValue(0);

	// sampler for automatically positioning control frames
	ImageSampler_ImageUC::SPtr sampler = addNew<ImageSampler_ImageUC> (rbicepmedNode, "sampler");
	helper::OptionsGroup methodOptions(2,"0 - Regular sampling (at voxel center(0) or corners (1)) ","1 - Uniform sampling using Fast Marching and Lloyd relaxation (nbSamples | bias distances=false | nbiterations=100  | FastMarching(0)/Dijkstra(1)=1)");
    methodOptions.setSelectedItem(1);
	sampler->method.setValue(methodOptions);
	helper::WriteAccessor< Data< vector<double> > > p(sampler->param); 	p.push_back(5); 
	sampler->setSrc("", image.get());
	// sampler computed position	
	vector< Vec<3,double> > r_bicep_med_movingframe; r_bicep_med_movingframe.resize(5);
	r_bicep_med_movingframe[0][0]=-0.19065; r_bicep_med_movingframe[0][1]=0.058865; r_bicep_med_movingframe[0][2]=-0.02014;
	r_bicep_med_movingframe[1][0]=-0.18565; r_bicep_med_movingframe[1][1]=0.140865; r_bicep_med_movingframe[1][2]=-0.01514; 
	r_bicep_med_movingframe[2][0]=-0.20765; r_bicep_med_movingframe[2][1]=-0.021135; r_bicep_med_movingframe[2][2]=-0.02314;
	r_bicep_med_movingframe[3][0]=-0.19465; r_bicep_med_movingframe[3][1]=0.020865; r_bicep_med_movingframe[3][2]=-0.02114; 
	r_bicep_med_movingframe[4][0]=-0.18765; r_bicep_med_movingframe[4][1]=0.096865; r_bicep_med_movingframe[4][2]=-0.01914;
	// fixed position
	vector< Vec<3,double> > r_bicep_med_fixedframe; r_bicep_med_fixedframe.resize(2);
	r_bicep_med_fixedframe[0][0]=-0.15882; r_bicep_med_fixedframe[0][1]=0.22436; r_bicep_med_fixedframe[0][2]=-0.009336;
	r_bicep_med_fixedframe[1][0]=-0.22731; r_bicep_med_fixedframe[1][1]=-0.09389; r_bicep_med_fixedframe[1][2]=-0.041530; 
	
	// define frame container
	MechanicalObjectAffine3d::SPtr frameDof = addNew<MechanicalObjectAffine3d>(rbicepmedNode, "dof");
	frameDof->showObject.setValue(true);
	frameDof->showObjectScale.setValue(0.05);
	//frameDof->setSrc("", sampler.get());
	unsigned int r_bicep_med_NbFrameFree = r_bicep_med_fixedframe.size() + r_bicep_med_movingframe.size();
	frameDof->resize( r_bicep_med_NbFrameFree );	// number of degree of freedom
    MechanicalObjectAffine3d::WriteVecCoord xframeX = frameDof->writePositions();
	// origin and insertion (attach frame) / ind = 0,1
	for(unsigned int i=0; i<r_bicep_med_fixedframe.size(); ++i)
		xframeX[i].getCenter()=Vec3d(r_bicep_med_fixedframe[i][0], r_bicep_med_fixedframe[i][1], r_bicep_med_fixedframe[i][2]);
	// moving frame  / ind = 2,3,4,5,6
	for(unsigned int i=0; i<r_bicep_med_movingframe.size(); ++i)
		xframeX[i+r_bicep_med_fixedframe.size()].getCenter()=Vec3d(r_bicep_med_movingframe[i][0], r_bicep_med_movingframe[i][1], r_bicep_med_movingframe[i][2]);

	// Voronoi shape functions
	VoronoiShapeFunction::SPtr shapeFunction = addNew<VoronoiShapeFunction>(rbicepmedNode, "shapeFunction");
	shapeFunction->useDijkstra.setValue(1);
	shapeFunction->f_nbRef.setValue(7);
	shapeFunction->method.setValue(0);
	shapeFunction->setSrc("",image.get());
	shapeFunction->f_position.setParent("@"+frameDof->getName()+".rest_position");
	
	// MultiMapping
	originNode->addChild(rbicepmedNode);  // second parent
	insertionNode->addChild(rbicepmedNode);  // third parent
	SubsetMultiMapping_Affine_Affine::SPtr multiMapping = addNew<SubsetMultiMapping_Affine_Affine>(rbicepmedNode,"mapping");
    multiMapping->addInputModel(independentParticlesDof.get()); // first parent
    multiMapping->addInputModel(originFrameDof.get()); // second parent
    multiMapping->addInputModel(insertionFrameDof.get()); // third parent
    multiMapping->addOutputModel(frameDof.get()); // deformable structure
	////////// init subsetmultimapping position //////////
	// origin and inertion
	multiMapping->addPoint( originFrameDof.get(), 0 );
	multiMapping->addPoint( insertionFrameDof.get(), 0 );
	// moving frame
	multiMapping->addPoint( independentParticlesDof.get(), 0 );
	multiMapping->addPoint( independentParticlesDof.get(), 1 );
	multiMapping->addPoint( independentParticlesDof.get(), 2 );
	multiMapping->addPoint( independentParticlesDof.get(), 3 );
	multiMapping->addPoint( independentParticlesDof.get(), 4 );
	
	///// Passive Behavior Node  /////
	Node::SPtr passiveBehaviorNode = rbicepmedNode->createChild("passiveBehavior");

	// Gauss Sampler
	ImageGaussPointSampler_ImageD::SPtr gaussPtsSampler = addNew<ImageGaussPointSampler_ImageD>(passiveBehaviorNode, "sampler");
	gaussPtsSampler->f_w.setParent("@../"+shapeFunction->getName()+".weights");
	gaussPtsSampler->f_index.setParent("@../"+shapeFunction->getName()+".indices");
	gaussPtsSampler->f_transform.setParent("@../"+shapeFunction->getName()+".transform");
	helper::OptionsGroup gaussPtsSamplerMethodOptions(3,"0 - Gauss-Legendre", "1 - Newton-Cotes", "2 - Elastons");
    gaussPtsSamplerMethodOptions.setSelectedItem(2);
	gaussPtsSampler->f_method.setValue(gaussPtsSamplerMethodOptions);
	gaussPtsSampler->f_order.setValue(4);
	gaussPtsSampler->targetNumber.setValue(100);
	
	// define gauss point container
	MechanicalObjectF332d::SPtr F = addNew<MechanicalObjectF332d>(passiveBehaviorNode, "F");
	F->setSrc("", gaussPtsSampler.get());

	// mapping between frame and gauss points
	LinearMapping_Affine_F332::SPtr FG = addNew<LinearMapping_Affine_F332>(passiveBehaviorNode, "Mapping");
	FG->setModels(frameDof.get(), F.get());

	/// strain E ///
	Node::SPtr ENode = passiveBehaviorNode->createChild("E");
	// strain container
	MechanicalObjectE332d::SPtr E = addNew<MechanicalObjectE332d>(ENode, "E");

	// Mapping
	GreenStrainMapping_F332_E332::SPtr EF = addNew<GreenStrainMapping_F332_E332>(ENode, "mapping");
	EF->setModels(F.get(), E.get());

	// Material property	
	sofa::helper::vector<double> v_youngModulus; v_youngModulus.push_back(1.0E6);
	sofa::helper::vector<double> v_poissonRatio; v_poissonRatio.push_back(0.499);
	HookeForceField_E332::SPtr material = addNew<HookeForceField_E332>(ENode, "ff");
	material->_youngModulus.setValue(v_youngModulus);
	material->_poissonRatio.setValue(v_poissonRatio);

	///// Mass Node  /////
	Node::SPtr massNode = rbicepmedNode->createChild("mass");
	MechanicalObject3d::SPtr particles_dof = addNew< MechanicalObject3d>(massNode,"dof");
	particles_dof->x.setParent("@../"+passiveBehaviorNode->getName()+"/"+gaussPtsSampler->getName()+".position");
	
	UniformMass3::SPtr particles_mass = addNew<UniformMass3>(massNode,"mass");
    particles_mass->totalMass.setValue(0.25);

	LinearMapping_Affine_Vec3d::SPtr mass_mapping = addNew<LinearMapping_Affine_Vec3d>(massNode, "mapping");
	mass_mapping->setModels(frameDof.get(), particles_dof.get());
		
	///// Visual Node  /////
	Node::SPtr muscleVisuNode = rbicepmedNode->createChild("visual");    
	component::visualmodel::OglModel::SPtr m_visual = addNew< component::visualmodel::OglModel >(muscleVisuNode,"visual");
	m_visual->setSrc("", loader.get());
	m_visual->setColor(0.75f, 0.25f, 0.25f, 1.0f);
	LinearMapping_Affine_ExtVec3f::SPtr m_visualMapping = addNew<LinearMapping_Affine_ExtVec3f>(muscleVisuNode,"mapping");
    m_visualMapping->setModels(frameDof.get(), m_visual.get());
	
	return root;
}

int main(int argc, char** argv)
{
    sofa::simulation::graph::init();
    sofa::helper::BackTrace::autodump();
    sofa::core::ExecParams::defaultInstance()->setAspectID(0);

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&startAnim,'a',"start","start the animation loop")
    .option(&verbose,'v',"verbose","print debug info")
    (argc,argv);

    glutInit(&argc,argv);

    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();
    sofa::gui::initMain();

    if (int err = sofa::gui::GUIManager::Init(argv[0],"")) return err;
    if (int err=sofa::gui::GUIManager::createGUI(NULL)) return err;
    sofa::gui::GUIManager::SetDimension(800,600);

    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    //=================================================
	sofa::simulation::Node::SPtr groot = createScene();
    //=================================================

    sofa::simulation::getSimulation()->init(groot.get());
    sofa::gui::GUIManager::SetScene(groot);


    // Run the main loop
    if (int err = sofa::gui::GUIManager::MainLoop(groot))
        return err;

    sofa::simulation::getSimulation()->unload(groot);
    sofa::gui::GUIManager::closeGUI();

    sofa::simulation::graph::cleanup();
    return 0;
}



