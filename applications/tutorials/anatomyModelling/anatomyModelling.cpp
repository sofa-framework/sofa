/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
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

//#include <sofa/simulation/tree/TreeSimulation.h>
#ifdef SOFA_HAVE_DAG
#include <sofa/simulation/graph/DAGSimulation.h>
#endif
#ifdef SOFA_HAVE_BGL
#include <sofa/simulation/bgl/BglSimulation.h>
#endif
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/component/init.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/topology/CubeTopology.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/loader/MeshObjLoader.h>

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
#endif

#ifdef SOFA_HAVE_SOHUSIM
#include <forcefield\StiffSpringLink.h>
#endif

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>

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

// mapping
typedef sofa::component::container::MechanicalObject< E332(double) > MechanicalObjectE332d;
typedef sofa::component::container::MechanicalObject< F332(double) > MechanicalObjectF332d;
typedef sofa::component::container::MechanicalObject< Affine3(double) > MechanicalObjectAffine3d;

typedef sofa::component::mapping::LinearMapping< Affine3(double) , V3(double) > LinearMapping_Affine_Vec3d;
typedef sofa::component::mapping::LinearMapping< Affine3(double) , EV3(float) > LinearMapping_Affine_ExtVec3f;
typedef sofa::component::mapping::LinearMapping< Affine3(double) , F332(double) > LinearMapping_Affine_F332;

typedef sofa::component::mapping::CorotationalStrainMapping< F332(double), E332(double) > CorotationalStrainMapping_F332_E332;

// sampler
typedef sofa::component::engine::ImageGaussPointSampler<ImageD> ImageGaussPointSampler_ImageD;

// material
typedef sofa::component::forcefield::HookeForceField< E332(double) > HookeForceField_E332;

// shape function
typedef sofa::component::shapefunction::VoronoiShapeFunction< ShapeFunctionTypes<3,double>, ImageUC> VoronoiShapeFunction;
typedef sofa::component::shapefunction::ShepardShapeFunction< ShapeFunctionTypes<3,double>> ShepardShapeFunction;

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
    root->setDt(0.005);
    addVisualStyle(root)->setShowVisual(true).setShowCollision(false).setShowMapping(false).setShowBehavior(false);
	
	// Solver
    EulerImplicitSolver::SPtr eulerImplicitSolver = New<EulerImplicitSolver>();
    CGLinearSolver::SPtr cgLinearSolver = New<CGLinearSolver>();
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
    rigid_mass->addMass(1.701798355);
    rigid_mass->addMass(0.685277744);
    rigid_mass->addMass(0.618109080);
    rigid_mass->addMass(0.640758879);
    rigid_mass->addMass(0.700385585);
		
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
	
	// Mapping between bones and 
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
	s3.setFreeAxis(true, true, true, false, true, false);
	Vec<3,double> t3; t3[0]=1.E-4; t3[1]=1.E-4; t3[2]=1.E-4;
	s3.setInitLength(t3);

	// s4
	JointSpringForceFieldRigid3d::Spring s4(4, 8, softKst, hardKst, softKsr, hardKsr, blocKsr, -0.00001, 0.00001, -2.26893, 0.00001, -0.00001, 0.00001, kd);
	s4.setFreeAxis(true, true, true, false, true, false);
	Vec<3,double> t4; t4[0]=1.E-4; t4[1]=1.E-4; t4[2]=1.E-4;
	s4.setInitLength(t4);

	// s5
	JointSpringForceFieldRigid3d::Spring s5(5, 9, softKst, hardKst, softKsr, hardKsr, blocKsr, -0.95993, 1.134465, -0.00001, 0.00001, -0.04906, 0.34906, kd);
	s5.setFreeAxis(true, true, true, true, false, true);
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
	//Node::SPtr r_humerusNode = visuNode->createChild("r_humerus");	
    //component::visualmodel::OglModel::SPtr r_humerus = addNew< component::visualmodel::OglModel >(r_humerusNode,"r_humerus");
    //r_humerus->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_humerus.obj") );
    //RigidMappingRigid3d_to_Ext3f::SPtr r_humerusMapping = addNew<RigidMappingRigid3d_to_Ext3f>(r_humerusNode,"mapping");
    //r_humerusMapping->setModels( rigid_dof.get(), r_humerus.get() );
	//r_humerusMapping->index.setValue(1);

	// Raduis
	//Node::SPtr r_radiusNode = visuNode->createChild("r_radius");	
    //component::visualmodel::OglModel::SPtr r_radius = addNew< component::visualmodel::OglModel >(r_radiusNode,"r_radius");
    //r_radius->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_radius.obj") );
    //RigidMappingRigid3d_to_Ext3f::SPtr r_radiusMapping = addNew<RigidMappingRigid3d_to_Ext3f>(r_radiusNode,"mapping");
    //r_radiusMapping->setModels( rigid_dof.get(), r_radius.get() );
	//r_radiusMapping->index.setValue(2);

	//// Ulna
	//Node::SPtr r_ulnaNode = visuNode->createChild("r_ulna");	
    //component::visualmodel::OglModel::SPtr r_ulna = addNew< component::visualmodel::OglModel >(r_ulnaNode,"r_ulna");
    //r_ulna->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_ulna.obj") );
    //RigidMappingRigid3d_to_Ext3f::SPtr r_ulnaMapping = addNew<RigidMappingRigid3d_to_Ext3f>(r_ulnaNode,"mapping");
    //r_ulnaMapping->setModels( rigid_dof.get(), r_ulna.get() );
	//r_ulnaMapping->index.setValue(3);

	//// Hand
	//Node::SPtr r_handNode = visuNode->createChild("r_hand");	
    //component::visualmodel::OglModel::SPtr r_hand = addNew< component::visualmodel::OglModel >(r_handNode,"r_hand");
    //r_hand->setFilename( sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_hand.obj") );
    //RigidMappingRigid3d_to_Ext3f::SPtr r_handMapping = addNew<RigidMappingRigid3d_to_Ext3f>(r_handNode,"mapping");
    //r_handMapping->setModels( rigid_dof.get(), r_hand.get() );
	//r_handMapping->index.setValue(4);

	/**********************************************************************************/
	/************************* Muscles attach in bones (Node)  ************************/
	/**********************************************************************************/
    Node::SPtr attachNode = mainScene->createChild("attach");

	//r_bicep_med origin on scapula
	Node::SPtr originNode = attachNode->createChild("r_bicep_med_origin");

	//Add mesh obj loader
	sofa::component::loader::MeshObjLoader::SPtr originLoader = addNew< sofa::component::loader::MeshObjLoader >(originNode,"loader");
	originLoader->setFilename(sofa::helper::system::DataRepository.getFile("../applications/tutorials/anatomyModelling/mesh/bones/r_scapula.obj"));
	originLoader->triangulate.setValue(true);
    originLoader->load();

	//Bones gravity center - rigid node which contains bones, articuated system and ...
    MechanicalObjectRigid3d::SPtr originRigid_dof = addNew<MechanicalObjectRigid3d>(originNode, "dof");
	// write position of dof
    originRigid_dof->resize(1);	// number of degree of freedom
    MechanicalObjectRigid3d::WriteVecCoord xoriginrigid = originRigid_dof->writePositions();
    xoriginrigid[0].getCenter()=Vec3d(-0.15882, 0.22436, -0.009336);

	// Shepard shape function
	ShepardShapeFunction::SPtr originShapeFunction = addNew<ShepardShapeFunction>(originNode, "shapeFunction");
	originShapeFunction->f_nbRef.setValue(1);

	// affine node
	Node::SPtr frameAttachNode = attachNode->createChild("frame_attach");

	// affine position

	// linear mapping

	//r_bicep_med insertion on radius
	
	
	/**********************************************************************************/
	/*************************** Deformable Structure Node  ***************************/
	/**********************************************************************************/
    Node::SPtr musclesNode = mainScene->createChild("muscles");
	
	// ==================================r_bicep_med  ==================================
	Node::SPtr rbicepmedNode = musclesNode->createChild("r_bicep_med");

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
	helper::WriteAccessor< Data< vector<double> > > p(sampler->param); 	p.push_back(3); 
	sampler->setSrc("", image.get());

	// define frame container
	MechanicalObjectAffine3d::SPtr frameDof = addNew<MechanicalObjectAffine3d>(rbicepmedNode, "dof");
	frameDof->showObject.setValue(true);
	frameDof->showObjectScale.setValue(0.05);
	frameDof->setSrc("", sampler.get());

	// Voronoi shape functions
	VoronoiShapeFunction::SPtr shapeFunction = addNew<VoronoiShapeFunction>(rbicepmedNode, "shapeFunction");
	shapeFunction->useDijkstra.setValue(1);
	shapeFunction->f_nbRef.setValue(3);
	shapeFunction->method.setValue(0);
	shapeFunction->setSrc("",image.get());
	shapeFunction->f_position.setParent("@"+frameDof->getName()+".rest_position");
		
	/**** Passive Behavior Node  ****/
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

	/** strain E **/
	Node::SPtr ENode = passiveBehaviorNode->createChild("E");
	// strain container
	MechanicalObjectE332d::SPtr E = addNew<MechanicalObjectE332d>(ENode, "E");

	// Mapping
	CorotationalStrainMapping_F332_E332::SPtr EF = addNew<CorotationalStrainMapping_F332_E332>(ENode, "mapping");
	EF->setModels(F.get(), E.get());

	// Material property	
	sofa::helper::vector<double> v_youngModulus; v_youngModulus.push_back(1.0E6);
	sofa::helper::vector<double> v_poissonRatio; v_poissonRatio.push_back(0.499);
	HookeForceField_E332::SPtr material = addNew<HookeForceField_E332>(ENode, "ff");
	material->_youngModulus.setValue(v_youngModulus);
	material->_poissonRatio.setValue(v_poissonRatio);

	/**** Mass Node  ****/
	Node::SPtr massNode = rbicepmedNode->createChild("mass");
	MechanicalObject3d::SPtr particles_dof = addNew< MechanicalObject3d>(massNode,"dof");
	particles_dof->x.setParent("@../"+passiveBehaviorNode->getName()+"/"+gaussPtsSampler->getName()+".position");
	
	UniformMass3::SPtr particles_mass = addNew<UniformMass3>(massNode,"mass");
    particles_mass->totalMass.setValue(0.25);

	LinearMapping_Affine_Vec3d::SPtr mass_mapping = addNew<LinearMapping_Affine_Vec3d>(massNode, "mapping");
	mass_mapping->setModels(frameDof.get(), particles_dof.get());
		
	/**** Visual Node  ****/
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

    sofa::helper::BackTrace::autodump();
    sofa::core::ExecParams::defaultInstance()->setAspectID(0);

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&startAnim,'a',"start","start the animation loop")
    .option(&verbose,'v',"verbose","print debug info")
    (argc,argv);

    glutInit(&argc,argv);

    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    sofa::component::init();

    sofa::gui::initMain();
    if (int err = sofa::gui::GUIManager::Init(argv[0],"")) return err;
    if (int err=sofa::gui::GUIManager::createGUI(NULL)) return err;
    sofa::gui::GUIManager::SetDimension(800,600);

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

    return 0;
}



