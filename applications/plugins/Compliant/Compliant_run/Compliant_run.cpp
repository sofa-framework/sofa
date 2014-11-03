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
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <SofaComponentMain/init.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaMiscMapping/DistanceMapping.h>
#include <SofaMiscMapping/DistanceFromTargetMapping.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseTopology/CubeTopology.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
#include <plugins/SceneCreator/SceneCreator.h>

#include <plugins/Compliant/odesolver/CompliantImplicitSolver.h>
#include <plugins/Compliant/numericalsolver/LDLTSolver.h>
#include <plugins/Compliant/compliance/UniformCompliance.h>
#include <plugins/Compliant/misc/CompliantAttachButtonSetting.h>
#include <plugins/Compliant/constraint/ConstraintValue.h>
using sofa::component::configurationsetting::CompliantAttachButtonSetting;

#include <sofa/simulation/common/Simulation.h>
#include <plugins/SceneCreator/SceneCreator.h>

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

typedef SReal Scalar;
typedef Vec<3,SReal> Vec3;
typedef Vec<1,SReal> Vec1;
typedef DistanceMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> DistanceMapping31;
typedef DistanceFromTargetMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> DistanceFromTargetMapping31;
typedef UniformCompliance<defaulttype::Vec1Types> UniformCompliance1;

typedef component::odesolver::CompliantImplicitSolver CompliantImplicitSolver;
typedef component::linearsolver::LDLTSolver LDLTSolver;
typedef component::odesolver::EulerImplicitSolver EulerImplicitSolver;
typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;


bool startAnim = true;
bool verbose = false;
std::string simulationType = "bgl";
SReal complianceValue = 0.1;
SReal dampingRatio = 0.1;
Vec3 gravity(0,-1,0);
SReal dt = 0.01;

/// Create a compliant string
simulation::Node::SPtr createCompliantString(simulation::Node::SPtr parent, Vec3 startPoint, Vec3 endPoint, unsigned numParticles, double totalMass, double complianceValue=0, double /*dampingRatio*/=0 )
{
    static unsigned numObject = 1;
    std::ostringstream oss;
    oss << "string_" << numObject++;
    SReal totalLength = (endPoint-startPoint).norm();

    //--------
    Node::SPtr  string_node = parent->createChild(oss.str());

    MechanicalObject3::SPtr DOF = New<MechanicalObject3>();
    string_node->addObject(DOF);
    DOF->setName(oss.str()+"_DOF");

    UniformMass3::SPtr mass = New<UniformMass3>();
    string_node->addObject(mass);
    mass->setName(oss.str()+"_mass");
    mass->mass.setValue( totalMass/numParticles );




    //--------
    Node::SPtr extension_node = string_node->createChild( oss.str()+"_ExtensionNode");

    MechanicalObject1::SPtr extensions = New<MechanicalObject1>();
    extension_node->addObject(extensions);

    EdgeSetTopologyContainer::SPtr edgeSet = New<EdgeSetTopologyContainer>();
    extension_node->addObject(edgeSet);

    DistanceMapping31::SPtr extensionMapping = New<DistanceMapping31>();
    extensionMapping->setModels(DOF.get(),extensions.get());
    extension_node->addObject( extensionMapping );
    extensionMapping->setName(oss.str()+"_DistanceMapping");
    extensionMapping->setModels( DOF.get(), extensions.get() );

    UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
    extension_node->addObject(compliance);
    compliance->setName(oss.str()+"_compliance");
    compliance->compliance.setValue(complianceValue);

//    if( dampingRatio )
//    {
//        component::odesolver::ConstraintValue::SPtr constraintValue = New<component::odesolver::ConstraintValue>( extensions.get() );
//        constraintValue->dampingRatio.setValue(dampingRatio);
//        extension_node->addObject(constraintValue);
//    }


    //--------
    // create the particles
    DOF->resize(numParticles);
    MechanicalObject3::WriteVecCoord x = DOF->writePositions();
    helper::vector<SReal> restLengths;
    for( unsigned i=0; i<numParticles; i++ )
    {
        double alpha = (double)i/(numParticles-1);
        x[i] = startPoint * (1-alpha)  +  endPoint * alpha;
        if(i>0)
        {
            edgeSet->addEdge(i-1,i);
            restLengths.push_back( totalLength/(numParticles-1) );
        }
    }
    extensionMapping->f_restLengths.setValue( restLengths );

    //    {
    //        //-------- fix a particle
    //        Node::SPtr fixNode = string_node->createChild("fixNode");
    //        MechanicalObject1::SPtr extensions = New<MechanicalObject1>();
    //        fixNode->addObject(extensions);

    //        DistanceFromTargetMapping31::SPtr distanceMapping = New<DistanceFromTargetMapping31>();
    //        distanceMapping->setModels(DOF.get(),extensions.get());
    //        fixNode->addObject( distanceMapping );
    //        distanceMapping->setName("fix_distanceMapping");
    //        distanceMapping->setModels( DOF.get(), extensions.get() );
    //        distanceMapping->createTarget( numParticles-1, endPoint, 0.0 );

    //        UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
    //        fixNode->addObject(compliance);
    //        compliance->setName("fix_compliance");
    //        compliance->compliance.setValue(complianceValue);
    //        compliance->dampingRatio.setValue(dampingRatio);
    //    }

    return string_node;

}


/// Create the compliant string composed of three parts
simulation::Node::SPtr createCompliantScene()
{
    // The graph root node
    Node::SPtr  root = simulation::getSimulation()->createNewGraph("root");
    root->setGravity( Coord3(0,-1,0) );
    root->setAnimate(false);
    root->setDt(0.01);
    addVisualStyle(root)->setShowVisual(false).setShowCollision(false).setShowMapping(true).setShowBehavior(true);


//    CompliantAttachButtonSetting::SPtr buttonSetting = New<CompliantAttachButtonSetting>();
//    root->addObject(buttonSetting);
//    sofa::helper::OptionsGroup b=buttonSetting->button.getValue();
//    b.setSelectedItem("Left");
//    buttonSetting->button.setValue(b);

    Node::SPtr simulatedScene = root->createChild("simulatedScene");


    CompliantImplicitSolver::SPtr assembledSolver = New<CompliantImplicitSolver>();
    simulatedScene->addObject( assembledSolver );
//    assembledSolver->verbose.setValue(verbose);

    LDLTSolver::SPtr lDLTSolver = New<LDLTSolver>();
    simulatedScene->addObject( lDLTSolver );
//    lDLTSolver->verbose.setValue(verbose);


    // ========  first string
    unsigned n1 = 2;
    Node::SPtr  string1 = createCompliantString( simulatedScene, Vec3(0,0,0), Vec3(1,0,0), n1, 1.0*n1, complianceValue, dampingRatio );


    FixedConstraint3::SPtr fixed1 = New<FixedConstraint3>();
    string1->addObject( fixed1 );

    // ========  second string
    unsigned n2 = 2;
    Node::SPtr  string2 = createCompliantString( simulatedScene, Vec3(3,0,0), Vec3(2,0,0), n2, 1.0*n2, complianceValue, dampingRatio );
    FixedConstraint3::SPtr fixed2 = New<FixedConstraint3>();
    string2->addObject( fixed2 );


    // ========  Node with multiple parents to create an interaction using a MultiMapping
    Node::SPtr commonChild = string1->createChild("commonChild");
    string2->addChild(commonChild);

    MechanicalObject3::SPtr mappedDOF = New<MechanicalObject3>(); // to contain particles from the two strings
    commonChild->addObject(mappedDOF);

    SubsetMultiMapping3_to_3::SPtr multimapping = New<SubsetMultiMapping3_to_3>();
    multimapping->setName("InteractionMultiMapping");
    multimapping->addInputModel( string1->getMechanicalState() );
    multimapping->addInputModel( string2->getMechanicalState() );
    multimapping->addOutputModel( mappedDOF.get() );
    multimapping->addPoint( string1->getMechanicalState(), n1-1 );
    multimapping->addPoint( string2->getMechanicalState(), n2-1 );
    commonChild->addObject(multimapping);

    // Node to handle the extension of the interaction link
    Node::SPtr extension_node = commonChild->createChild("InteractionExtensionNode");

    MechanicalObject1::SPtr extensions = New<MechanicalObject1>();
    extension_node->addObject(extensions);

    EdgeSetTopologyContainer::SPtr edgeSet = New<EdgeSetTopologyContainer>();
    extension_node->addObject(edgeSet);
    edgeSet->addEdge(0,1);

    DistanceMapping31::SPtr extensionMapping = New<DistanceMapping31>();
    extensionMapping->setModels(mappedDOF.get(),extensions.get());
    extension_node->addObject( extensionMapping );
    extensionMapping->setName("InteractionExtension_mapping");


    UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
    extension_node->addObject(compliance);
    compliance->compliance.setName("connectionCompliance");
    compliance->compliance.setValue(complianceValue);

//    if( dampingRatio )
//    {
//        component::odesolver::ConstraintValue::SPtr constraintValue = New<component::odesolver::ConstraintValue>( extensions.get() );
//        constraintValue->dampingRatio.setValue(dampingRatio);
//        extension_node->addObject(constraintValue);
//    }

    return root;
}



/// Create a stiff string
simulation::Node::SPtr createStiffString(simulation::Node::SPtr parent, Vec3 startPoint, Vec3 endPoint, unsigned numParticles, double totalMass, double stiffnessValue=1.0, double dampingRatio=0 )
{
    static unsigned numObject = 1;
    std::ostringstream oss;
    oss << "string_" << numObject++;
    SReal totalLength = (endPoint-startPoint).norm();

    //--------
    Node::SPtr  string_node = parent->createChild(oss.str());

    MechanicalObject3::SPtr DOF = New<MechanicalObject3>();
    string_node->addObject(DOF);
    DOF->setName(oss.str()+"_DOF");

    UniformMass3::SPtr mass = New<UniformMass3>();
    string_node->addObject(mass);
    mass->setName(oss.str()+"_mass");
    mass->mass.setValue( totalMass/numParticles );

    StiffSpringForceField3::SPtr spring = New<StiffSpringForceField3>();
    string_node->addObject(spring);
    spring->setName(oss.str()+"_spring");



    //--------
    // create the particles and the springs
    DOF->resize(numParticles);
    MechanicalObject3::WriteVecCoord x = DOF->writePositions();
    for( unsigned i=0; i<numParticles; i++ )
    {
        double alpha = (double)i/(numParticles-1);
        x[i] = startPoint * (1-alpha)  +  endPoint * alpha;
        if(i>0)
        {
            spring->addSpring(i-1,i,stiffnessValue,dampingRatio,totalLength/(numParticles-1));
         }
    }

    return string_node;

}

template<class Component>
typename Component::SPtr addNew( Node::SPtr parentNode, std::string name="")
{
    typename Component::SPtr component = New<Component>();
    parentNode->addObject(component);
    component->setName(parentNode->getName()+"_"+name);
    return component;
}

/// Create a stiff hexehedral grid
simulation::Node::SPtr createStiffGrid(simulation::Node::SPtr parent, Vec3 startPoint, Vec3 endPoint, unsigned numX, unsigned numY, unsigned numZ, double totalMass, double stiffnessValue=1.0, double dampingRatio=0 )
{
    static unsigned numObject = 1;
    std::ostringstream oss;
    oss << "Grid_" << numObject++;

    //--------
    Node::SPtr  grid_node = parent->createChild(oss.str());

    RegularGridTopology::SPtr grid = addNew<RegularGridTopology>( grid_node, oss.str()+"_grid" );
    grid->setNumVertices(numX,numX,numZ);
    grid->setPos(startPoint[0],endPoint[0],startPoint[1],endPoint[1],startPoint[2],endPoint[2]);

    MechanicalObject3::SPtr DOF = addNew<MechanicalObject3>(grid_node, oss.str()+"_DOF" );

    UniformMass3::SPtr mass = addNew<UniformMass3>(grid_node, oss.str()+"_mass" );
    mass->mass.setValue( totalMass/(numX*numY*numZ) );

    RegularGridSpringForceField3::SPtr spring = addNew<RegularGridSpringForceField3>(grid_node, oss.str()+"_spring");
    spring->setLinesStiffness(stiffnessValue);
    spring->setLinesDamping(dampingRatio);

    return grid_node;
}





/// Create the stiff string composed of three parts
simulation::Node::SPtr createStiffScene()
{
    // The graph root node
    Node::SPtr  root = simulation::getSimulation()->createNewGraph("root");
    root->setGravity( Coord3(0,-1,0) );
    root->setAnimate(false);
    root->setDt(0.01);
    addVisualStyle(root)->setShowVisual(false).setShowCollision(false).setShowMapping(true).setShowBehavior(true);

    Node::SPtr simulatedScene = root->createChild("simulatedScene");


    EulerImplicitSolver::SPtr eulerImplicitSolver = New<EulerImplicitSolver>();
    simulatedScene->addObject( eulerImplicitSolver );
    CGLinearSolver::SPtr cgLinearSolver = New<CGLinearSolver>();
    simulatedScene->addObject(cgLinearSolver);


    // ========  first string
    unsigned n1 = 3;
    Node::SPtr  string1 = createStiffString( simulatedScene, Vec3(0,0,0), Vec3(1,0,0), n1, 1.0*n1, 1/complianceValue, dampingRatio );


    FixedConstraint3::SPtr fixed1 = New<FixedConstraint3>();
    string1->addObject( fixed1 );

    // ========  second string
    unsigned n2 = 3;
    Node::SPtr  string2 = createStiffString( simulatedScene, Vec3(3,0,0), Vec3(2,0,0), n2, 1.0*n2, 1/complianceValue, dampingRatio );
    FixedConstraint3::SPtr fixed2 = New<FixedConstraint3>();
    string2->addObject( fixed2 );


    // ========  Node with multiple parents to create an interaction using a MultiMapping
    Node::SPtr commonChild = string1->createChild("commonChild");
    string2->addChild(commonChild);

    MechanicalObject3::SPtr mappedDOF = New<MechanicalObject3>(); // to contain particles from the two strings
    commonChild->addObject(mappedDOF);

    SubsetMultiMapping3_to_3::SPtr multimapping = New<SubsetMultiMapping3_to_3>();
    multimapping->setName("InteractionMultiMapping");
    multimapping->addInputModel( string1->getMechanicalState() );
    multimapping->addInputModel( string2->getMechanicalState() );
    multimapping->addOutputModel( mappedDOF.get() );
    multimapping->addPoint( string1->getMechanicalState(), n1-1 );
    multimapping->addPoint( string2->getMechanicalState(), n2-1 );
    commonChild->addObject(multimapping);

    StiffSpringForceField3::SPtr spring = New<StiffSpringForceField3>();
    commonChild->addObject(spring);
    spring->setName("InteractionSpring");
    spring->addSpring(0,1,1/complianceValue,dampingRatio,1.0);

    return root;
}


int main(int argc, char** argv)
{

    sofa::helper::BackTrace::autodump();
    sofa::core::ExecParams::defaultInstance()->setAspectID(0);

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&startAnim,'a',"start","start the animation loop")
    .option(&simulationType,'s',"simu","select the type of simulation (bgl, tree)")
    .option(&verbose,'v',"verbose","print debug info")
    (argc,argv);
#ifndef SOFA_NO_OPENGL
    glutInit(&argc,argv);
#endif
//#ifdef SOFA_DEV
//    if (simulationType == "bgl")
//        sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
//    else
//#endif
//        sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());
#ifdef SOFA_HAVE_DAG
    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());
#endif

    sofa::component::init();
#ifdef SOFA_GPU_CUDA
#ifdef WIN32
#ifdef NDEBUG
    std::string name("sofagpucuda_1_0.dll");
#else
    std::string name("sofagpucuda_1_0d.dll");
#endif
    sofa::helper::system::DynamicLibrary::load(name);
#endif
#endif

    sofa::gui::initMain();
    if (int err = sofa::gui::GUIManager::Init(argv[0],"")) return err;
    if (int err=sofa::gui::GUIManager::createGUI(NULL)) return err;
    sofa::gui::GUIManager::SetDimension(800,600);

    //=================================================
    //    sofa::simulation::Node::SPtr groot = createStiffScene();
    sofa::simulation::Node::SPtr groot = createCompliantScene();
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



