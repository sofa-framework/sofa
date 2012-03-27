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
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/simulation/bgl/BglSimulation.h>
#include <sofa/simulation/common/Node.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/helper/system/FileRepository.h>


#include <sofa/component/mapping/SubsetMultiMapping.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/topology/CubeTopology.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/helper/vector.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
#include "../../../applications/tutorials/objectCreator/ObjectCreator.h"

#include <plugins/Compliant/ComplianceSolver.h>
#include <plugins/Compliant/UniformCompliance.h>
#include <plugins/Flexible/ExtensionMapping.h>

using namespace sofa;
using namespace sofa::helper;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using namespace sofa::component::container;
using namespace sofa::component::topology;
using namespace sofa::component::collision;
using namespace sofa::component::visualmodel;
using namespace sofa::component::mapping;
using namespace sofa::component::compliance;

typedef SReal Scalar;
typedef Vec<3,SReal> Vec3;
typedef Vec<1,SReal> Vec1;
typedef ExtensionMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> ExtensionMapping31;
typedef UniformCompliance<Vec1Types> UniformCompliance1;
typedef component::odesolver::ComplianceSolver ComplianceSolver;



/// Resize the target, then copy the source to the target
template <class V1, class V2>
void copyContainer( V1& target, const V2& source )
{
    target.resize( source.size() );
    std::copy(source.begin(),source.end(),target.begin());
}



/// Create a string
static simulation::Node::SPtr createString(simulation::Node::SPtr parent, Vec3 startPoint, Vec3 endPoint, unsigned numParticles, double totalMass, double complianceValue=0, double dampingRatio=0 )
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

    ExtensionMapping31::SPtr extensionMapping = New<ExtensionMapping31>();
    extensionMapping->setModels(DOF.get(),extensions.get());
    extension_node->addObject( extensionMapping );
    extensionMapping->setName(oss.str()+"_ExtensionMapping");
    extensionMapping->setModels( DOF.get(), extensions.get() );

    UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
    extension_node->addObject(compliance);
    compliance->setName(oss.str()+"_compliance");
    compliance->setCompliance(complianceValue);
    compliance->dampingRatio.setValue(dampingRatio);


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


    return string_node;

}





// ---------------------------------------------------------------------
int main( int argc, char** argv )
{


    glutInit(&argc,argv);
    sofa::gui::GUIManager::Init(argv[0]);

    std::vector<std::string> files;
    std::string simulationType="bgl";

    SReal complianceValue = 0.1;
    SReal dampingRatio = 0.1;

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&simulationType,'s',"simulation","type of the simulation(bgl,tree)")
    (argc,argv);

    if (simulationType == "bgl")
        sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
    else
        sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());


    //*************************************
    // BEGIN create the scene

    // The graph root node
    Node::SPtr  root = sofa::ObjectCreator::CreateRootWithCollisionPipeline(simulationType);
    root->setGravity( Coord3(0,-1,0) );
    root->setAnimate(false);
    root->setDt(0.001);
    addVisualStyle(root)->setShowVisual(false).setShowCollision(false).setShowMapping(true).setShowBehavior(true);

    ComplianceSolver::SPtr complianceSolver = New<ComplianceSolver>();
    root->addObject( complianceSolver );
    complianceSolver->implicitVelocity.setValue(1.0);
    complianceSolver->implicitPosition.setValue(1.0);
//    complianceSolver->verbose.setValue(true);


    // first string
    unsigned n1 = 2;
    Node::SPtr  string1 = createString( root, Vec3(0,0,0), Vec3(1,0,0), n1, 2.0, complianceValue, dampingRatio );
    FixedConstraint3::SPtr fixed1 = New<FixedConstraint3>();
    string1->addObject( fixed1 );

    // second string
    unsigned n2 = 2;
    Node::SPtr  string2 = createString( root, Vec3(3,0,0), Vec3(2,0,0), n2, 2.0, complianceValue, dampingRatio );
    FixedConstraint3::SPtr fixed2 = New<FixedConstraint3>();
    string2->addObject( fixed2 );

    // Node with multiple parents to create an interaction using a MultiMapping
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

    ExtensionMapping31::SPtr extensionMapping = New<ExtensionMapping31>();
    extensionMapping->setModels(mappedDOF.get(),extensions.get());
    extension_node->addObject( extensionMapping );
    extensionMapping->setName("InteractionExtension_mapping");


    UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
    extension_node->addObject(compliance);
    compliance->setCompliance(complianceValue);
    compliance->dampingRatio.setValue(dampingRatio);

    // END create the scene
    //*************************************


    getSimulation()->init(root.get());

    // Run the main loop
    sofa::gui::GUIManager::MainLoop(root);


    return 0;
}

