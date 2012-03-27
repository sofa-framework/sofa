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

#include "../../../../tutorials/objectCreator/ObjectCreator.h"

#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/simulation/bgl/BglSimulation.h>
#include <sofa/simulation/common/Node.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/helper/system/FileRepository.h>


#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/topology/CubeTopology.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/helper/vector.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

using namespace sofa::simulation;
using namespace sofa::component::container;
using namespace sofa::component::topology;
using namespace sofa::component::collision;
using namespace sofa::component::visualmodel;
using sofa::core::objectmodel::New;

Node::SPtr  createCube(Node::SPtr  parent, double dx, double dy, double dz)
{
    static int i = 1;
    std::ostringstream oss;
    oss << "cube_" << i++;

    Node::SPtr  cube_node = parent->createChild(oss.str());
    MechanicalObject3::SPtr DOF = New<MechanicalObject3>();
    cube_node->addObject(DOF);
    DOF->setName("cube");
    DOF->setTranslation(dx,dy,dz);

    CubeTopology::SPtr cubeTopology = New<CubeTopology>(2,2,2);
    cubeTopology->setPos(-1,1,-1,1,-1,1);
    cube_node->addObject(cubeTopology);


    TriangleFEMForceField3::SPtr triangleFEM = New<TriangleFEMForceField3>();
    triangleFEM->setName("FEM");
    //triangleFEM->setComputeGlobalMatrix(false);
    triangleFEM->setMethod(0);
    triangleFEM->setPoisson(0.3);
    triangleFEM->setYoung(500);
    cube_node->addObject(triangleFEM);


    UniformMass3::SPtr uniMassCube = New<UniformMass3>();
    uniMassCube->setTotalMass(1);
    cube_node->addObject(uniMassCube);


    return cube_node;
}



int main( int argc, char** argv )
{
    glutInit(&argc,argv);

    std::vector<std::string> files;
    bool implicit=true;
    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&implicit,'i',"implicit","Implicit Integration Scheme")
    (argc,argv);

    sofa::simulation::setSimulation( new sofa::simulation::bgl::BglSimulation() );
    sofa::gui::GUIManager::Init(argv[0]);

    // The graph root node
    Node::SPtr root = sofa::ObjectCreator::CreateRootWithCollisionPipeline("bgl");
    root->setGravity( Coord3(0,0,0) );

//  VisualStyle::SPtr visualStyle = New<sofa::component::visualmodel::VisualStyle>();
//  root->addObject(visualStyle);
//  VisualStyle::DisplayFlags displayFlags;
//  displayFlags.setShowAll();
//  visualStyle->displayFlags.setValue(displayFlags);

    addVisualStyle(root)->setShowVisual().setShowBehavior().setShowMapping(false);

    std::string scheme="Implicit";

    if (!implicit) scheme="Explicit";

    Node::SPtr SolverNode = sofa::ObjectCreator::CreateEulerSolverNode(root,"SolverNode", scheme);

    Node::SPtr  cube1 = createCube(SolverNode, 0,0,0   );
    Node::SPtr  cube2 = createCube(SolverNode,  10,0,0  );
    Node::SPtr  cube3 = createCube(SolverNode,  0,0,10  );
    Node::SPtr  cube4 = createCube(SolverNode,  10,0,10 );
    Node::SPtr  MultiParentsNode = cube1->createChild("MultiParentsNode");
    cube2->addChild(MultiParentsNode);
    cube3->addChild(MultiParentsNode);
    cube4->addChild(MultiParentsNode);

    MechanicalObject3::SPtr dofMultiMapping = New<MechanicalObject3>();
    dofMultiMapping->setName("Center Of Mass");

    MultiParentsNode->addObject(dofMultiMapping);

    CenterOfMassMultiMapping3_to_3::SPtr multiMappingCOM = New< CenterOfMassMultiMapping3_to_3>();
    multiMappingCOM->addInputModel( dynamic_cast<MechanicalObject3*>(cube1->getMechanicalState()) );
    multiMappingCOM->addInputModel( dynamic_cast<MechanicalObject3*>(cube2->getMechanicalState()) );
    multiMappingCOM->addInputModel( dynamic_cast<MechanicalObject3*>(cube3->getMechanicalState()) );
    multiMappingCOM->addInputModel( dynamic_cast<MechanicalObject3*>(cube4->getMechanicalState()) );
    multiMappingCOM->addOutputModel(dofMultiMapping.get());


    MultiParentsNode->addObject(multiMappingCOM);

    ConstantForceField3::SPtr constantFF = New<ConstantForceField3>();
    constantFF->setForce( 0,MechanicalObject3::Deriv(0,10,0) );
    MultiParentsNode->addObject(constantFF) ;
    SphereModel::SPtr aSphere = New<SphereModel>();
    MultiParentsNode->addObject( aSphere );

    //cube1->addChild(MultiParentsNode);


    //MultiParentsNode->setShowCollisionModels(true);


    root->setAnimate(false);


    getSimulation()->init(root.get());


    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(root);

    return 0;
}

