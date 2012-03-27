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
#include <sofa/gui/GUIManager.h>


#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/gui/SofaGUI.h>


//Including component for topological description of the objects
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/RegularGridTopology.h>


//SOFA_HAS_BOOST_KERNEL to define in chainHybrid.pro

#include <sofa/simulation/bgl/BglSimulation.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/component/visualmodel/OglModel.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>



using sofa::component::visualmodel::OglModel;
using namespace sofa::simulation;
using namespace sofa::component::forcefield;
using namespace sofa::component::collision;
using namespace sofa::component::topology;

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

Node::SPtr  createRegularGrid(Node::SPtr  parent, double x, double y, double z)
{
    static unsigned int i = 1;
    std::ostringstream oss;
    oss << "regularGrid_" << i++;

    Node::SPtr  node =parent->createChild(oss.str()) ;

    RegularGridTopology::SPtr grid = New<RegularGridTopology>(3,3,3);
    grid->setPos(-1+x,1+x,-1+y,1+y,-1+z,1+z);
    MechanicalObject3::SPtr  dof = New<MechanicalObject3>();

    UniformMass3::SPtr  mass = New<UniformMass3>();
    mass->setTotalMass(10);

    HexahedronFEMForceField3::SPtr  ff = New<HexahedronFEMForceField3>();
    ff->setYoungModulus(400);
    ff->setPoissonRatio(0.3);
    ff->setMethod(1);

    node->addObject(dof);
    node->addObject(mass);
    node->addObject(grid);
    node->addObject(ff);

    const Deriv3 translation(x,y,z);

    //Node VISUAL
    Node::SPtr  VisualNode = sofa::ObjectCreator::CreateVisualNodeVec3(node,dof,"mesh/ball.obj", "red", translation);

    //node->setShowBehaviorModels(true);
    return node;
}


int main(int argc, char** argv)
{
    glutInit(&argc,argv);

    std::vector<std::string> files;

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
//       .option(&simulationType,'s',"simulation","type of the simulation(bgl,tree)")
    (argc,argv);

    sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
    sofa::gui::GUIManager::Init(argv[0]);

    // The graph root node
    Node::SPtr  root = sofa::ObjectCreator::CreateRootWithCollisionPipeline("bgl");
    root->setGravity( Coord3(0,0,0) );
    sofa::component::visualmodel::VisualStyle::SPtr visualStyle = New<sofa::component::visualmodel::VisualStyle>();
    root->addObject(visualStyle);
    visualStyle->displayFlags.setValue( sofa::core::visual::DisplayFlags().setShowForceFields().setShowBehaviorModels());

    Node::SPtr  solverNode = sofa::ObjectCreator::CreateEulerSolverNode(root,"Solver");

    Node::SPtr  grid1 = createRegularGrid(solverNode,-1.5,0,0);
    Node::SPtr  grid2 = createRegularGrid(solverNode,1.5,0,0);


    MechanicalObject3::SPtr  subsetDof = New<MechanicalObject3>();
    SubsetMultiMapping3_to_3::SPtr subsetMultiMapping = New<SubsetMultiMapping3_to_3>();
    MechanicalObject3*  input1 = dynamic_cast<MechanicalObject3* >(grid2->getMechanicalState());
    MechanicalObject3*  input2 = dynamic_cast<MechanicalObject3* >(grid1->getMechanicalState());

    input1->f_printLog.setValue(true);
    input1->setName("input1");
    input2->f_printLog.setValue(true);
    input2->setName("input2");
    subsetDof->f_printLog.setValue(true);
    subsetDof->setName("subsetDof");

    subsetMultiMapping->addInputModel( input1 );
    subsetMultiMapping->addInputModel( input2 );
    subsetMultiMapping->addOutputModel( subsetDof.get() );

    subsetMultiMapping->addPoint( input1, 21);
    subsetMultiMapping->addPoint( input1, 18);
    subsetMultiMapping->addPoint( input1, 9);
    subsetMultiMapping->addPoint( input1, 12);

    subsetMultiMapping->addPoint( input2, 11);
    subsetMultiMapping->addPoint( input2, 20);
    subsetMultiMapping->addPoint( input2, 14);
    subsetMultiMapping->addPoint( input2, 23);


    MeshTopology::SPtr topology = New<MeshTopology>();

    topology->addHexa(4,2,3,6,5,1,0,7);

    HexahedronFEMForceField3::SPtr  ff = New<HexahedronFEMForceField3>();
    ff->setYoungModulus(400);
    ff->setPoissonRatio(0.3);
    ff->setMethod(1);


    Node::SPtr  multiParentsNode = grid1->createChild("MultiParents");
    grid2->addChild(multiParentsNode);

    multiParentsNode->addObject(topology);
    multiParentsNode->addObject(subsetDof);
    multiParentsNode->addObject(subsetMultiMapping);

    multiParentsNode->addObject(ff);

    //multiParentsNode->setShowForceFields(true);


    root->setAnimate(false);

    getSimulation()->init(root.get());


    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(root);

    return 0;
}
