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
#include "stdafx.h"
#include "Sofa_test.h"
#include <SofaComponentMain/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <plugins/SceneCreator/SceneCreator.h>


namespace sofa {
using namespace modeling;
using defaulttype::Vector3;
using core::objectmodel::New;

/** Test the Simulation class
*/
struct Simulation_test: public Sofa_test<double>
{
    // root
   simulation::Node::SPtr root;

   /// Test Simulation::computeBBox
   void computeBBox()
   {
       // Init Sofa
       sofa::component::init();
       simulation::Simulation* simulation;
       sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

       root = simulation::getSimulation()->createNewGraph("root");

       // create DOFs and its expected bounding box
       MechanicalObject3::SPtr DOF = New<MechanicalObject3>();
       root->addObject(DOF);
       DOF->resize(4);
       MechanicalObject3::WriteVecCoord x = DOF->writePositions();
       x[0] = Vector3(0,0,0);
       x[1] = Vector3(1,0,0);
       x[2] = Vector3(0,1,0);
       x[3] = Vector3(0,0,1);
       Vector3 expectedMin(0,0,0), expectedMax(1,1,1);
       DOF->showObject.setValue(true); // bbox is updated only for drawn MO

       // end create scene
       //*********
       initScene();
       //*********

       Vector3 sceneMinBBox, sceneMaxBBox;
       simulation->computeBBox(root.get(), sceneMinBBox.ptr(), sceneMaxBBox.ptr());

       if( vectorMaxDiff(sceneMinBBox,expectedMin)>this->epsilon() || vectorMaxDiff(sceneMaxBBox,expectedMax)>this->epsilon() )
       {
           ADD_FAILURE() << "Wrong bounding box, expected (" << expectedMin <<", "<<expectedMax<<") , got ("<< sceneMinBBox <<", "<<sceneMaxBBox << ")" << endl;
       }

   }
        

};

TEST_F( Simulation_test,SimulationTest)
{
     this->computeBBox();
}

}// namespace sofa







