/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/VecId.h>
#include <sofa/gui/GUIManager.h>

#include <sofa/helper/system/glut.h>
#include <sofa/helper/accessor.h>



using namespace sofa::simulation::tree;
using sofa::component::odesolver::EulerSolver;
using sofa::core::objectmodel::Data;
using sofa::helper::ReadAccessor;
using sofa::helper::WriteAccessor;
using sofa::core::VecId;

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{

    glutInit(&argc,argv);
    sofa::helper::parse("This is a SOFA application.")
    (argc,argv);
    sofa::gui::GUIManager::Init(argv[0]);

    // The graph root node
    GNode* groot = new GNode;
    groot->setName( "root" );
    groot->setGravityInWorld( Coord3(0,-10,0) );

    // One solver for all the graph
    EulerSolver* solver = new EulerSolver;
    solver->setName("solver");
    solver->f_printLog.setValue(false);
    groot->addObject(solver);

    // One node to define the particle
    GNode* particule_node = new GNode("particle_node", groot);

    // The particule, i.e, its degrees of freedom : a point with a velocity
    MechanicalObject3* particle = new MechanicalObject3;
    particle->setName("particle");
    particule_node->addObject(particle);
    particle->resize(1);
    // get write access the particle positions vector
    WriteAccessor< Data<MechanicalObject3::VecCoord> > positions = *particle->write( VecId::position() );
    positions[0] = Coord3(0,0,0);
    // get write access the particle velocities vector
    WriteAccessor< Data<MechanicalObject3::VecDeriv> > velocities = *particle->write( VecId::velocity() );
    velocities[0] = Deriv3(0,0,0);

    // Its properties, i.e, a simple mass node
    UniformMass3* mass = new UniformMass3;
    mass->setName("mass");
    particule_node->addObject(mass);
    mass->setMass( 1 );

    sofa::simulation::tree::getSimulation()->init(groot);
    groot->setAnimate(false);
    groot->setShowBehaviorModels(true);

    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(groot);

    return 0;
}
