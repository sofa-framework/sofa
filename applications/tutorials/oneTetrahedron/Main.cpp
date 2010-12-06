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
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/typedef/Sofa_typedef.h>
#include <sofa/component/visualmodel/OglModel.h>

#include <sofa/core/objectmodel/Context.h>

#include <sofa/gui/GUIManager.h>

#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/glut.h>

#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/TreeSimulation.h>

#include <iostream>
#include <fstream>

using namespace sofa::simulation::tree;
using sofa::simulation::Node;
using sofa::component::odesolver::EulerSolver;
using sofa::component::topology::MeshTopology;
using sofa::component::visualmodel::OglModel;
using sofa::core::objectmodel::Data;
using sofa::helper::ReadAccessor;
using sofa::helper::WriteAccessor;
using sofa::core::VecId;
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    glutInit(&argc,argv);
    sofa::helper::parse("This is a SOFA application.")
    (argc,argv);

    sofa::gui::GUIManager::Init(argv[0]);

    // The graph root node : gravity already exists in a GNode by default
    GNode* groot = new GNode;
    groot->setName( "root" );
    groot->setGravityInWorld( Coord3(0,-10,0) );

    // One solver for all the graph
    EulerSolver* solver = new EulerSolver;
    solver->setName("solver");
    solver->f_printLog.setValue(false);
    groot->addObject(solver);

    // Tetrahedron degrees of freedom
    MechanicalObject3* DOF = new MechanicalObject3;
    groot->addObject(DOF);
    DOF->resize(4);
    DOF->setName("DOF");
    //get write access to the position vector of mechanical object DOF
    WriteAccessor<Data<VecCoord3> > x = *DOF->write(VecId::position());

    x[0] = Coord3(0,10,0);
    x[1] = Coord3(10,0,0);
    x[2] = Coord3(-10*0.5,0,10*0.866);
    x[3] = Coord3(-10*0.5,0,-10*0.866);

    // Tetrahedron uniform mass
    UniformMass3* mass = new UniformMass3;
    groot->addObject(mass);
    mass->setMass(2);
    mass->setName("mass");

    // Tetrahedron topology
    MeshTopology* topology = new MeshTopology;
    topology->setName("mesh topology");
    groot->addObject( topology );
    topology->addTetra(0,1,2,3);

    // Tetrahedron constraints
    FixedConstraint3* constraints = new FixedConstraint3;
    constraints->setName("constraints");
    groot->addObject(constraints);
    constraints->addConstraint(0);

    // Tetrahedron force field
    TetrahedronFEMForceField3* fem = new  TetrahedronFEMForceField3;
    fem->setName("FEM");
    groot->addObject(fem);
    fem->setMethod("polar");
    fem->setUpdateStiffnessMatrix(true);
    fem->setYoungModulus(6);

    // Tetrahedron skin
    GNode* skin = new GNode("skin",groot);;

    // The visual model
    OglModel* visual = new OglModel();
    visual->setName( "visual" );
    visual->load(sofa::helper::system::DataRepository.getFile("mesh/liver-smooth.obj"), "", "");
    visual->setColor("red");
    visual->applyScale(0.7, 0.7, 0.7);
    visual->applyTranslation(1.2, 0.8, 0);
    skin->addObject(visual);

    // The mapping between the tetrahedron (DOF) and the liver (visual)
    BarycentricMapping3_to_Ext3* mapping = new BarycentricMapping3_to_Ext3(DOF, visual);
    mapping->setName( "mapping" );
    skin->addObject(mapping);

    // Init the scene
    sofa::simulation::tree::getSimulation()->init(groot);
    groot->setAnimate(false);
    groot->setShowNormals(false);
    groot->setShowInteractionForceFields(false);
    groot->setShowMechanicalMappings(false);
    groot->setShowCollisionModels(false);
    groot->setShowBoundingCollisionModels(false);
    groot->setShowMappings(false);
    groot->setShowForceFields(true);
    groot->setShowWireFrame(true);
    groot->setShowVisualModels(true);
    groot->setShowBehaviorModels(true);



    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(groot);

    return 0;
}
