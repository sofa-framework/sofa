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

#include "../objectCreator/ObjectCreator.h"

#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/TreeSimulation.h>
#ifdef SOFA_DEV
#include <sofa/simulation/bgl/BglSimulation.h>
#endif
#include <sofa/simulation/common/Node.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/component/collision/MinProximityIntersection.h>
#include <sofa/component/constraintset/LMConstraintSolver.h>
#include <sofa/core/CollisionModel.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
using namespace sofa::simulation;
using namespace sofa::component::container;
using namespace sofa::component::topology;

SReal convertDegreeToRadian(const SReal& angle)
{
    const SReal PI=3.14159265;
    return angle*PI/180.0;
}

Node *createCard(const Coord3& position, const Coord3& rotation)
{
    const std::string visualModel="mesh/card.obj";
    const std::string collisionModel="mesh/card.obj";
    const std::string inertiaMatrix="BehaviorModels/card.rigid";

    std::vector<std::string> modelTypes;
    modelTypes.push_back("Triangle");
    modelTypes.push_back("Line");
    modelTypes.push_back("Point");

    Node* card = sofa::ObjectCreator::CreateEulerSolverNode("Rigid");
    sofa::component::constraintset::LMConstraintSolver *constraintSolver = new sofa::component::constraintset::LMConstraintSolver();
    constraintSolver->constraintVel.setValue(true);
    constraintSolver->constraintPos.setValue(true);
//  constraintSolver->numIterations.setValue(20);

    card->addObject(constraintSolver);

    MechanicalObjectRigid3* dofRigid = new MechanicalObjectRigid3; dofRigid->setName("Rigid Object");
    dofRigid->setTranslation(position[0],position[1],position[2]);
    dofRigid->setRotation(rotation[0],rotation[1],rotation[2]);
    card->addObject(dofRigid);

    UniformMassRigid3* uniMassRigid = new UniformMassRigid3;
    uniMassRigid->setFileMass(inertiaMatrix);
    card->addObject(uniMassRigid);

    //Node VISUAL
    Node* RigidVisualNode = sofa::ObjectCreator::CreateVisualNodeRigid(dofRigid, visualModel,"red");
    card->addChild(RigidVisualNode);

    //Node COLLISION
    Node* RigidCollisionNode = sofa::ObjectCreator::CreateCollisionNodeRigid(dofRigid,collisionModel,modelTypes);
    card->addChild(RigidCollisionNode);

    return card;
}

Node *create2Cards(const Coord3& globalPosition, SReal distanceInBetween=SReal(0.1), SReal angle=SReal(15.0))
{
    //We assume the card has a one unit length
    const SReal displacement=sin(convertDegreeToRadian(angle));
    const Coord3 separation=Coord3(displacement+distanceInBetween,0,0);
    Node *TwoCards=sofa::simulation::getSimulation()->newNode("TwoCards");

    //************************************
    //Left Rigid Card
    {
        const Coord3 position=globalPosition + separation;
        const Coord3 rotation(0,0,angle);
        Node* leftCard = createCard(position, rotation);
        TwoCards->addChild(leftCard);
    }
    //************************************
    //Right Rigid Card
    {
        const Coord3 position=globalPosition - separation;
        const Coord3 rotation(0,0,-angle);
        Node* rightCard = createCard(position, rotation);
        TwoCards->addChild(rightCard);
    }
    return TwoCards;
}

Node *createHouseOfCards(Node *root,  unsigned int size, SReal distanceInBetween=SReal(0.1), SReal angle=SReal(15.0))
{

    //Elements of the scene
    //------------------------------------
    Node* houseOfCards = getSimulation()->newNode("HouseOfCards");
    root->addChild(houseOfCards);

    //************************************
    //Floor
    {
        Node* torusFixed = sofa::ObjectCreator::CreateObstacle("mesh/floor.obj", "mesh/floor.obj", "gray");
        houseOfCards->addChild(torusFixed);
    }

    //Space between two levels of the house of cards
    const SReal space=0.5;
    //Size of a card
    const SReal sizeCard=2;
    //overlap of the cards
    const SReal factor=0.95;
    const SReal distanceH=sizeCard*factor;
    const SReal distanceV=sizeCard*cos(convertDegreeToRadian(angle))+space;

    for (unsigned int i=0; i<size; ++i)
    {

        //Create the 2Cards
        for (unsigned int j=0; j<=i; ++j)
        {
            Coord3 position=Coord3((i+j)*(distanceH)*0.5,
                    (i-j)*(distanceV),
                    0);
            Node *cards=create2Cards(position,distanceInBetween, angle);
            houseOfCards->addChild(cards);
        }

        //Create the support for the cards
        const Coord3 initPosition(0,sizeCard*0.5,0);
        const Coord3 supportRotation(0,0,90);
        for (unsigned int j=0; j<i; ++j)
        {
            Coord3 position((i+j)*distanceH*0.5,
                    (i-j)*distanceV-space*0.5+distanceInBetween*(j%2),
                    0);
            Node *supportCard=createCard(position-initPosition, supportRotation);
            houseOfCards->addChild(supportCard);
        }
    }



    return root;
}



int main(int argc, char** argv)
{
#ifndef WIN32
    // Reset local settings to make sure that floating-point values are interpreted correctly
    setlocale(LC_ALL,"C");
    setlocale(LC_NUMERIC,"C");
#endif

    glutInit(&argc,argv);


    std::string simulationType="tree";
    unsigned int sizeHouseOfCards=4;

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&simulationType,'s',"simulation","type of the simulation(bgl,tree)")
    .option(&sizeHouseOfCards,'l',"level","number of level of the house of cards")
    (argc,argv);

#ifdef SOFA_DEV
    if (simulationType == "bgl")
        sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
    else
#endif
        sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());

    sofa::gui::GUIManager::Init(argv[0]);

    // The graph root node
    Node* root = sofa::ObjectCreator::CreateRootWithCollisionPipeline(simulationType,"distanceLMConstraint");
    root->setGravityInWorld( Coord3(0,-10,0) );
    root->setDt(0.001);

    sofa::component::collision::MinProximityIntersection *intersection;
    root->get(intersection, sofa::core::objectmodel::BaseContext::SearchDown);
    const SReal contactD=0.02;
    intersection->alarmDistance.setValue(contactD);
    intersection->contactDistance.setValue(contactD*0.5);

    //Add the objects
    createHouseOfCards(root,sizeHouseOfCards);

    sofa::helper::vector< sofa::core::CollisionModel* > listCollisionModels;
    root->getTreeObjects<sofa::core::CollisionModel>(&listCollisionModels);
    for (unsigned int i=0; i<listCollisionModels.size(); ++i) listCollisionModels[i]->setContactFriction(0.95);
    root->setAnimate(false);

    sofa::simulation::getSimulation()->exportXML(root,"HouseOfCards.xml", true);

    getSimulation()->init(root);


    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(root);

    return 0;
}
