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

#include <SceneCreator/SceneCreator.h>

#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/CollisionModel.h>
#include <SofaSimulationTree/init.h>
#include <SofaSimulationTree/TreeSimulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/DeleteVisitor.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>

#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <SofaBaseCollision/MinProximityIntersection.h>
#include <SofaConstraint/LMConstraintSolver.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
using namespace sofa::simulation;
using namespace sofa::component::container;
using namespace sofa::component::topology;
typedef sofa::component::linearsolver::CGLinearSolver< sofa::component::linearsolver::GraphScatteredMatrix, sofa::component::linearsolver::GraphScatteredVector> CGLinearSolverGraph;

const std::string colors[7]= {"red","green","blue","cyan","magenta","yellow","white"};

SReal convertDegreeToRadian(const SReal& angle)
{
    const SReal pi=3.14159265;
    return angle*pi/180.0;
}

Node::SPtr createCard(Node::SPtr  parent, const Coord3& position, const Coord3& rotation)
{
    const std::string visualModel="mesh/card.obj";
    const std::string collisionModel="mesh/cardCollision.obj";
    const std::string inertiaMatrix="BehaviorModels/card.rigid";
    static int colorIdx=0;

    std::vector<std::string> modelTypes;
    modelTypes.push_back("Triangle");
    modelTypes.push_back("Line");
    modelTypes.push_back("Point");

    Node::SPtr  card = sofa::modeling::createEulerSolverNode(parent,"Rigid","Implicit");

    sofa::component::odesolver::EulerImplicitSolver *odeSolver; card->get(odeSolver);
    odeSolver->f_rayleighStiffness.setValue(0.1);
    odeSolver->f_rayleighMass.setValue(0.1);

    CGLinearSolverGraph *cgLinearSolver; card->get(cgLinearSolver);
    cgLinearSolver->f_maxIter.setValue(15);
    cgLinearSolver->f_tolerance.setValue(1e-5);
    cgLinearSolver->f_smallDenominatorThreshold.setValue(1e-5);


    sofa::component::constraintset::LMConstraintSolver::SPtr constraintSolver = sofa::core::objectmodel::New<sofa::component::constraintset::LMConstraintSolver>();
    constraintSolver->constraintVel.setValue(true);
    constraintSolver->constraintPos.setValue(true);
    constraintSolver->numIterations.setValue(25);

    card->addObject(constraintSolver);

    MechanicalObjectRigid3::SPtr dofRigid = sofa::core::objectmodel::New<MechanicalObjectRigid3>(); dofRigid->setName("Rigid Object");
    dofRigid->setTranslation(position[0],position[1],position[2]);
    dofRigid->setRotation(rotation[0],rotation[1],rotation[2]);
    card->addObject(dofRigid);

    UniformMassRigid3::SPtr uniMassRigid = sofa::core::objectmodel::New<UniformMassRigid3>();
    uniMassRigid->setTotalMass(0.5);
    uniMassRigid->setFileMass(inertiaMatrix);
    card->addObject(uniMassRigid);

    //Node VISUAL
    Node::SPtr  RigidVisualNode = sofa::modeling::createVisualNodeRigid(card, dofRigid.get(), visualModel,colors[(colorIdx++)%7]);

    //Node COLLISION
    Node::SPtr  RigidCollisionNode = sofa::modeling::createCollisionNodeRigid(card, dofRigid.get(),collisionModel,modelTypes);

    return card;
}

std::pair<Node::SPtr , Node::SPtr  > create2Cards(Node::SPtr  parent, const Coord3& globalPosition, SReal distanceInBetween=SReal(0.1), SReal angle=SReal(15.0))
{
    //We assume the card has a 2 unit length, centered in (0,0,0)
    const SReal displacement=sin(convertDegreeToRadian(angle));
    const Coord3 separation=Coord3(displacement+distanceInBetween,0,0);

    //************************************
    //Left Rigid Card
    const Coord3 positionLeft=globalPosition + separation;
    const Coord3 rotationLeft(0,0,angle);
    Node::SPtr  leftCard = createCard(parent, positionLeft, rotationLeft);

    //************************************
    //Right Rigid Card
    const Coord3 positionRight=globalPosition - separation;
    const Coord3 rotationRight(0,0,-angle);
    Node::SPtr  rightCard = createCard(parent, positionRight, rotationRight);

    return std::make_pair(leftCard, rightCard);
}

Node::SPtr createHouseOfCards(Node::SPtr root,  unsigned int size, SReal distanceInBetween=SReal(0.1), SReal angle=SReal(15.0))
{

    //Elements of the scene
    //------------------------------------
    Node::SPtr  houseOfCards = root;


    //Space between two levels of the house of cards
    const SReal space=0.75;
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
            const std::pair<Node::SPtr , Node::SPtr > &cards=create2Cards(houseOfCards, position,distanceInBetween, angle);
            std::ostringstream out;
            out << "Card["<< i << "|" << j << "]";
            cards.first->setName("Left"+out.str());
            cards.second->setName("Right"+out.str());
        }

        //Create the support for the cards
        const Coord3 initPosition(0,sizeCard*0.5,0);
        const Coord3 supportRotation(0,0,90);
        for (unsigned int j=0; j<i; ++j)
        {
            Coord3 position((i+j)*distanceH*0.5,
                    (i-j)*distanceV-space*0.5+distanceInBetween*(j%2),
                    0);
            Node::SPtr supportCard=createCard(houseOfCards, position-initPosition, supportRotation);
            std::ostringstream out;
            out << "SupportCard["<< i << "|" << j << "]";
            supportCard->setName(out.str());
        }
    }



    return root;
}



int main(int argc, char** argv)
{
    sofa::simulation::tree::init();
    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();
    sofa::gui::initMain();

    unsigned int sizeHouseOfCards=4;
    SReal angle=20.0;
    SReal distanceInBetween=0.1;
    SReal friction=0.8;
    SReal contactDistance=0.03;
    std::string gui = "";


    std::string gui_help = "choose the UI (";
    gui_help += sofa::gui::GUIManager::ListSupportedGUI('|');
    gui_help += ")";

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&sizeHouseOfCards,'l',"level","number of level of the house of cards")
    .option(&angle,'a',"angle","angle formed by two cards")
    .option(&distanceInBetween,'d',"distance","distance between two cards")
    .option(&friction,'f',"friction","friction coeff")
    .option(&contactDistance,'c',"contactDistance","contact distance")
    .option(&gui,'g',"gui",gui_help.c_str())
    (argc,argv);


    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());

    // The graph root node
    Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline("distanceLMConstraint");
    root->setGravity( Coord3(0,-10,0) );
    root->setDt(0.001);

    sofa::component::collision::MinProximityIntersection *intersection;
    root->get(intersection, sofa::core::objectmodel::BaseContext::SearchDown);
    intersection->alarmDistance.setValue(contactDistance);
    intersection->contactDistance.setValue(contactDistance*0.5);

    //************************************
    //Floor
    Node::SPtr  torusFixed = sofa::modeling::createObstacle(root.get(),"mesh/floor.obj", "mesh/floor.obj", "gray");

    //Add the objects
    createHouseOfCards(root.get(),sizeHouseOfCards,distanceInBetween, angle);


    const SReal contactFriction=sqrt(friction);
    sofa::helper::vector< sofa::core::CollisionModel* > listCollisionModels;
    root->getTreeObjects<sofa::core::CollisionModel>(&listCollisionModels);
    for (unsigned int i=0; i<listCollisionModels.size(); ++i) listCollisionModels[i]->setContactFriction(contactFriction);
    root->setAnimate(false);

    //=======================================
    // Export the scene to file
    const std::string fileName="HouseOfCards.xml";
    sofa::simulation::getSimulation()->exportXML(root.get(),fileName.c_str());

    //=======================================
    // Destroy created scene: step needed, as I can't get rid of the locales (the mass can't init correctly as 0.1 is not considered as a floating point).
    sofa::simulation::DeleteVisitor deleteScene(sofa::core::ExecParams::defaultInstance() );
    root->execute(deleteScene);
    root.reset();

    //=======================================
    // Create the GUI
    if (int err=sofa::gui::GUIManager::Init(argv[0],gui.c_str()))
        return err;

    if (int err=sofa::gui::GUIManager::createGUI(NULL))
        return err;

    sofa::gui::GUIManager::SetDimension(800,600);
    //=======================================
    // Load the Scene

    sofa::simulation::Node::SPtr groot = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()));
    if (groot==NULL)
    {
        groot = sofa::simulation::getSimulation()->createNewGraph("");
    }

    sofa::simulation::getSimulation()->init(groot.get());
    sofa::gui::GUIManager::SetScene(groot,fileName.c_str());


    //=======================================
    // Run the main loop
    if (int err=sofa::gui::GUIManager::MainLoop(groot,fileName.c_str()))
        return err;
    Node* ggroot =  sofa::gui::GUIManager::CurrentSimulation();


    if (ggroot!=NULL) sofa::simulation::getSimulation()->unload(groot);

    sofa::simulation::tree::cleanup();
    return 0;
}
