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

#include "../../../../tutorials/objectCreator/ObjectCreator.h"

#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/simulation/bgl/BglSimulation.h>
#include <sofa/simulation/common/Node.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/helper/system/FileRepository.h>


#include <sofa/component/container/MeshLoader.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/collision/SphereModel.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
using namespace sofa::simulation;
using namespace sofa::component::container;
using namespace sofa::component::topology;
using namespace sofa::component::collision;

Node *createChainHybrid(Node *root)
{
    const std::string visualModel="mesh/torus.obj";
    const std::string collisionModel="mesh/torus_for_collision.obj";

    std::vector<std::string> modelTypes;
    modelTypes.push_back("Triangle");
    modelTypes.push_back("Line");
    modelTypes.push_back("Point");

    //Elements of the scene
    //------------------------------------
    Node* chain = getSimulation()->newNode("Chain");
    root->addChild(chain);


    //************************************
    //Torus Fixed

    Node* torusFixed = sofa::ObjectCreator::CreateObstacle("mesh/torus_for_collision.obj", "mesh/torus.obj", "gray");
    chain->addChild(torusFixed);

    //************************************
    //Torus FEM

    Node* torusFEM = sofa::ObjectCreator::CreateEulerSolverNode("FEM");
    chain->addChild(torusFEM);

    MeshLoader* loaderFEM = new MeshLoader;
    loaderFEM->load(sofa::helper::system::DataRepository.getFile("mesh/torus_low_res.msh").c_str());
    torusFEM->addObject(loaderFEM);

    MeshTopology* meshTorusFEM = new MeshTopology;
    torusFEM->addObject(meshTorusFEM);

    const Deriv3 translationFEM(2.5,0,0);
    const Deriv3 rotationFEM(90,0,0);

    MechanicalObject3* dofFEM = new MechanicalObject3; dofFEM->setName("FEM Object");
    dofFEM->setTranslation(translationFEM[0],translationFEM[1],translationFEM[2]);
    dofFEM->setRotation(rotationFEM[0],rotationFEM[1],rotationFEM[2]);
    torusFEM->addObject(dofFEM);

    UniformMass3* uniMassFEM = new UniformMass3;
    uniMassFEM->setTotalMass(5); //the whole object will have 5 as given mass
    torusFEM->addObject(uniMassFEM);

    TetrahedronFEMForceField3* tetraFEMFF = new TetrahedronFEMForceField3;
    tetraFEMFF->setName("FEM");
    tetraFEMFF->setComputeGlobalMatrix(false);
    tetraFEMFF->setMethod("large");
    tetraFEMFF->setPoissonRatio(0.3);
    tetraFEMFF->setYoungModulus(1000);
    torusFEM->addObject(tetraFEMFF);

    //Node VISUAL
    Node* FEMVisualNode = sofa::ObjectCreator::CreateVisualNodeVec3(dofFEM,visualModel, "red", translationFEM, rotationFEM);
    torusFEM->addChild(FEMVisualNode);

    //Node COLLISION
    Node* FEMCollisionNode = sofa::ObjectCreator::CreateCollisionNodeVec3(dofFEM,collisionModel, modelTypes, translationFEM, rotationFEM );
    torusFEM->addChild(FEMCollisionNode);

    //************************************
    //Torus Spring

    Node* torusSpring = sofa::ObjectCreator::CreateEulerSolverNode("Spring");
    chain->addChild(torusSpring);

    MeshLoader* loaderSpring = new MeshLoader;
    loaderSpring->load(sofa::helper::system::DataRepository.getFile("mesh/torus_low_res.msh").c_str());
    torusSpring->addObject(loaderSpring);
    loaderSpring->init();

    MeshTopology* meshTorusSpring = new MeshTopology;
    torusSpring->addObject(meshTorusSpring);

    const Deriv3 translationSpring(5,0,0);
    const Deriv3 rotationSpring(0,0,0);

    MechanicalObject3* dofSpring = new MechanicalObject3; dofSpring->setName("Spring Object");

    dofSpring->setTranslation(translationSpring[0],translationSpring[1],translationSpring[2]);
    dofSpring->setRotation(rotationSpring[0],rotationSpring[1],rotationSpring[2]);

    torusSpring->addObject(dofSpring);

    UniformMass3* uniMassSpring = new UniformMass3;
    uniMassSpring->setTotalMass(5); //the whole object will have 5 as given mass
    torusSpring->addObject(uniMassSpring);

    MeshSpringForceField3* springFF = new MeshSpringForceField3;
    springFF->setName("Springs");
    springFF->setStiffness(400);
    springFF->setDamping(0);
    torusSpring->addObject(springFF);


    //Node VISUAL
    Node* SpringVisualNode = sofa::ObjectCreator::CreateVisualNodeVec3(dofSpring, visualModel,"green", translationSpring, rotationSpring);
    torusSpring->addChild(SpringVisualNode);

    //Node COLLISION
    Node* SpringCollisionNode = sofa::ObjectCreator::CreateCollisionNodeVec3(dofSpring, collisionModel, modelTypes, translationSpring, rotationSpring);
    torusSpring->addChild(SpringCollisionNode);


    //************************************
    //Torus FFD

    Node* torusFFD = sofa::ObjectCreator::CreateEulerSolverNode("FFD");
    chain->addChild(torusFFD);

    const Deriv3 translationFFD(7.5,0,0);
    const Deriv3 rotationFFD(90,0,0);

    MechanicalObject3* dofFFD = new MechanicalObject3; dofFFD->setName("FFD Object");
    dofFFD->setTranslation(translationFFD[0],translationFFD[1],translationFFD[2]);
    dofFFD->setRotation(rotationFFD[0],rotationFFD[1],rotationFFD[2]);
    torusFFD->addObject(dofFFD);

    UniformMass3* uniMassFFD = new UniformMass3;
    uniMassFFD->setTotalMass(5); //the whole object will have 5 as given mass
    torusFFD->addObject(uniMassFFD);

    RegularGridTopology* gridTopo = new RegularGridTopology(6,2,5); //dimension of the grid
    gridTopo->setPos(
        -2.5,2.5,  //Xmin, Xmax
        -0.5,0.5,  //Ymin, Ymax
        -2,2       //Zmin, Zmax
    );
    torusFFD->addObject(gridTopo);

    RegularGridSpringForceField3* FFDFF = new RegularGridSpringForceField3;
    FFDFF->setName("Springs FFD");
    FFDFF->setStiffness(200);
    FFDFF->setDamping(0);
    torusFFD->addObject(FFDFF);

    //Node VISUAL
    Node* FFDVisualNode = sofa::ObjectCreator::CreateVisualNodeVec3(dofFFD, visualModel,"yellow", translationFFD);
    torusFFD->addChild(FFDVisualNode);

    //Node COLLISION
    Node* FFDCollisionNode = sofa::ObjectCreator::CreateCollisionNodeVec3(dofFFD,collisionModel,  modelTypes, translationFFD);
    torusFFD->addChild(FFDCollisionNode);


    //************************************
    //Torus Rigid

    Node* torusRigid = sofa::ObjectCreator::CreateEulerSolverNode("Rigid");
    chain->addChild(torusRigid);

    const Deriv3 translationRigid(10,0,0);
    const Deriv3 rotationRigid(0,0,0);

    MechanicalObjectRigid3* dofRigid = new MechanicalObjectRigid3; dofRigid->setName("Rigid Object");
    dofRigid->setTranslation(translationRigid[0],translationRigid[1],translationRigid[2]);
    dofRigid->setRotation(rotationRigid[0],rotationRigid[1],rotationRigid[2]);
    torusRigid->addObject(dofRigid);

    UniformMassRigid3* uniMassRigid = new UniformMassRigid3;
    uniMassRigid->setTotalMass(1); //the whole object will have 5 as given mass
    torusRigid->addObject(uniMassRigid);

    //Node VISUAL
    Node* RigidVisualNode = sofa::ObjectCreator::CreateVisualNodeRigid(dofRigid, visualModel,"gray");
    torusRigid->addChild(RigidVisualNode);

    //Node COLLISION
    Node* RigidCollisionNode = sofa::ObjectCreator::CreateCollisionNodeRigid(dofRigid,collisionModel, modelTypes);
    torusRigid->addChild(RigidCollisionNode);


    //************************************
    //Multi Mapping
    Node* MultiParentsNode = getSimulation()->newNode("MultiParentsNode");
    MultiParentsNode->setShowCollisionModels(false);

    torusFEM->addChild(MultiParentsNode);
    torusFFD->addChild(MultiParentsNode);
    torusSpring->addChild(MultiParentsNode);

    MechanicalObject3* dofMultiMapping = new MechanicalObject3; dofMultiMapping->setName("Center Of Mass");
    MultiParentsNode->addObject(dofMultiMapping);
    CenterOfMassMultiMapping3_to_3* multiMappingCOM = new CenterOfMassMultiMapping3_to_3;
    multiMappingCOM->addInputModel(dofFEM);
    multiMappingCOM->addInputModel(dofFFD);
    multiMappingCOM->addInputModel(dofSpring);
    multiMappingCOM->addOutputModel(dofMultiMapping);

    MultiParentsNode->addObject(multiMappingCOM);

    SphereModel *spheres=new SphereModel;
    spheres->setRadius(0.5);
    MultiParentsNode->addObject( spheres);

    MultiParentsNode->setShowCollisionModels(true);
    return root;
}



int main(int argc, char** argv)
{
    glutInit(&argc,argv);

    std::vector<std::string> files;
    std::string simulationType="tree";

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&simulationType,'s',"simulation","type of the simulation(bgl,tree)")
    (argc,argv);

    if (simulationType == "bgl")
        sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
    else
        sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());

    sofa::gui::GUIManager::Init(argv[0]);

    // The graph root node
    Node* root = sofa::ObjectCreator::CreateRootWithCollisionPipeline(simulationType);
    root->setGravity( Coord3(0,0,-10) );

    //Add the objects
    createChainHybrid(root);

    root->setAnimate(false);

    getSimulation()->init(root);


    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(root);

    return 0;
}
