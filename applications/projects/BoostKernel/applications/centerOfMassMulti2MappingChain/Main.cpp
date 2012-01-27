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


#include <sofa/component/loader/MeshGmshLoader.h>
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
using namespace sofa::component::loader;
using namespace sofa::component::topology;
using namespace sofa::component::collision;
using sofa::core::objectmodel::New;

/// create the chain as a child of root, and return root
Node::SPtr createChainHybrid(Node::SPtr root)
{
    const std::string visualModel="mesh/torus.obj";
    const std::string collisionModel="mesh/torus_for_collision.obj";

    std::vector<std::string> modelTypes;
    modelTypes.push_back("Triangle");
    modelTypes.push_back("Line");
    modelTypes.push_back("Point");

    //Elements of the scene
    //------------------------------------
    Node::SPtr  chain = root->createChild("Chain");


    //************************************
    //Torus Fixed

    Node::SPtr  torusFixed = sofa::ObjectCreator::CreateObstacle(chain,"mesh/torus_for_collision.obj", "mesh/torus.obj", "gray");

    //************************************
    //Torus FEM

    Node::SPtr  torusFEM = sofa::ObjectCreator::CreateEulerSolverNode(chain,"FEM");

    MeshGmshLoader::SPtr  loaderFEM = New<MeshGmshLoader>();
    loaderFEM->setFilename(sofa::helper::system::DataRepository.getFile("mesh/torus_low_res.msh"));
    loaderFEM->load();
    torusFEM->addObject(loaderFEM);

    MeshTopology::SPtr  meshTorusFEM = New<MeshTopology>();
    torusFEM->addObject(meshTorusFEM);

    const Deriv3 translationFEM(2.5,0,0);
    const Deriv3 rotationFEM(90,0,0);

    MechanicalObject3::SPtr  dofFEM = New<MechanicalObject3>(); dofFEM->setName("FEM Object");
    dofFEM->setTranslation(translationFEM[0],translationFEM[1],translationFEM[2]);
    dofFEM->setRotation(rotationFEM[0],rotationFEM[1],rotationFEM[2]);
    torusFEM->addObject(dofFEM);

    UniformMass3::SPtr  uniMassFEM = New <UniformMass3>();
    uniMassFEM->setTotalMass(5); //the whole object will have 5 as given mass
    torusFEM->addObject(uniMassFEM);

    TetrahedronFEMForceField3::SPtr  tetraFEMFF = New< TetrahedronFEMForceField3 >();
    tetraFEMFF->setName("FEM");
    tetraFEMFF->setComputeGlobalMatrix(false);
    tetraFEMFF->setMethod("large");
    tetraFEMFF->setPoissonRatio(0.3);
    tetraFEMFF->setYoungModulus(1000);
    torusFEM->addObject(tetraFEMFF);

    //Node VISUAL
    Node::SPtr  FEMVisualNode = sofa::ObjectCreator::CreateVisualNodeVec3(torusFEM,dofFEM,visualModel, "red", translationFEM, rotationFEM);

    //Node COLLISION
    Node::SPtr  FEMCollisionNode = sofa::ObjectCreator::CreateCollisionNodeVec3(torusFEM, dofFEM,collisionModel, modelTypes, translationFEM, rotationFEM );

    //************************************
    //Torus Spring

    Node::SPtr  torusSpring = sofa::ObjectCreator::CreateEulerSolverNode(chain,"Spring");

    MeshGmshLoader::SPtr  loaderSpring = New<MeshGmshLoader>();
    loaderSpring->setFilename(sofa::helper::system::DataRepository.getFile("mesh/torus_low_res.msh"));
    loaderSpring->load();

    torusSpring->addObject(loaderSpring);
    loaderSpring->init();

    MeshTopology::SPtr  meshTorusSpring = New< MeshTopology>();
    torusSpring->addObject(meshTorusSpring);

    const Deriv3 translationSpring(5,0,0);
    const Deriv3 rotationSpring(0,0,0);

    MechanicalObject3::SPtr  dofSpring = New<MechanicalObject3>(); dofSpring->setName("Spring Object");

    dofSpring->setTranslation(translationSpring[0],translationSpring[1],translationSpring[2]);
    dofSpring->setRotation(rotationSpring[0],rotationSpring[1],rotationSpring[2]);

    torusSpring->addObject(dofSpring);

    UniformMass3::SPtr  uniMassSpring = New <UniformMass3>();
    uniMassSpring->setTotalMass(5); //the whole object will have 5 as given mass
    torusSpring->addObject(uniMassSpring);

    MeshSpringForceField3::SPtr  springFF = New <MeshSpringForceField3>();
    springFF->setName("Springs");
    springFF->setStiffness(400);
    springFF->setDamping(0);
    torusSpring->addObject(springFF);


    //Node VISUAL
    Node::SPtr  SpringVisualNode = sofa::ObjectCreator::CreateVisualNodeVec3(torusSpring,dofSpring, visualModel,"green", translationSpring, rotationSpring);

    //Node COLLISION
    Node::SPtr  SpringCollisionNode = sofa::ObjectCreator::CreateCollisionNodeVec3(torusSpring ,dofSpring, collisionModel, modelTypes, translationSpring, rotationSpring);

    //************************************
    //Torus FFD

    Node::SPtr  torusFFD = sofa::ObjectCreator::CreateEulerSolverNode(chain,"FFD");

    const Deriv3 translationFFD(7.5,0,0);
    const Deriv3 rotationFFD(90,0,0);

    MechanicalObject3::SPtr   dofFFD = New<MechanicalObject3>(); dofFFD->setName("FFD Object");
    dofFFD->setTranslation(translationFFD[0],translationFFD[1],translationFFD[2]);
    dofFFD->setRotation(rotationFFD[0],rotationFFD[1],rotationFFD[2]);
    torusFFD->addObject(dofFFD);

    UniformMass3::SPtr   uniMassFFD = New<UniformMass3>();
    uniMassFFD->setTotalMass(5); //the whole object will have 5 as given mass
    torusFFD->addObject(uniMassFFD);

    RegularGridTopology::SPtr   gridTopo = New< RegularGridTopology>(6,2,5); //dimension of the grid
    gridTopo->setPos(
        -2.5,2.5,  //Xmin, Xmax
        -0.5,0.5,  //Ymin, Ymax
        -2,2       //Zmin, Zmax
    );
    torusFFD->addObject(gridTopo);

    RegularGridSpringForceField3::SPtr   FFDFF = New<RegularGridSpringForceField3>();
    FFDFF->setName("Springs FFD");
    FFDFF->setStiffness(200);
    FFDFF->setDamping(0);
    torusFFD->addObject(FFDFF);

    //Node VISUAL
    Node::SPtr  FFDVisualNode = sofa::ObjectCreator::CreateVisualNodeVec3(torusFFD, dofFFD, visualModel,"yellow", translationFFD);

    //Node COLLISION
    Node::SPtr  FFDCollisionNode = sofa::ObjectCreator::CreateCollisionNodeVec3(torusFFD,dofFFD,collisionModel,  modelTypes, translationFFD);



    //************************************
    //Torus Rigid

    Node::SPtr  torusRigid = sofa::ObjectCreator::CreateEulerSolverNode(chain,"Rigid");

    const Deriv3 translationRigid(10,0,0);
    const Deriv3 rotationRigid(0,0,0);

    MechanicalObjectRigid3::SPtr   dofRigid = New<MechanicalObjectRigid3>(); dofRigid->setName("Rigid Object");
    dofRigid->setTranslation(translationRigid[0],translationRigid[1],translationRigid[2]);
    dofRigid->setRotation(rotationRigid[0],rotationRigid[1],rotationRigid[2]);
    torusRigid->addObject(dofRigid);

    UniformMassRigid3::SPtr   uniMassRigid = New<UniformMassRigid3>();
    uniMassRigid->setTotalMass(1); //the whole object will have 5 as given mass
    torusRigid->addObject(uniMassRigid);

    //Node VISUAL
    Node::SPtr  RigidVisualNode = sofa::ObjectCreator::CreateVisualNodeRigid(torusRigid, dofRigid, visualModel,"gray");

    //Node COLLISION
    Node::SPtr  RigidCollisionNode = sofa::ObjectCreator::CreateCollisionNodeRigid(torusRigid,dofRigid,collisionModel, modelTypes);


    //************************************
    //Multi Mapping
    Node::SPtr  MultiParentsNode =torusFEM->createChild("MultiParentsNode");
    torusSpring->addChild(MultiParentsNode);
    torusFFD->addChild(MultiParentsNode);
    torusRigid->addChild(MultiParentsNode);

    //MultiParentsNode->setShowCollisionModels(false);

    MechanicalObject3::SPtr   dofMultiMapping = New<MechanicalObject3>(); dofMultiMapping->setName("Center Of Mass");
    MultiParentsNode->addObject(dofMultiMapping);
    sofa::helper::vector<State3*> stateIn;
    sofa::helper::vector<State3*> stateOut;
    CenterOfMassMulti2Mapping3_Rigid3_to_3::SPtr multiMappingCOM = New<CenterOfMassMulti2Mapping3_Rigid3_to_3>();
    multiMappingCOM->addInputModel(dofFEM.get());
    multiMappingCOM->addInputModel(dofSpring.get());
    multiMappingCOM->addInputModel(dofFFD.get());
    multiMappingCOM->addInputModel(dofRigid.get());
    multiMappingCOM->addOutputModel(dofMultiMapping.get());

    MultiParentsNode->addObject(multiMappingCOM);

    SphereModel::SPtr  spheres=New<SphereModel>();
    spheres->defaultRadius.setValue(0.5);
    MultiParentsNode->addObject( spheres);

    //MultiParentsNode->setShowCollisionModels(true);
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
    Node::SPtr  root = sofa::ObjectCreator::CreateRootWithCollisionPipeline(simulationType);
    root->setGravity( Coord3(0,0,-10) );

    //Add the objects
    createChainHybrid(root);

    root->setAnimate(false);

    getSimulation()->init(root.get());


    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(root);

    return 0;
}
