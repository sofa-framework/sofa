/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
using sofa::defaulttype::Vec3Types;
using Coord3 = sofa::type::Vec3;
using Deriv3 = sofa::defaulttype::Vec3Types::Deriv;
using VecCoord3 = sofa::type::vector<Coord3>;
#include <sofa/defaulttype/RigidTypes.h>
using sofa::defaulttype::Rigid3Types;
using sofa::defaulttype::Rigid3Mass;
#include <sofa/gui/common/GUIManager.h>
#include <sofa/gui/init.h>
#include <sofa/gui/common/ArgumentParser.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <sofa/component/io/mesh/MeshGmshLoader.h>
using sofa::component::io::mesh::MeshGmshLoader;
#include <sofa/component/statecontainer/MechanicalObject.h>
using MechanicalObject3 = sofa::component::statecontainer::MechanicalObject<Vec3Types>;
using MechanicalObjectRigid3 = sofa::component::statecontainer::MechanicalObject<Rigid3Types>;
#include <sofa/component/mass/UniformMass.h>
using UniformMass3 = sofa::component::mass::UniformMass<Vec3Types>;
using UniformMassRigid3 = sofa::component::mass::UniformMass<Rigid3Types>;
#include <sofa/component/topology/container/constant/MeshTopology.h>
using sofa::component::topology::container::constant::MeshTopology;
#include <sofa/component/topology/container/grid/RegularGridTopology.h>
using sofa::component::topology::container::grid::RegularGridTopology;
#include <sofa/component/solidmechanics/spring/MeshSpringForceField.h>
using MeshSpringForceField3 = sofa::component::solidmechanics::spring::MeshSpringForceField<Vec3Types>;
#include <sofa/component/solidmechanics/spring/RegularGridSpringForceField.h>
using RegularGridSpringForceField3 = sofa::component::solidmechanics::spring::RegularGridSpringForceField<Vec3Types>;
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>
using TetrahedronFEMForceField3 = sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField<Vec3Types>;
#include <sofa/simulation/graph/SimpleApi.h>
#include <sofa/component/init.h>

using sofa::core::objectmodel::New;
#include <SceneCreator/SceneCreator.h>


Node *createChainHybrid(Node *root)
{
    const std::string visualModel="mesh/torus.obj";
    const std::string collisionModel="mesh/torus_for_collision.obj";
    std::vector<std::string> collisionModelTypes = { "TriangleCollisionModel", "LineCollisionModel", "PointCollisionModel" };

    // Elements of the scene
    //------------------------------------
    Node::SPtr  chain = root->createChild("Chain");

    //************************************
    // Torus Fixed
    {
        Node::SPtr  torusFixed = sofa::modeling::createObstacle(chain,"mesh/torus_for_collision.obj", "mesh/torus.obj", "gray");
    }
    //************************************
    // Torus FEM
    {
        Node::SPtr  torusFEM = sofa::modeling::createEulerSolverNode(chain,"FEM");

        MeshGmshLoader::SPtr  loaderFEM = New<MeshGmshLoader>();
        loaderFEM->setFilename(sofa::helper::system::DataRepository.getFile("mesh/torus_low_res.msh"));
        loaderFEM->load();
        torusFEM->addObject(loaderFEM);

        MeshTopology::SPtr meshTorusFEM = sofa::core::objectmodel::New<MeshTopology>();
        meshTorusFEM->setSrc("",loaderFEM.get());
        torusFEM->addObject(meshTorusFEM);

        const Deriv3 translation(2.5,0,0);
        const Deriv3 rotation(90,0,0);

        MechanicalObject3::SPtr dofFEM = sofa::core::objectmodel::New<MechanicalObject3>(); 
        dofFEM->setName("FEM Object");
        dofFEM->setTranslation(translation[0],translation[1],translation[2]);
        dofFEM->setRotation(rotation[0],rotation[1],rotation[2]);
        torusFEM->addObject(dofFEM);

        UniformMass3::SPtr uniMassFEM = sofa::core::objectmodel::New<UniformMass3>();
        uniMassFEM->setTotalMass(5); //the whole object will have 5 as given mass
        torusFEM->addObject(uniMassFEM);

        TetrahedronFEMForceField3::SPtr tetraFEMFF = sofa::core::objectmodel::New<TetrahedronFEMForceField3>();
        tetraFEMFF->setName("FEM");
        tetraFEMFF->setComputeGlobalMatrix(false);
        tetraFEMFF->setMethod("large");
        tetraFEMFF->setPoissonRatio(0.3);
        tetraFEMFF->setYoungModulus(1000);
        torusFEM->addObject(tetraFEMFF);

        // Visual node
        sofa::modeling::createVisualNodeVec3(torusFEM, dofFEM.get(), visualModel, "red", translation, rotation);

        // Collision node
        sofa::modeling::createCollisionNodeVec3(torusFEM, dofFEM.get(), collisionModel, collisionModelTypes, translation, rotation );
    }
    //************************************
    // Torus Spring
    {
        Node::SPtr  torusSpring = sofa::modeling::createEulerSolverNode(chain,"Spring");

        MeshGmshLoader::SPtr  loaderSpring = New<MeshGmshLoader>();
        loaderSpring->setFilename(sofa::helper::system::DataRepository.getFile("mesh/torus_low_res.msh"));
        loaderSpring->load();

        torusSpring->addObject(loaderSpring);
        loaderSpring->init();

        MeshTopology::SPtr meshTorusSpring = sofa::core::objectmodel::New<MeshTopology>();
        meshTorusSpring->setSrc("",loaderSpring.get());
        torusSpring->addObject(meshTorusSpring);

        const Deriv3 translation(5,0,0);
        const Deriv3 rotation(0,0,0);

        MechanicalObject3::SPtr dofSpring = sofa::core::objectmodel::New<MechanicalObject3>(); 
        dofSpring->setName("Spring Object");

        dofSpring->setTranslation(translation[0],translation[1],translation[2]);
        dofSpring->setRotation(rotation[0],rotation[1],rotation[2]);

        torusSpring->addObject(dofSpring);

        UniformMass3::SPtr uniMassSpring = sofa::core::objectmodel::New<UniformMass3>();
        uniMassSpring->setTotalMass(5); //the whole object will have 5 as given mass
        torusSpring->addObject(uniMassSpring);

        MeshSpringForceField3::SPtr springFF = sofa::core::objectmodel::New<MeshSpringForceField3>();
        springFF->setName("Springs");
        springFF->setStiffness(400);
        springFF->setDamping(0);
        torusSpring->addObject(springFF);


        // Visual node
        Node::SPtr  SpringVisualNode = sofa::modeling::createVisualNodeVec3(torusSpring, dofSpring.get(), visualModel, "green", translation, rotation);

        // Collision node
        Node::SPtr  SpringCollisionNode = sofa::modeling::createCollisionNodeVec3(torusSpring, dofSpring.get(), collisionModel, collisionModelTypes, translation, rotation);
    }
    //************************************
    // Torus FFD
    {
        Node::SPtr  torusFFD = sofa::modeling::createEulerSolverNode(chain,"FFD");

        const Deriv3 translation(7.5,0,0);
        const Deriv3 rotation(90,0,0);

        MechanicalObject3::SPtr dofFFD = sofa::core::objectmodel::New<MechanicalObject3>(); 
        dofFFD->setName("FFD Object");
        dofFFD->setTranslation(translation[0],translation[1],translation[2]);
        dofFFD->setRotation(rotation[0],rotation[1],rotation[2]);
        torusFFD->addObject(dofFFD);

        UniformMass3::SPtr uniMassFFD = sofa::core::objectmodel::New<UniformMass3>();
        uniMassFFD->setTotalMass(5); //the whole object will have 5 as given mass
        torusFFD->addObject(uniMassFFD);

        RegularGridTopology::SPtr gridTopo = sofa::core::objectmodel::New<RegularGridTopology>(6,2,5); //dimension of the grid
        gridTopo->setPos(
            -2.5,2.5,  //Xmin, Xmax
            -0.5,0.5,  //Ymin, Ymax
            -2,2       //Zmin, Zmax
        );
        torusFFD->addObject(gridTopo);

        RegularGridSpringForceField3::SPtr FFDFF = sofa::core::objectmodel::New<RegularGridSpringForceField3>();
        FFDFF->setName("Springs FFD");
        FFDFF->setStiffness(200);
        FFDFF->setDamping(0);
        torusFFD->addObject(FFDFF);

        // Visual node
        sofa::modeling::createVisualNodeVec3(torusFFD, dofFFD.get(), visualModel, "yellow");

        // Collision node
        sofa::modeling::createCollisionNodeVec3(torusFFD, dofFFD.get(), collisionModel, collisionModelTypes);
    }

    //************************************
    // Torus Rigid
    {
        Node::SPtr  torusRigid = sofa::modeling::createEulerSolverNode(chain,"Rigid");

        const Deriv3 translation(10,0,0);
        const Deriv3 rotation(0,0,0);

        MechanicalObjectRigid3::SPtr dofRigid = sofa::core::objectmodel::New<MechanicalObjectRigid3>(); 
        dofRigid->setName("Rigid_Object");
        dofRigid->setTranslation(translation[0],translation[1],translation[2]);
        dofRigid->setRotation(rotation[0],rotation[1],rotation[2]);
        torusRigid->addObject(dofRigid);

        UniformMassRigid3::SPtr uniMassRigid = sofa::core::objectmodel::New<UniformMassRigid3>();
        uniMassRigid->setTotalMass(1); //the whole object will have 5 as given mass
        torusRigid->addObject(uniMassRigid);

        // Visual node
        sofa::modeling::createVisualNodeRigid(torusRigid, dofRigid.get(), visualModel, "gray");

        // Collision node
        sofa::modeling::createCollisionNodeRigid(torusRigid, dofRigid.get(), collisionModel, collisionModelTypes);
    }
    return root;
}



int main(int argc, char** argv)
{
    sofa::gui::common::ArgumentParser argParser(argc, argv);
    sofa::gui::common::GUIManager::RegisterParameters(&argParser);
    argParser.parse();

    //force load SofaComponentAll
    sofa::component::init();
    //force load SofaGui (registering guis)
    sofa::gui::init();

    sofa::gui::common::GUIManager::Init(argv[0]);

    auto simulation = sofa::simpleapi::createSimulation();

    // The graph root node
    Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    root->setGravity( Coord3(0,0,-10) );

    // Add the objects
    createChainHybrid(root.get());

    root->setAnimate(false);

    sofa::simulation::node::initRoot(root.get());


    //=======================================
    // Run the main loop
    sofa::gui::common::GUIManager::MainLoop(root);
    return 0;
}
