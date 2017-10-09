#include <SofaTest/Sofa_test.h>
#include <SceneCreator/SceneCreator.h>

#include <SofaSimpleFem/TetrahedronFEMForceField.h>
#include <SofaBaseMechanics/MechanicalObject.h>
using sofa::component::container::MechanicalObject ;
typedef sofa::component::container::MechanicalObject<Vec3Types>                   MechanicalObject3;

#include <SofaMiscFem/TriangularFEMForceField.h>
using sofa::component::forcefield::TetrahedronFEMForceField;
using sofa::component::forcefield::TriangularFEMForceField;
typedef sofa::component::forcefield::TetrahedronFEMForceField<Vec3Types>          TetrahedronFEMForceField3;
typedef sofa::component::forcefield::TriangularFEMForceField<Vec3Types>           TriangularFEMForceField3;

#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaGeneralTopology/CylinderGridTopology.h>
#include <SofaGeneralTopology/SphereGridTopology.h>

using sofa::component::topology::RegularGridTopology;
using sofa::component::topology::CylinderGridTopology;
using sofa::component::topology::SphereGridTopology;

using sofa::core::objectmodel::BaseContext;
using sofa::defaulttype::Vec3Types;


#include <sofa/simulation/Simulation.h>
using sofa::simulation::Node;

typedef MechanicalObject<sofa::defaulttype::Vec3Types>      MechanicalObject3;
typedef TetrahedronFEMForceField<Vec3Types>                 TetrahedronFEMForceField3;
typedef TriangularFEMForceField<Vec3Types>                  TriangularFEMForceField3;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

class SceneCreator_test : public sofa::Sofa_test<>
{
public:
    bool createCubeFailed();
    bool createCubeSuccess();
    bool createRigidCubeSuccess();

    bool createCylinderFailed();
    bool createCylinderSuccess();
    bool createRigidCylinderSuccess();

    bool createSphereFailed();
    bool createSphereSuccess();
    bool createRigidSphereSuccess();

    bool createPlaneFailed();
    bool createPlaneSuccess();
    bool createRigidPlaneSuccess();

};

///////////////////////////////////////////////////////////////////
/////////////////////////// Cube Methods //////////////////////////
///////////////////////////////////////////////////////////////////

bool SceneCreator_test::createCubeFailed()
{
    // Null parent for Cube case
    Node::SPtr cube = sofa::modeling::addCube(nullptr, "cubeFEM_1", Vec3Types::Deriv(5, 5, 5),
                                              10, 1000, 0.45,
                                              Vec3Types::Deriv(0, 5, 0));
    EXPECT_EQ(cube, nullptr);

    // Null parent for rigid Cube case
    cube = sofa::modeling::addRigidCube(nullptr, "cubeFIX_2",
                                        Vec3Types::Deriv(5, 5, 5),
                                        Vec3Types::Deriv(0, 5, 0));

    EXPECT_EQ(cube, nullptr);

    // Cube with bad grid size
    Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    cube = sofa::modeling::addRigidCube(root, "cubeFIX_3", Vec3Types::Deriv(0, 5, 5),
                                        Vec3Types::Deriv(0, 5, 0));
    sofa::simulation::getSimulation()->init(root.get());

    EXPECT_EQ(cube, nullptr);

    return true;
}

bool SceneCreator_test::createCubeSuccess()
{
    // Create cube
    SReal poissonRatio = 0.45;
    Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    Node::SPtr node = sofa::modeling::addCube(root, "cubeFEM",
                                              Vec3Types::Deriv(5, 5, 5),
                                              10, 1000, poissonRatio,
                                              Vec3Types::Deriv(0, 5, 0));

    EXPECT_NE(node, nullptr);

    // Check MecaObj
    std::vector<MechanicalObject3*> mecaObjs;
    node->get<MechanicalObject3>(&mecaObjs, BaseContext::SearchDown);
    EXPECT_EQ(mecaObjs.size(), 1u);


    // check Grid
    std::vector<RegularGridTopology*> grids;
    node->get<RegularGridTopology>(&grids, BaseContext::SearchDown);
    EXPECT_EQ(grids.size(), 1u);

    RegularGridTopology* grid = grids[0];
    EXPECT_NE(grid->getNbPoints(), 0);
    EXPECT_NE(grid->getNbEdges(), 0);


    // check FEM
    std::vector<TetrahedronFEMForceField3*> FEMs;
    node->get<TetrahedronFEMForceField3>(&FEMs, BaseContext::SearchDown);
    EXPECT_EQ(grids.size(), 1u);

    TetrahedronFEMForceField3* fem = FEMs[0];
    EXPECT_EQ(fem->_poissonRatio.getValue(), poissonRatio);

    return true;
}

bool SceneCreator_test::createRigidCubeSuccess()
{
    Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    Node::SPtr node = sofa::modeling::addRigidCube(root, "cubeFIX",
                                                   Vec3Types::Deriv(5, 5, 5),
                                                   Vec3Types::Deriv(0, 5, 0));

    EXPECT_NE(node, nullptr);

    // complementary with test createCubeSuccess

    // check Grid
    std::vector<RegularGridTopology*> grids;
    node->get<RegularGridTopology>(&grids, BaseContext::SearchDown);
    EXPECT_EQ(grids.size(), 1u);

    // check No FEM
    std::vector<TetrahedronFEMForceField3*> FEMs;
    node->get<TetrahedronFEMForceField3>(&FEMs, BaseContext::SearchDown);
    EXPECT_EQ(FEMs.size(), 0u);

    return true;
}


///////////////////////////////////////////////////////////////////
//////////////////////// Cylinder Methods /////////////////////////
///////////////////////////////////////////////////////////////////

bool SceneCreator_test::createCylinderFailed()
{
    // Null parent for Cylinder case
    Node::SPtr cyl = sofa::modeling::addCylinder(nullptr, "cylinderFEM_1",
                                                 Vec3Types::Deriv(5, 5, 5),
                                                 Vec3Types::Deriv(0, 1, 0), 1.0, 3.0,
                                                 10, 1000, 0.45,
                                                 Vec3Types::Deriv(0, 5, 0));
    EXPECT_EQ(cyl, nullptr);

    // Null parent for rigid Cylinder case
    cyl = sofa::modeling::addRigidCylinder(nullptr, "cylinderFIX_2",
                                           Vec3Types::Deriv(5, 5, 5),
                                           Vec3Types::Deriv(1, 0, 0), 0.5, 3.0,
                                           Vec3Types::Deriv(0, 5, 0));

    EXPECT_EQ(cyl, nullptr);

    // Cylinder with bad grid size
    Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    cyl = sofa::modeling::addRigidCylinder(root, "cylinderFIX_3",
                                           Vec3Types::Deriv(0, 5, 5),
                                           Vec3Types::Deriv(1, 0, 0), 0.5, 3.0,
                                           Vec3Types::Deriv(0, 5, 0));

    sofa::simulation::getSimulation()->init(root.get());

    EXPECT_EQ(cyl, nullptr);
    return true;
}

bool SceneCreator_test::createCylinderSuccess()
{
    // Create cylinder
    SReal poissonRatio = 0.45;
    Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    Node::SPtr node = sofa::modeling::addCylinder(root, "cylinderFEM_1",
                                                  Vec3Types::Deriv(5, 5, 5),
                                                  Vec3Types::Deriv(0, 1, 0), 1.0, 3.0,
                                                  10, 1000, 0.45,
                                                  Vec3Types::Deriv(0, 5, 0));

    EXPECT_NE(node, nullptr);

    // Check MecaObj
    std::vector<MechanicalObject3*> mecaObjs;
    node->get<MechanicalObject3>(&mecaObjs, BaseContext::SearchDown);
    EXPECT_EQ(mecaObjs.size(), 1u);


    // check Grid
    std::vector<CylinderGridTopology*> grids;
    node->get<CylinderGridTopology>(&grids, BaseContext::SearchDown);
    EXPECT_EQ(grids.size(), 1u);

    CylinderGridTopology* grid = grids[0];
    EXPECT_NE(grid->getNbPoints(), 0);
    EXPECT_NE(grid->getNbEdges(), 0);


    // check FEM
    std::vector<TetrahedronFEMForceField3*> FEMs;
    node->get<TetrahedronFEMForceField3>(&FEMs, BaseContext::SearchDown);
    EXPECT_EQ(grids.size(), 1u);

    TetrahedronFEMForceField3* fem = FEMs[0];
    EXPECT_EQ(fem->_poissonRatio.getValue(), poissonRatio);

    return true;
}

bool SceneCreator_test::createRigidCylinderSuccess()
{
    Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    Node::SPtr node = sofa::modeling::addRigidCylinder(root, "cylinderFIX_3",
                                                       Vec3Types::Deriv(5, 5, 5),
                                                       Vec3Types::Deriv(1, 0, 0),
                                                       0.5, 3.0,
                                                       Vec3Types::Deriv(0, 5, 0));

    EXPECT_NE(node, nullptr);

    // complementary with test createCylinderSuccess

    // check Grid
    std::vector<CylinderGridTopology*> grids;
    node->get<CylinderGridTopology>(&grids, BaseContext::SearchDown);
    EXPECT_EQ(grids.size(), 1u);

    // check No FEM
    std::vector<TetrahedronFEMForceField3*> FEMs;
    node->get<TetrahedronFEMForceField3>(&FEMs, BaseContext::SearchDown);
    EXPECT_EQ(FEMs.size(), 0u);

    return true;
}


///////////////////////////////////////////////////////////////////
//////////////////////// Sphere Methods /////////////////////////
///////////////////////////////////////////////////////////////////

bool SceneCreator_test::createSphereFailed()
{
    // Null parent for Sphere case
    sofa::simulation::Node::SPtr cyl = sofa::modeling::addSphere(nullptr, "SphereFEM_1", sofa::defaulttype::Vec3Types::Deriv(5, 5, 5),
                                                                   sofa::defaulttype::Vec3Types::Deriv(0, 1, 0), 1.0,
                                                                   10, 1000, 0.45,
                                                                   sofa::defaulttype::Vec3Types::Deriv(0, 5, 0));
    EXPECT_EQ(cyl, nullptr);

    // Null parent for rigid Sphere case
    cyl = sofa::modeling::addRigidSphere(nullptr, "SphereFIX_2", sofa::defaulttype::Vec3Types::Deriv(5, 5, 5),
                                           sofa::defaulttype::Vec3Types::Deriv(1, 0, 0), 0.5,
                                           sofa::defaulttype::Vec3Types::Deriv(0, 5, 0));

    EXPECT_EQ(cyl, nullptr);

    // Sphere with bad grid size
    sofa::simulation::Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    cyl = sofa::modeling::addRigidSphere(root, "SphereFIX_3", sofa::defaulttype::Vec3Types::Deriv(0, 5, 5),
                                            sofa::defaulttype::Vec3Types::Deriv(1, 0, 0), 0.5,
                                            sofa::defaulttype::Vec3Types::Deriv(0, 5, 0));
    sofa::simulation::getSimulation()->init(root.get());

    EXPECT_EQ(cyl, nullptr);

    return true;
}

bool SceneCreator_test::createSphereSuccess()
{
    // Create Sphere
    SReal poissonRatio = 0.45;
    sofa::simulation::Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    sofa::simulation::Node::SPtr node = sofa::modeling::addSphere(root, "SphereFEM_1", sofa::defaulttype::Vec3Types::Deriv(5, 5, 5),
                                                                    sofa::defaulttype::Vec3Types::Deriv(0, 1, 0), 1.0,
                                                                    10, 1000, 0.45,
                                                                    sofa::defaulttype::Vec3Types::Deriv(0, 5, 0));

    EXPECT_NE(node, nullptr);

    // Check MecaObj
    std::vector<MechanicalObject3*> mecaObjs;
    node->get<MechanicalObject3>(&mecaObjs, sofa::core::objectmodel::BaseContext::SearchDown);
    EXPECT_EQ(mecaObjs.size(), (size_t)1);


    // check Grid
    std::vector<SphereGridTopology*> grids;
    node->get<SphereGridTopology>(&grids, sofa::core::objectmodel::BaseContext::SearchDown);
    EXPECT_EQ(grids.size(), (size_t)1);

    SphereGridTopology* grid = grids[0];
    EXPECT_NE(grid->getNbPoints(), 0);
    EXPECT_NE(grid->getNbEdges(), 0);


    // check FEM
    std::vector<TetrahedronFEMForceField3*> FEMs;
    node->get<TetrahedronFEMForceField3>(&FEMs, sofa::core::objectmodel::BaseContext::SearchDown);
    EXPECT_EQ(grids.size(), (size_t)1);

    TetrahedronFEMForceField3* fem = FEMs[0];
    EXPECT_EQ(fem->_poissonRatio.getValue(), poissonRatio);

    return true;
}

bool SceneCreator_test::createRigidSphereSuccess()
{
    sofa::simulation::Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    sofa::simulation::Node::SPtr node = sofa::modeling::addRigidSphere(root, "SphereFIX_3", sofa::defaulttype::Vec3Types::Deriv(5, 5, 5),
                                                                         sofa::defaulttype::Vec3Types::Deriv(1, 0, 0), 0.5,
                                                                         sofa::defaulttype::Vec3Types::Deriv(0, 5, 0));

    EXPECT_NE(node, nullptr);

    // complementary with test createSphereSuccess

    // check Grid
    std::vector<SphereGridTopology*> grids;
    node->get<SphereGridTopology>(&grids, sofa::core::objectmodel::BaseContext::SearchDown);
    EXPECT_EQ(grids.size(), (size_t)1);

    // check No FEM
    std::vector<TetrahedronFEMForceField3*> FEMs;
    node->get<TetrahedronFEMForceField3>(&FEMs, sofa::core::objectmodel::BaseContext::SearchDown);
    EXPECT_EQ(FEMs.size(), (size_t)0);

    return true;
}

///////////////////////////////////////////////////////////////////
////////////////////////// Plane Methods //////////////////////////
///////////////////////////////////////////////////////////////////

bool SceneCreator_test::createPlaneFailed()
{
    // Null parent for Plane case
    Node::SPtr plane = sofa::modeling::addPlane(nullptr, "Drap",
                                                Vec3Types::Deriv(50, 1, 50), 30, 600, 0.3,
                                                Vec3Types::Deriv(0, 30, 0),
                                                Vec3Types::Deriv(0, 0, 0),
                                                Vec3Types::Deriv(20, 0, 20));
    EXPECT_EQ(plane, nullptr);

    // Null parent for rigid Plane case
    plane = sofa::modeling::addRigidPlane(nullptr, "Floor",
                                          Vec3Types::Deriv(50, 1, 50),
                                          Vec3Types::Deriv(0, 0, 0),
                                          Vec3Types::Deriv(0, 0, 0),
                                          Vec3Types::Deriv(40, 0, 40));

    EXPECT_EQ(plane, nullptr);

    // Plane with bad grid size
    Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    plane = sofa::modeling::addRigidPlane(root, "Floor",
                                          Vec3Types::Deriv(50, 0, 50),
                                          Vec3Types::Deriv(0, 0, 0),
                                          Vec3Types::Deriv(0, 0, 0),
                                          Vec3Types::Deriv(40, 0, 40));

    sofa::simulation::getSimulation()->init(root.get());

    EXPECT_EQ(plane, nullptr);

    return true;
}

bool SceneCreator_test::createPlaneSuccess()
{
    // TODO: epernod TriangularFEM creation is not working yet on regularGrid
    return true;
}

bool SceneCreator_test::createRigidPlaneSuccess()
{
    Node::SPtr root = sofa::modeling::createRootWithCollisionPipeline();
    Node::SPtr node = sofa::modeling::addRigidPlane(root, "Floor",
                                                    Vec3Types::Deriv(50, 1, 50),
                                                    Vec3Types::Deriv(0, 0, 0),
                                                    Vec3Types::Deriv(0, 0, 0),
                                                    Vec3Types::Deriv(40, 0, 40));

    EXPECT_NE(node, nullptr);

    // complementary with test createPlaneSuccess

    // check Grid
    std::vector<RegularGridTopology*> grids;
    node->get<RegularGridTopology>(&grids, BaseContext::SearchDown);
    EXPECT_EQ(grids.size(), 1u);

    // check No FEM
    std::vector<TriangularFEMForceField3*> FEMs;
    node->get<TriangularFEMForceField3>(&FEMs, BaseContext::SearchDown);
    EXPECT_EQ(FEMs.size(), 0u);

    return true;
}


TEST_F(SceneCreator_test, createCubeFailed ) { ASSERT_TRUE( createCubeFailed()); }
TEST_F(SceneCreator_test, createCubeSuccess ) { ASSERT_TRUE( createCubeSuccess()); }
TEST_F(SceneCreator_test, createRigidCubeSuccess ) { ASSERT_TRUE( createRigidCubeSuccess()); }

TEST_F(SceneCreator_test, createCylinderFailed ) { ASSERT_TRUE( createCylinderFailed()); }
TEST_F(SceneCreator_test, createCylinderSuccess ) { ASSERT_TRUE( createCylinderSuccess()); }
TEST_F(SceneCreator_test, createRigidCylinderSuccess ) { ASSERT_TRUE( createRigidCylinderSuccess()); }

TEST_F(SceneCreator_test, createSphereFailed ) { ASSERT_TRUE( createSphereFailed()); }
TEST_F(SceneCreator_test, createSphereSuccess ) { ASSERT_TRUE( createSphereSuccess()); }
TEST_F(SceneCreator_test, createRigidSphereSuccess ) { ASSERT_TRUE( createRigidSphereSuccess()); }

TEST_F(SceneCreator_test, createPlaneFailed ) { ASSERT_TRUE( createPlaneFailed()); }
TEST_F(SceneCreator_test, createPlaneSuccess ) { ASSERT_TRUE( createPlaneSuccess()); }
TEST_F(SceneCreator_test, createRigidPlaneSuccess ) { ASSERT_TRUE( createRigidPlaneSuccess()); }

