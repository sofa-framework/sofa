#include "stdafx.h"
#include "PrimitiveCreation.h"

#include "Sofa_test.h"
#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaBaseMechanics/BarycentricMapping.h>

using sofa::core::objectmodel::New;
using namespace sofa::defaulttype;

namespace sofa {
namespace {

typedef sofa::component::topology::MeshTopology MeshTopology;
typedef sofa::simulation::Node::SPtr NodePtr;
typedef sofa::component::collision::TriangleModel TriangleModel;
typedef sofa::defaulttype::Vec3Types DataTypes;
typedef DataTypes::VecCoord VecCoord;
struct BaryMapperTest  : public ::testing::Test{


    MeshTopology* initMesh(NodePtr & father);

    bool test_inside(SReal alpha,SReal beta);
    bool test_outside(int index);
    void initTriPts();

    VecCoord triPts;
    Vector3 norm;
};

void BaryMapperTest::initTriPts(){
    triPts.resize(3);
    triPts[0] = Vector3(-1,-1,0);
    triPts[1] = Vector3(1,-1,0);
    triPts[2] = Vector3(0,1,0);
}

SReal tol = 1e-6;
SReal tol2 = tol*tol;
//BaryMapperTest::initTriPts();

//Vector3 BaryMapperTest::triPts[] = {Vector3(-1,-1,0),Vector3(1,-1,0),Vector3(0,1,0)};

static bool equal(const Vector3 & v0,const Vector3 & v1){
    return (v0 - v1).norm2() <= tol2;
}

MeshTopology* BaryMapperTest::initMesh(NodePtr &father){
    PrimitiveCreationTest::makeTri(triPts[0],triPts[1],triPts[2],Vector3(0,0,0),father);
    norm = cross(-triPts[0] + triPts[1],triPts[2] - triPts[0]);
    norm.normalize();

    return father->getTreeObject<MeshTopology>();
}

//sofa::BaryMapperTest::triPts[0] = Vector3(-1,-1,0);
//triPts[0];// = Vector3(-1,-1,0);



//sofa::defaulttype::Rigid3Types zz;
//BarycentricMapperMeshTopology(core::topology::BaseMeshTopology* fromTopology,
//        component::topology::PointSetTopologyContainer* toTopology,
//        helper::ParticleMask *_maskFrom,
//        helper::ParticleMask *_maskTo)
//    : TopologyBarycentricMapper<In,Out>(fromTopology, toTopology),
//      maskFrom(_maskFrom), maskTo(_maskTo),
//      matrixJ(NULL), updateJ(true)
//{
//}

bool BaryMapperTest::test_inside(SReal alpha,SReal beta){
    initTriPts();
    sofa::simulation::Node::SPtr father = New<sofa::simulation::tree::GNode>();
    MeshTopology * topo = initMesh(father);
    //makeTri()
    component::mapping::BarycentricMapperMeshTopology<DataTypes, DataTypes>::SPtr mapper = sofa::core::objectmodel::New<component::mapping::BarycentricMapperMeshTopology<DataTypes, DataTypes> >(topo,(component::topology::PointSetTopologyContainer*)0x0,(helper::ParticleMask *)0x0,(helper::ParticleMask *)0x0/*model->getMeshTopology(), (topology::PointSetTopologyContainer*)NULL, &model->getMechanicalState()->forceMask, &mstate->forceMask*/);

    Vector3 the_point = ((SReal)(1.0) - alpha - beta) * triPts[0] + alpha * triPts[1] + beta * triPts[2];
    Vector3 the_point_trans = the_point + (SReal)(10.0) * norm;
    mapper->createPointInTriangle( the_point_trans, 0, &triPts );

    VecCoord res;

    mapper->apply ( res, triPts );

    return equal(the_point,res[0]);
}


bool BaryMapperTest::test_outside(int index){
    initTriPts();
    sofa::simulation::Node::SPtr father = New<sofa::simulation::tree::GNode>();
    MeshTopology * topo = initMesh(father);
    //makeTri()
    component::mapping::BarycentricMapperMeshTopology<DataTypes, DataTypes>::SPtr mapper = sofa::core::objectmodel::New<component::mapping::BarycentricMapperMeshTopology<DataTypes, DataTypes> >(topo,(component::topology::PointSetTopologyContainer*)0x0,(helper::ParticleMask *)0x0,(helper::ParticleMask *)0x0/*model->getMeshTopology(), (topology::PointSetTopologyContainer*)NULL, &model->getMechanicalState()->forceMask, &mstate->forceMask*/);

    Vector3 the_point = (SReal)(2.0) * triPts[index] + (SReal)(10.0) * norm;

    mapper->createPointInTriangle( the_point, 0, &triPts );

    VecCoord res;

    mapper->apply ( res, triPts );

    return equal(triPts[index],res[0]);
}

TEST_F(BaryMapperTest, alpha0dot3_beta0dot2 ) { ASSERT_TRUE( test_inside((SReal)(0.3),(SReal)(0.2))); }
TEST_F(BaryMapperTest, alpha0dot4_beta0dot6 ) { ASSERT_TRUE( test_inside((SReal)(0.4),(SReal)(0.6))); }
TEST_F(BaryMapperTest, alpha0_beta0 ) { ASSERT_TRUE( test_inside((SReal)(0),(SReal)(0))); }
TEST_F(BaryMapperTest, out_0 ) { ASSERT_TRUE( test_outside(0)); }
TEST_F(BaryMapperTest, out_1 ) { ASSERT_TRUE( test_outside(1)); }
TEST_F(BaryMapperTest, out_2 ) { ASSERT_TRUE( test_outside(2)); }


} // namespace
} // namespace sofa

