/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/simulation/graph/DAGNode.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperMeshTopology.h>
#include <sofa/component/topology/container/dynamic/PointSetTopologyContainer.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/component/collision/testing/MeshPrimitiveCreator.h>

namespace sofa {

using type::Vec3;
using core::objectmodel::New;

typedef sofa::component::topology::container::constant::MeshTopology MeshTopology;
typedef sofa::simulation::Node::SPtr NodePtr;
typedef sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types> TriangleModel;
typedef sofa::defaulttype::Vec3Types DataTypes;
typedef DataTypes::VecCoord VecCoord;

struct BaryMapperTest  : public ::testing::Test{


    MeshTopology* initMesh(NodePtr & father);

    bool test_inside(SReal alpha,SReal beta);
    bool test_outside(int index);
    void initTriPts();

    VecCoord triPts;
    Vec3 norm;
};

void BaryMapperTest::initTriPts(){
    triPts.resize(3);
    triPts[0] = Vec3(-1_sreal,-1_sreal,0_sreal);
    triPts[1] = Vec3(1_sreal,-1_sreal,0_sreal);
    triPts[2] = Vec3(0_sreal,1_sreal,0_sreal);
}

SReal tol = 1e-6_sreal;
SReal tol2 = tol*tol;
//BaryMapperTest::initTriPts();

//Vec3 BaryMapperTest::triPts[] = {Vec3(-1,-1,0),Vec3(1,-1,0),Vec3(0,1,0)};

static bool equal(const Vec3 & v0,const Vec3 & v1){
    return (v0 - v1).norm2() <= tol2;
}

MeshTopology* BaryMapperTest::initMesh(NodePtr &father){

    sofa::collision_test::makeTri(triPts[0],triPts[1],triPts[2],Vec3(0,0,0),father);

    norm = cross(-triPts[0] + triPts[1],triPts[2] - triPts[0]);
    norm.normalize();

    return father->getTreeObject<MeshTopology>();
}

bool BaryMapperTest::test_inside(SReal alpha,SReal beta){
    initTriPts();
    sofa::simulation::Node::SPtr father = New<sofa::simulation::graph::DAGNode>();
    MeshTopology * topo = initMesh(father);
    //makeTri()
    const component::mapping::linear::BarycentricMapperMeshTopology<DataTypes, DataTypes>::SPtr mapper = sofa::core::objectmodel::New<component::mapping::linear::BarycentricMapperMeshTopology<DataTypes, DataTypes> >(topo, (component::topology::container::dynamic::PointSetTopologyContainer*)0x0);

    const Vec3 the_point = ((SReal)(1.0) - alpha - beta) * triPts[0] + alpha * triPts[1] + beta * triPts[2];
    const Vec3 the_point_trans = the_point + (SReal)(10.0) * norm;
    mapper->createPointInTriangle( the_point_trans, 0, &triPts );

    VecCoord res;

    mapper->apply ( res, triPts );

    return equal(the_point,res[0]);
}


bool BaryMapperTest::test_outside(int index){
    initTriPts();
    sofa::simulation::Node::SPtr father = New<sofa::simulation::graph::DAGNode>();
    MeshTopology * topo = initMesh(father);
    //makeTri()
    const component::mapping::linear::BarycentricMapperMeshTopology<DataTypes, DataTypes>::SPtr mapper = sofa::core::objectmodel::New<component::mapping::linear::BarycentricMapperMeshTopology<DataTypes, DataTypes> >(topo,(component::topology::container::dynamic::PointSetTopologyContainer*)0x0);

    const Vec3 the_point = 2.0_sreal * triPts[index] + 10.0_sreal * norm;

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


}
