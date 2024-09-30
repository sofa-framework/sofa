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
#include <sofa/component/solidmechanics/testing/ForceFieldTestCreation.h>
#include <sofa/component/solidmechanics/fem/elastic/TriangleFEMForceField.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>


namespace sofa
{

template <typename _TriangleFEMForceField>
struct TriangleFEMForceField_stepTest : public ForceField_test<_TriangleFEMForceField>
{
    typedef typename _TriangleFEMForceField::DataTypes DataTypes;

    typedef typename _TriangleFEMForceField::VecCoord VecCoord;
    typedef typename _TriangleFEMForceField::VecDeriv VecDeriv;
    typedef typename _TriangleFEMForceField::Coord Coord;
    typedef typename _TriangleFEMForceField::Deriv Deriv;
    typedef typename Coord::value_type Real;

    VecCoord x;
    VecDeriv v,f;

    TriangleFEMForceField_stepTest()
    {
        auto topology = modeling::addNew<component::topology::container::constant::MeshTopology>(this->node);
        topology->setName("topology");
        topology->d_seqTriangles.setValue({{0,1,2}});
        topology->d_seqPoints.setParent(&this->dof->x);

        //Position
        x.resize(3);
        DataTypes::set( x[0], 0,0,0);
        DataTypes::set( x[1], 1,0,0);
        DataTypes::set( x[2], 0,1,0);

        this->dof->x.setValue(x);

        //Velocity
        v.resize(3);
        DataTypes::set( v[0], 0,0,0);
        DataTypes::set( v[1], 0,0,0);
        DataTypes::set( v[2], 0,0,0);

        //force
        f.resize(3);

        // Set force parameters
        this->force->setPoissonRatio(0);
        this->force->setYoungModulus(40);
        this->force->d_method.setValue("large");

        sofa::simulation::node::initRoot(this->node.get());
    }

    void test()
    {
        this->run_test( x, v, f );
    }
};

typedef ::testing::Types<
    sofa::component::solidmechanics::fem::elastic::TriangleFEMForceField<defaulttype::Vec3Types>
> TestTypes;



// ========= Tests to run for each instanciated type
TYPED_TEST_SUITE(TriangleFEMForceField_stepTest, TestTypes);

TYPED_TEST(TriangleFEMForceField_stepTest, test )
{
    this->errorMax *= 10;
    this->deltaRange = std::make_pair( 1, this->errorMax * 10 );
    this->debug = false;

    // potential energy is not implemented and won't be tested
    this->flags &= ~TriangleFEMForceField_stepTest<TypeParam>::TEST_POTENTIAL_ENERGY;

    this->test();
}

}
