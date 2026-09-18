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
#include <sofa/component/solidmechanics/fem/elastic/LinearSmallStrainFEMForceField.inl>
#include <sofa/component/solidmechanics/testing/ForceFieldTestCreation.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>

namespace sofa
{
template<class DataTypes>
using TetrahedronLinearSmallStrainFEMForceField =
    sofa::component::solidmechanics::fem::elastic::LinearSmallStrainFEMForceField<DataTypes, sofa::geometry::Tetrahedron>;

/**
 * This test is based on the generic test valid on every force field.
 *
 * It checks the consistency of the derivative functions (addDForce, addKToMatrix,
 * buildStiffnessMatrix) compared to the force function (addForce) using finite differences.
 */
template <class DataTypes>
struct TET4LinearSmallStrainFEMForceField_stepTest :
    public sofa::ForceField_test<TetrahedronLinearSmallStrainFEMForceField<DataTypes>>
{
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;

    TET4LinearSmallStrainFEMForceField_stepTest()
    {
        auto topology = sofa::core::objectmodel::New<sofa::component::topology::container::constant::MeshTopology>();
        this->node->addObject(topology);

        topology->addTetra(0,1,2,3);

        auto x = this->dof->writeOnlyRestPositions();
        x.resize(4);
        DataTypes::set(x[0], 0, 0, 0);
        DataTypes::set(x[1], 1, 0, 0);
        DataTypes::set(x[2], 0, 1, 0);
        DataTypes::set(x[3], 0, 0, 1);
    }

    void runTest()
    {
        VecCoord x = this->dof->readRestPositions().ref();
        VecDeriv v,f;

        //Position: extension of 10% in each dimension
        x.resize(4);
        DataTypes::set(x[0], 0, 0, 0);
        DataTypes::set(x[1], 1.1, 0., 0.);
        DataTypes::set(x[2], 0., 1.1, 0.);
        DataTypes::set(x[3], 0., 0., 1.1);

        //Velocity
        v.resize(4);
        for (auto& vel : v)
        {
            DataTypes::set(vel, 0, 0, 0);
        }

        //mechanical parameters
        this->force->d_poissonRatio.setValue({0});
        this->force->d_youngModulus.setValue({1});

        //Force e*E*S*1/3  = 1*40*sqrt(3)/4*1/3
        f.resize(4);
        constexpr auto k = 1./ 60.;
        DataTypes::set( f[0], k, k, k);
        DataTypes::set( f[1], -k, 0., 0.);
        DataTypes::set( f[2], 0., -k, 0.);
        DataTypes::set( f[3], 0., 0., -k);

        sofa::simulation::node::initRoot(this->node.get());

        this->run_test( x, v, f );
    }
};

typedef ::testing::Types<
    sofa::defaulttype::Vec3Types
> TestTypes;

TYPED_TEST_SUITE(TET4LinearSmallStrainFEMForceField_stepTest, TestTypes);

TYPED_TEST(TET4LinearSmallStrainFEMForceField_stepTest, extension )
{
    this->errorMax *= 1e2;
    this->deltaRange = std::make_pair( 1, this->errorMax * 10 );
    this->debug = true;

    this->runTest();
}

}
