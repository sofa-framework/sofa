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
#include <sofa/component/solidmechanics/fem/elastic/HexahedralFEMForceField.h>

#include <sofa/component/solidmechanics/testing/ForceFieldTestCreation.h>
#include "HexahedronFEMForceField_test.h"

namespace sofa
{

template <typename _HexahedralFEMForceField>
struct HexahedralFEMForceField_test : public HexahedronFEMForceField_test<_HexahedralFEMForceField>
{};

// ========= Define the list of types to instanciate.
//using ::testing::Types;
typedef ::testing::Types<
::sofa::component::solidmechanics::fem::elastic::HexahedralFEMForceField<defaulttype::Vec3Types>
> TestTypes; // the types to instanciate.

// ========= Tests to run for each instanciated type
TYPED_TEST_SUITE(HexahedralFEMForceField_test, TestTypes);

// test case
TYPED_TEST( HexahedralFEMForceField_test , extension )
{
    this->errorMax *= 1000000;
    this->deltaRange = std::make_pair( 1, this->errorMax * 10 );
    this->debug = false;
    this->flags &= ~TEST_POTENTIAL_ENERGY;

    // run test
    this->test_valueForce();
}

TYPED_TEST( HexahedralFEMForceField_test, testComputeBBox )
{
    ASSERT_NO_THROW(this->testComputeBBox()) ;
}
} // namespace sofa
