/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

// #include "stdafx.h"

#include <SofaTest/Mapping_test.h>
#include <sofa/defaulttype/VecTypes.h>
#include <Compliant/mapping/AssembledRigidRigidMapping.h>

#include <Compliant/utils/se3.h>
#include <Compliant/utils/edit.h>


namespace sofa {

/**  Test suite for AssembledRigidRigidMapping
  */
template <typename Mapping>
struct AssembledRigidRigidMappingTest : public Mapping_test<Mapping>
{

    typedef AssembledRigidRigidMappingTest self;
    typedef Mapping_test<Mapping> base;

    typedef SE3< typename self::Real > se3;
    
    Mapping* mapping;

    AssembledRigidRigidMappingTest() {
        mapping = static_cast<Mapping*>(this->base::mapping);
        this->errorMax *= 1.6;
    }

    bool test()
    {

        

        // parent
        typename self::InVecCoord xin(1);

        typename se3::vec3 v;

        v << M_PI / 3, 0, 0;
        
        xin[0].getOrientation() = se3::coord( se3::exp(v) );
        se3::map(xin[0].getCenter()) << 1, -5, 20;
        
        // child
        typename self::OutVecCoord xout(1);

        // offset
        typename self::OutCoord offset;

        v << 0, M_PI / 4, 0;
        offset.getOrientation() = se3::coord( se3::exp(v) );
        
        se3::map(offset.getCenter()) << 0, 1, 0;

        typename self::OutVecCoord expected(1);        

        expected[0].getOrientation() = xin[0].getOrientation() * offset.getOrientation();
        expected[0].getCenter() =
            xin[0].getOrientation().rotate( offset.getCenter() )
            + xin[0].getCenter();
        
        typename Mapping::source_type src;

        src.first = 0;
        src.second = offset;

        // mapping parameters
        edit(this->mapping->source)->push_back(src);
        this->mapping->geometricStiffness.setValue( 1 ); // non-symmetric geometric stiffness
        
        return this->runTest(xin, xout, xin, expected);
    }


};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
    component::mapping::AssembledRigidRigidMapping<defaulttype::Rigid3Types,
                                                   defaulttype::Rigid3Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(AssembledRigidRigidMappingTest, DataTypes);

TYPED_TEST( AssembledRigidRigidMappingTest, test )
{
    ASSERT_TRUE( this->test() );
}

} // namespace sofa
