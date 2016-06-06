/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

// #include "stdafx.h"

#include <SofaTest/MultiMapping_test.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <Compliant/mapping/WinchMultiMapping.h>



namespace sofa {

namespace {

    using std::cout;
    using std::cerr;
    using std::endl;
    using namespace core;
    using namespace component;
    using defaulttype::Vec;
    using defaulttype::Mat;
    using sofa::helper::vector;

/**  Test suite for WinchMultiMapping
  */
template <typename Mapping>
struct WinchMultiMappingTest : public MultiMapping_test<Mapping>
{

    typedef WinchMultiMappingTest<Mapping> self;
    typedef MultiMapping_test<Mapping> base;


    bool test()
    {
        this->setupScene(2);
        Mapping* mapping = static_cast<Mapping*>(this->base::mapping);

        //parent positions
        helper::vector< self::InVecCoord > incoords(2);
        for( int i=0; i<2; i++ )
        {
            incoords[i].resize(1);
            incoords[i][0] = self::OutCoord(i+1.0) ;
        }

        // error
        mapping->factor.setValue(0.25);
        typename self::OutVecCoord outcoords(1);  
        outcoords[0] =  self::OutCoord(-0.5);

        return this->runTest(incoords, outcoords);

    }


};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
    sofa::component::mapping::WinchMultiMapping<defaulttype::Vec1dTypes, defaulttype::Vec1dTypes>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(WinchMultiMappingTest, DataTypes);

TYPED_TEST( WinchMultiMappingTest, test )
{
    ASSERT_TRUE( this->test() );
}

} // namespace 
} // namespace sofa
