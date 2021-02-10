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

#include <SofaTest/MultiMapping_test.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaSimulationGraph/DAGSimulation.h>
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
    typedef typename self::OutVecCoord OutVecCoord;
    typedef typename self::InVecCoord InVecCoord;
    typedef typename helper::vector < InVecCoord> VecOfInVecCoord;

    bool test()
    {
        this->setupScene(2);
        Mapping* mapping = static_cast<Mapping*>(this->base::mapping);

        VecOfInVecCoord incoords(2);
        for( int i=0; i<2; i++ )
        {
            incoords[i].resize(1);
            incoords[i][0] = typename self::OutCoord(i + 1.0);
        }            
        
        // error
        mapping->factor.setValue(0.25);
        typename self::OutVecCoord outcoords(1);
        outcoords[0] = typename self::OutCoord(-0.5);

        return this->runTest(incoords, outcoords);

    }
};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
    sofa::component::mapping::WinchMultiMapping<defaulttype::Vec1dTypes, defaulttype::Vec1dTypes>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_SUITE(WinchMultiMappingTest, DataTypes);

TYPED_TEST( WinchMultiMappingTest, test )
{
    ASSERT_TRUE( this->test() );
}

} // namespace 
} // namespace sofa