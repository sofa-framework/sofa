/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaMiscMapping/ProjectionToPlaneMapping.h>


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


/**  Test suite for ProjectToTargetLineMapping.
  */
template <typename ProjectionToPlaneMultiMapping>
struct ProjectionToPlaneMultiMappingTest : public MultiMapping_test<ProjectionToPlaneMultiMapping>
{
    typedef typename ProjectionToPlaneMultiMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::Coord InCoord;

    typedef typename ProjectionToPlaneMultiMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::Coord OutCoord;


    bool test()
    {
        this->setupScene(2); // 2 parents, 1 child

//        ProjectionToPlaneMultiMapping* pttlm = static_cast<ProjectionToPlaneMultiMapping*>( this->mapping );


        // parent positions
        vector< InVecCoord > incoords(2);

        incoords[0].resize(1);
        InDataTypes::set( incoords[0][0], 0,0,10 );

        incoords[1].resize(2); // center / normal
        InDataTypes::set( incoords[1][0], 0,0,0 );
        InDataTypes::set( incoords[1][1], 1,1,1 );


        // expected child positions
        OutVecCoord expectedoutcoord(1);
        OutDataTypes::set( expectedoutcoord[0], -10,-10,0 );

        return this->runTest( incoords, expectedoutcoord );
    }

};


// Define the list of types to instanciate.
using testing::Types;
typedef Types<
mapping::ProjectionToPlaneMultiMapping<defaulttype::Vec3Types,defaulttype::Vec3Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE( ProjectionToPlaneMultiMappingTest, DataTypes );

// test case
TYPED_TEST( ProjectionToPlaneMultiMappingTest , test )
{
    ASSERT_TRUE(this->test());
}

} // namespace
} // namespace sofa
