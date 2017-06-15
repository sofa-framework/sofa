/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include <SofaTest/Mapping_test.h>
#include <SofaMiscMapping/SquareMapping.h>


namespace sofa {
namespace {


/**  Test suite for SquareMapping.
 *
 * @author Matthieu Nesme
  */
template <typename SquareMapping>
struct SquareMappingTest : public Mapping_test<SquareMapping>
{
    typedef typename SquareMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::Coord InCoord;

    typedef typename SquareMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::Coord OutCoord;


    bool test()
    {
        this->errorMax *= 30;

//        SquareMapping* map = static_cast<SquareMapping*>( this->mapping );

        // parent positions
        InVecCoord incoord(4);
        incoord[0]= 1;
        incoord[1]= -1;
        incoord[2]= 7;
        incoord[3]= -10;

        // expected child positions
        OutVecCoord expectedoutcoord(4);
        expectedoutcoord[0]= 1;
        expectedoutcoord[1]= 1;
        expectedoutcoord[2]= 49;
        expectedoutcoord[3]= 100;

        return this->runTest( incoord, expectedoutcoord );
    }

};


// Define the list of types to instanciate.
using testing::Types;
typedef Types<
component::mapping::SquareMapping<defaulttype::Vec1Types,defaulttype::Vec1Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE( SquareMappingTest, DataTypes );

// test case
TYPED_TEST( SquareMappingTest , test )
{
    ASSERT_TRUE(this->test());
}



} // namespace
} // namespace sofa
