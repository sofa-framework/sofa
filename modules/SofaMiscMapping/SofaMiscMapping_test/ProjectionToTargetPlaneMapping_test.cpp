/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaMiscMapping/ProjectionToPlaneMapping.h>


namespace sofa {
namespace {

using namespace core;
using namespace component;
using defaulttype::Vec;
using defaulttype::Mat;
using sofa::helper::vector;


/**  Test suite for ProjectToTargetLineMapping.
  */
template <typename ProjectionToTargetPlaneMapping>
struct ProjectionToTargetPlaneMappingTest : public Mapping_test<ProjectionToTargetPlaneMapping>
{
    typedef typename ProjectionToTargetPlaneMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::Coord InCoord;

    typedef typename ProjectionToTargetPlaneMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::Coord OutCoord;


    bool test()
    {
        ProjectionToTargetPlaneMapping* pttlm = static_cast<ProjectionToTargetPlaneMapping*>( this->mapping );

        // parent positions
        InVecCoord incoord(1);
        InDataTypes::set( incoord[0], 0,0,0 );

        // expected child positions
        OutVecCoord expectedoutcoord;

        // mapping data
        helper::WriteAccessor< Data<vector<unsigned> > > indices(pttlm->f_indices);
        helper::WriteAccessor< Data<OutVecCoord> > origins(pttlm->f_origins);
        helper::WriteAccessor< Data<OutVecCoord> > normals(pttlm->f_normals);

        indices.push_back( 0 );
        origins.push_back( OutCoord(0,0,0) );
        normals.push_back( OutCoord(1,0,0) );
        expectedoutcoord.push_back( OutCoord(0,0,0) );

        indices.push_back( 0 );
        origins.push_back( OutCoord(0,1,0) );
        normals.push_back( OutCoord(0,1,0) );
        expectedoutcoord.push_back( OutCoord(0,1,0) );

        return this->runTest( incoord, expectedoutcoord );
    }

};


// Define the list of types to instanciate.
using testing::Types;
typedef Types<
mapping::ProjectionToTargetPlaneMapping<defaulttype::Vec3Types,defaulttype::Vec3Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE( ProjectionToTargetPlaneMappingTest, DataTypes );

// test case
TYPED_TEST( ProjectionToTargetPlaneMappingTest , test )
{
    ASSERT_TRUE(this->test());
}

} // namespace
} // namespace sofa
