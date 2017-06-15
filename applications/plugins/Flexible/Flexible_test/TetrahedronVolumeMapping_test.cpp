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

#include "stdafx.h"
#include <SofaTest/Mapping_test.h>
#include <sofa/defaulttype/VecTypes.h>
#include <Flexible/deformationMapping/TetrahedronVolumeMapping.h>
#include <SofaBaseTopology/MeshTopology.h>


namespace sofa {
  namespace {
using std::cout;
using std::cerr;
using std::endl;
using namespace core;
using namespace component;
using defaulttype::Vec;
using defaulttype::Mat;
using mapping::TetrahedronVolumeMapping;


/**  Test suite for TetrahedronVolumeMapping.
  */
template <typename _TestedMapping>
struct TetrahedronVolumeMappingTest : public Mapping_test<_TestedMapping>
{

    typedef _TestedMapping TestedMapping;

    typedef typename TestedMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename TestedMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;


    TestedMapping* _testedMapping;

    TetrahedronVolumeMappingTest(){
        _testedMapping = static_cast<TestedMapping*>( this->mapping );
    }


    bool test( bool perNode )
    {
        component::topology::MeshTopology::SPtr topology = modeling::addNew<component::topology::MeshTopology>(this->root);
        topology->addPoint( 0,0,0 );
        topology->addPoint( 1,0,0 );
        topology->addPoint( 0,1,0 );
        topology->addPoint( 0,0,1 );
        topology->addTetra( 0,1,2,3 );

        const int Nin=4, Nout=perNode?4:1;
        this->inDofs->resize(Nin);
        this->outDofs->resize(Nout);

        _testedMapping->d_volumePerNodes.setValue(perNode);

        // parent position
        InVecCoord xin(Nin);
        InDataTypes::set( xin[0], 0,0,0 );
        InDataTypes::set( xin[1], 1,0,0 );
        InDataTypes::set( xin[2], 0,1,0 );
        InDataTypes::set( xin[3], 0,0,1 );

        // child position
        OutVecCoord xout(Nout);

        // expected mapped values
        OutVecCoord expectedChildCoords(Nout);
        SReal expectedVolume = 1. / 6. / (SReal)Nout;
        for(int i=0; i<Nout; i++ )
            expectedChildCoords[i] = expectedVolume;

        return this->runTest(xin,xout,xin,expectedChildCoords);
    }


};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
TetrahedronVolumeMapping<defaulttype::Vec3Types,defaulttype::Vec1Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(TetrahedronVolumeMappingTest, DataTypes);

TYPED_TEST( TetrahedronVolumeMappingTest, test_perTetra )
{
    ASSERT_TRUE( this->test( false ) );
}
TYPED_TEST( TetrahedronVolumeMappingTest, test_perNode )
{
    ASSERT_TRUE( this->test( true ) );
}

}//anonymous namespace
} // namespace sofa
