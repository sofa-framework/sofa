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

#include "stdafx.h"
#include <SofaTest/Mapping_test.h>
#include <sofa/defaulttype/VecTypes.h>
#include <Flexible/deformationMapping/VolumeMapping.h>
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
using mapping::VolumeMapping;


/**  Test suite for VolumeMapping.
  */
template <typename _TestedMapping>
struct VolumeMappingTest : public Mapping_test<_TestedMapping>
{

    typedef _TestedMapping TestedMapping;

    typedef typename TestedMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename TestedMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;


    TestedMapping* _testedMapping;

    VolumeMappingTest(){
        _testedMapping = static_cast<TestedMapping*>( this->mapping );
    }


    bool test( )
    {
        const int Nin=4, Nout=1;
        this->inDofs->resize(Nin);
        this->outDofs->resize(Nout);

        // parent position
        InVecCoord xin(Nin);
        InDataTypes::set( xin[0], 0,0,0 );
        InDataTypes::set( xin[1], 1,0,0 );
        InDataTypes::set( xin[2], 0,1,0 );
        InDataTypes::set( xin[3], 0,0,1 );

        // topology
        helper::WriteAccessor< Data< component::topology::MeshTopology::SeqTriangles > > tri0(_testedMapping->vf_triangles[0]);
        tri0.push_back(component::topology::MeshTopology::Triangle( 0,2,1 ));
        tri0.push_back(component::topology::MeshTopology::Triangle( 0,1,3 ));
        tri0.push_back(component::topology::MeshTopology::Triangle( 0,3,2 ));
        tri0.push_back(component::topology::MeshTopology::Triangle( 1,2,3 ));

        _testedMapping->f_geometricStiffness.setValue(true);

        // child position
        OutVecCoord xout(Nout);

        // expected mapped values
        OutVecCoord expectedChildCoords(Nout);
        SReal expectedVolume = 1. / 6.;
        for(int i=0; i<Nout; i++ )
            expectedChildCoords[i] = expectedVolume;

        return this->runTest(xin,xout,xin,expectedChildCoords);
    }


};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
VolumeMapping<defaulttype::Vec3Types,defaulttype::Vec1Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(VolumeMappingTest, DataTypes);

TYPED_TEST( VolumeMappingTest, test_auto )
{
    ASSERT_TRUE( this->test( ) );
}

}//anonymous namespace
} // namespace sofa
