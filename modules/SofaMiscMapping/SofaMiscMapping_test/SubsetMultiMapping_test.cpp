/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

/* Francois Faure, 2013 */
#include <SofaTest/MultiMapping_test.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
//#include <sofa/defaulttype/RigidTypes.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>


namespace sofa {
namespace {

using namespace core;
using namespace component;
using defaulttype::Vec;
using defaulttype::Mat;
using sofa::helper::vector;


/**  Test suite for SubsetMultiMapping.
  */
template <typename _SubsetMultiMapping>
struct SubsetMultiMappingTest : public MultiMapping_test<_SubsetMultiMapping>
{

    typedef _SubsetMultiMapping SubsetMultiMapping;

    typedef typename SubsetMultiMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef container::MechanicalObject<InDataTypes> InMechanicalObject;
    typedef typename InMechanicalObject::ReadVecCoord  ReadInVecCoord;
    typedef typename InMechanicalObject::WriteVecCoord WriteInVecCoord;
    typedef typename InMechanicalObject::WriteVecDeriv WriteInVecDeriv;
    typedef typename InDataTypes::Real InReal;
    typedef Mat<InDataTypes::spatial_dimensions,InDataTypes::spatial_dimensions,InReal> RotationMatrix;


    typedef typename SubsetMultiMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    typedef typename OutDataTypes::Coord OutCoord;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef container::MechanicalObject<OutDataTypes> OutMechanicalObject;
    typedef typename OutMechanicalObject::WriteVecCoord WriteOutVecCoord;
    typedef typename OutMechanicalObject::WriteVecDeriv WriteOutVecDeriv;
    typedef typename OutMechanicalObject::ReadVecCoord ReadOutVecCoord;
    typedef typename OutMechanicalObject::ReadVecDeriv ReadOutVecDeriv;

    /** @name Test_Cases
      For each of these cases, we can test if the mapping work
      */
    ///@{
    /** Two parent particles, two children
    */
    bool test_two_parents_one_child()
    {
        const int NP = 2;
        this->setupScene(NP); // NP parents, 1 child
        SubsetMultiMapping* smm = static_cast<SubsetMultiMapping*>( this->mapping );

        // parent positions
        vector< InVecCoord > incoords(NP);
        for( int i=0; i<NP; i++ )
        {
            incoords[i].resize(1);
            InDataTypes::set( incoords[i][0], i+1.,-2., 3. );
        }

        // subset
        smm->addPoint(smm->getMechFrom()[0],0);  // parent, index in parent
        smm->addPoint(smm->getMechFrom()[1],0);  // parent, index in parent

        // expected child positions
        OutVecCoord outcoords(2);
        OutDataTypes::set( outcoords[0], 1.  , -2., 3. );
        OutDataTypes::set( outcoords[1], 1+1., -2., 3. );

        return this->runTest(incoords,outcoords);
    }


    ///@}


};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
mapping::SubsetMultiMapping<defaulttype::Rigid3Types,defaulttype::Rigid3Types>,
mapping::SubsetMultiMapping<defaulttype::Vec3Types,defaulttype::Vec3Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(SubsetMultiMappingTest, DataTypes);
// first test case
TYPED_TEST( SubsetMultiMappingTest , two_parents_one_child )
{
    ASSERT_TRUE(this->test_two_parents_one_child());
}

} // namespace
} // namespace sofa
