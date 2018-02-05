/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sstream>

#include <Compliant/utils/se3.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/Multi2Mapping.inl>

#include "../miscMapping/RigidScaleToRigidMultiMapping.h"

#include <SofaTest/Multi2Mapping_test.h>


/**
 * @author Ali Dicko @date 2015
 */
namespace sofa
{
namespace
{

using std::cout;
using std::cerr;
using std::endl;
using namespace core;
using namespace component;
using defaulttype::Vec;
using defaulttype::Mat;
using sofa::helper::vector;
typedef std::size_t Index;

/**  Test suite for RigidScaleToRigidMultiMapping.
  */
template <typename _MultiMapping>
struct RigidScaleToRigidMultiMappingTest : public Multi2Mapping_test<_MultiMapping>
{
    typedef _MultiMapping Mapping;

    typedef typename Mapping::In1 In1Type;
    typedef typename In1Type::VecCoord In1VecCoord;
    typedef typename In1Type::VecDeriv In1VecDeriv;
    typedef typename In1Type::Coord In1Coord;
    typedef typename In1Type::Deriv In1Deriv;

    typedef typename Mapping::In2 In2Type;
    typedef typename In2Type::VecCoord In2VecCoord;
    typedef typename In2Type::VecDeriv In2VecDeriv;
    typedef typename In2Type::Coord In2Coord;
    typedef typename In2Type::Deriv In2Deriv;

    typedef typename Mapping::Out OutType;
    typedef typename OutType::VecCoord OutVecCoord;
    typedef typename OutType::VecDeriv OutVecDeriv;
    typedef typename OutType::Coord OutCoord;
    typedef typename OutType::Deriv OutDeriv;

    typedef SE3< typename In1Type::Real > se3;

    typedef mapping::RigidScaleToRigidMultiMapping<In1Type, In2Type, OutType> RigidScaleToRigidMultiMapping;

    /** @name Test_Cases
     * For each of these cases, we can test if the mapping work
     */
    ///@{
    /**
     * Two parent particles, two children
     */
    bool test()
    {
        this->setupScene(); // 1 child
        RigidScaleToRigidMultiMapping* smm = static_cast<RigidScaleToRigidMultiMapping*>(this->mapping);

        // parent positions
        // -- rigid
        vector< In1VecCoord > incoord1(1);
        incoord1[0].resize(2);
        In1Type::set(incoord1[0][0], 0, 0, 0);
        In1Type::set(incoord1[0][1],-1, 1, 1);

        // -- scale (vec3)
        vector< In2VecCoord > incoord2(1);
        incoord2[0].resize(2);
        //In2Type::set(incoord2[0][0], 2., 2., 2.);
        //In2Type::set(incoord2[0][1], 3., 8., 5.);
        In2Type::set(incoord2[0][0], 1, 1, 1);
        In2Type::set(incoord2[0][1], 1, 1, 1);

        // initial relative position
        vector< OutVecCoord > initcoord(1);
        initcoord[0].resize(2);
        OutType::set(initcoord[0][0],-1, 0, 0);
        OutType::set(initcoord[0][1],-2, 1, 3);
        (this->outDofs->x).setValue(initcoord[0]);
        (this->outDofs->x0).setValue(initcoord[0]);

        // index
        vector<unsigned> index(6);
        index[0] = 0; index[1] = 0; index[2] = 0;
        index[3] = 1; index[4] = 1; index[5] = 1;
        smm->index.setValue(index);  // parent, index in parent
        // activation of geometric stiffness
        smm->useGeometricStiffness.setValue(1);

        // expected child positions
        OutCoord r1, r2;
        // Init of the output rigid;
        // -- r1
        //r1[0] =-2; r1[1] = 0; r1[2] = 0;
        r1[0] =-1; r1[1] = 0; r1[2] = 0;
        r1[3] = 0; r1[4] = 0; r1[5] = 0; r1[6] = 1;
        // -- r2
        //r2[0] =-7; r2[1] = 9; r2[2] = 16;
        r2[0] =-3; r2[1] = 2; r2[2] = 4;
        r2[3] = 0; r2[4] = 0; r2[5] = 0; r2[6] = 1;
        OutVecCoord outcoords(2);
        outcoords[0] = r1;
        outcoords[1] = r2;

		return this->runTest(incoord1, incoord2, outcoords);
    }

    bool test_basic()
    {
        this->setupScene(); // 1 child
        RigidScaleToRigidMultiMapping* smm = static_cast<RigidScaleToRigidMultiMapping*>(this->mapping);

        // parent positions
        // -- rigid
        vector< In1VecCoord > incoord1(1);
        incoord1[0].resize(1);
        In1Type::set(incoord1[0][0], 0, 0, 0);

        // -- scale (vec3)
        vector< In2VecCoord > incoord2(1);
        incoord2[0].resize(1);
        In2Type::set(incoord2[0][0], 1, 1, 1);
        // initial relative position
        vector< OutVecCoord > initcoord(1);
        initcoord[0].resize(1);
        OutType::set(initcoord[0][0],-1, 0, 0);
        (this->outDofs->x).setValue(initcoord[0]);
        (this->outDofs->x0).setValue(initcoord[0]);

        // index
        vector<unsigned> index(3);
        index[0] = 0; index[1] = 0; index[2] = 0;
        smm->index.setValue(index);  // parent, index in parent
        // activation of geometric stiffness
        smm->useGeometricStiffness.setValue(1);

        // expected child positions
        OutCoord r1;
        // Init of the output rigid;
        // -- r1
        r1[0] =-1; r1[1] = 0; r1[2] = 0;
        r1[3] = 0; r1[4] = 0; r1[5] = 0; r1[6] = 1;
        OutVecCoord outcoords(1);
        outcoords[0] = r1;

        return this->runTest(incoord1, incoord2, outcoords);
    }

    bool test_AssembledRigidRigidMapping()
    {
        // init
        this->setupScene(); // 1 child
        RigidScaleToRigidMultiMapping* smm = static_cast<RigidScaleToRigidMultiMapping*>(this->mapping);

        // // parent, index in parent
        vector<unsigned> index(3);
        index[0] = 0; index[1] = 0; index[2] = 0;
        smm->index.setValue(index);
        // activation of geometric stiffness
        smm->useGeometricStiffness.setValue(1);

        // -- rigid
        In1VecCoord xin(1);
        typename se3::vec3 v;
        v << M_PI/3.0, 0, 0;
        xin[0].getOrientation() = se3::coord( se3::exp(v) );
        se3::map(xin[0].getCenter()) << 1, -5, 20;
        vector< In1VecCoord > incoord1(1);
        incoord1[0].push_back(xin[0]);

        // -- scale (vec3)
        vector< In2VecCoord > incoord2(1);
        incoord2[0].resize(1);
        In2Type::set(incoord2[0][0], 1., 1., 1.);

        // offset
        OutCoord offset;
        v << 0, M_PI/4, 0;
        offset.getOrientation() = se3::coord( se3::exp(v) );
        se3::map(offset.getCenter()) << 0, 1, 0;
        vector< OutVecCoord > initcoord(1);
        initcoord[0].push_back(offset);
        (this->outDofs->x).setValue(initcoord[0]);
        (this->outDofs->x0).setValue(initcoord[0]);

        OutVecCoord expected(1);
        expected[0].getOrientation() = xin[0].getOrientation() * offset.getOrientation();
        expected[0].getCenter() = xin[0].getOrientation().rotate( offset.getCenter() ) + xin[0].getCenter();

        return this->runTest(incoord1, incoord2, expected);
    }
};


// Define the list of types to instantiate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<mapping::RigidScaleToRigidMultiMapping<defaulttype::Rigid3Types, defaulttype::Vec3Types, defaulttype::Rigid3Types> > DataTypes; // the types to instantiate.

// Test suite for all the instantiations
TYPED_TEST_CASE(RigidScaleToRigidMultiMappingTest, DataTypes);
// first test case
TYPED_TEST( RigidScaleToRigidMultiMappingTest , test)
{
    ASSERT_TRUE(this->test_basic());
}

} // namespace
} // namespace sofa
