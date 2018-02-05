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

#include <SofaTest/Sofa_test.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/Multi2Mapping.inl>
#include <sofa/simulation/VectorOperations.h>
#include <SofaBaseLinearSolver/FullVector.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <SceneCreator/SceneCreator.h>
#include <sofa/helper/vector.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <Flexible/types/AffineTypes.h>
#include <mapping/RigidScaleToAffineMultiMapping.h>

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

/**
 * Test suite for RigidScaleToAffineMultiMapping.
 */
template <typename _MultiMapping>
struct RigidScaleToAffineMultiMappingTest  : public Multi2Mapping_test<_MultiMapping>
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

    typedef mapping::RigidScaleToAffineMultiMapping<In1Type, In2Type, OutType> RigidScaleToAffineMultiMapping;
	
    /** @name Test_Cases
     * For each of these cases, we can test if the mapping work
     */
    ///@{
    /**
     * Two parent particles, two children
     */
    bool test_two_parents_one_child()
    {
        this->setupScene(); // 1 child

        RigidScaleToAffineMultiMapping* smm = static_cast<RigidScaleToAffineMultiMapping*>(this->mapping);

        // parent positions
        // -- rigid
        vector< In1VecCoord > incoord1(1);
        incoord1[0].resize(2);
        In1Type::set(incoord1[0][0], 0., 0., 0.);
        In1Type::set(incoord1[0][1], 0., 1., 0.);
        // -- scale (vec3)
        vector< In2VecCoord > incoord2(1);
        incoord2[0].resize(2);
        In2Type::set(incoord2[0][0], 2., 2., 2.);
        In2Type::set(incoord2[0][1], 3., 8., 5.);

        // index
        vector<unsigned> index(6);
        index[0] = 0; index[1] = 0; index[2] = 0;
        index[3] = 1; index[4] = 1; index[5] = 1;
        smm->index.setValue(index);  // parent, index in parent

        // activation of geometric stiffness
        smm->useGeometricStiffness.setValue(1);

        // expected child positions
        OutCoord a1, a2;
        // Init of a1, a2;
        // -- a1
        a1[0] = 0; a1[1] = 0; a1[2] = 0;
        a1[3] = 2; a1[4] = 0; a1[5] = 0; a1[6] = 0; a1[7] = 2; a1[8] = 0; a1[9] = 0; a1[10] = 0; a1[11] = 2;
        // -- a2
        a2[0] = 0; a2[1] = 1; a2[2] = 0;
        a2[3] = 3; a2[4] = 0; a2[5] = 0; a2[6] = 0; a2[7] = 8; a2[8] = 0; a2[9] = 0; a2[10] = 0; a2[11] = 5;

        OutVecCoord outcoords(2);
        outcoords[0] = a1;
        outcoords[1] = a2;

        return this->runTest(incoord1, incoord2, outcoords);
    }
};

// Define the list of types to instantiate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<mapping::RigidScaleToAffineMultiMapping<defaulttype::Rigid3Types, defaulttype::Vec3Types, defaulttype::Affine3Types> > DataTypes; // the types to instantiate.

// Test suite for all the instantiations
TYPED_TEST_CASE(RigidScaleToAffineMultiMappingTest, DataTypes);
// first test case
TYPED_TEST( RigidScaleToAffineMultiMappingTest , two_parents_one_child )
{
    ASSERT_TRUE(this->test_two_parents_one_child());
}

} // namespace
} // namespace sofa
