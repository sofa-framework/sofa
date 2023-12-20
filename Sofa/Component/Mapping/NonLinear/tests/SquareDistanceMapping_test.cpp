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

#include <sofa/component/mapping/nonlinear/SquareDistanceMapping.h>

#include <sofa/component/mapping/testing/MappingTestCreation.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/simulation/graph/SimpleApi.h>

namespace sofa {
namespace {


/**  Test suite for SquareDistanceMapping.
 *
 * @author Matthieu Nesme
  */
template <typename SquareDistanceMapping>
struct SquareDistanceMappingTest : public sofa::mapping_test::Mapping_test<SquareDistanceMapping>
{
    typedef typename SquareDistanceMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::Coord InCoord;

    typedef typename SquareDistanceMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::Coord OutCoord;


    bool test()
    {
        this->errorMax *= 10;

        SquareDistanceMapping* map = static_cast<SquareDistanceMapping*>( this->mapping );
//        map->f_computeDistance.setValue(true);
        sofa::helper::getWriteAccessor(map->d_geometricStiffness)->setSelectedItem(1);

        const component::topology::container::dynamic::EdgeSetTopologyContainer::SPtr edges = sofa::core::objectmodel::New<component::topology::container::dynamic::EdgeSetTopologyContainer>();
        this->root->addObject(edges);
        edges->addEdge( 0, 1 );
        edges->addEdge( 2, 1 );

        // parent positions
        InVecCoord incoord(3);
        InDataTypes::set( incoord[0], 0,0,0 );
        InDataTypes::set( incoord[1], 1,1,1 );
        InDataTypes::set( incoord[2], 6,3,-1 );

        // expected child positions
        OutVecCoord expectedoutcoord;
        expectedoutcoord.push_back( type::Vec1( 3 ) );
        expectedoutcoord.push_back( type::Vec1( 33 ) );

        return this->runTest( incoord, expectedoutcoord );
    }

//    bool test_restLength()
//    {
//        this->errorMax *= 10;

//        SquareDistanceMapping* map = static_cast<SquareDistanceMapping*>( this->mapping );
////        map->f_computeDistance.setValue(true);
//        map->d_geometricStiffness.setValue(1);

//        type::vector< SReal > restLength(2);
//        restLength[0] = .5;
//        restLength[1] = 2;
//        map->f_restLengths.setValue( restLength );

//        component::topology::container::dynamic::EdgeSetTopologyContainer::SPtr edges = modeling::addNew<component::topology::container::dynamic::EdgeSetTopologyContainer>(this->root);
//        edges->addEdge( 0, 1 );
//        edges->addEdge( 2, 1 );

//        // parent positions
//        InVecCoord incoord(3);
//        InDataTypes::set( incoord[0], 0,0,0 );
//        InDataTypes::set( incoord[1], 1,1,1 );
//        InDataTypes::set( incoord[2], 6,3,-1 );

//        // expected child positions
//        OutVecCoord expectedoutcoord;
//        expectedoutcoord.push_back( type::Vector1( (sqrt(3.)-.5) * (sqrt(3.)-.5) ) );
//        expectedoutcoord.push_back( type::Vector1( (sqrt(33.)-2.) * (sqrt(33.)-2.) ) );

//        return this->runTest( incoord, expectedoutcoord );
//    }

};


// Define the list of types to instanciate.
using ::testing::Types;
typedef Types<
component::mapping::nonlinear::SquareDistanceMapping<defaulttype::Vec3Types,defaulttype::Vec1Types>
, component::mapping::nonlinear::SquareDistanceMapping<defaulttype::Rigid3Types,defaulttype::Vec1Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_SUITE( SquareDistanceMappingTest, DataTypes );

// test case
TYPED_TEST( SquareDistanceMappingTest , test )
{
    ASSERT_TRUE(this->test());
}

//TYPED_TEST( SquareDistanceMappingTest , test_restLength )
//{
//    ASSERT_TRUE(this->test_restLength());
//}


} // namespace

/**
 * This test checks two different methods to simulate a triple pendulum using quadratic springs:
 * 1) A SquaredDistanceMapping is used to transform the DOFs from the 3d space to a 1d space
 * (representing the squared distances between the DOFs). Then a spring acts in the 1d space
 * 2) A combination of two mappings: DistanceMapping and SquareMapping. It also transforms the 3d
 * space to a 1d space where a spring is added.
 *
 * Two methods are supposed to lead to the same result. This test checks that both mechanical
 * objects are at the same position, velocity and force.
 * However, this test can fail easily with different parameters (number of strings, number of time
 * steps etc).
 */
struct SquareDistanceMappingCompare_test : NumericTest<SReal>
{
    simulation::Node::SPtr root;
    simulation::Node::SPtr oneMapping;
    simulation::Node::SPtr twoMappings;

    void onSetUp() override
    {
        root = simulation::getSimulation()->createNewNode("root");

        simpleapi::createObject(root, "RequiredPlugin", {{"pluginName", "Sofa.Component"}});
        simpleapi::createObject(root, "DefaultAnimationLoop");
        simpleapi::createObject(root, "StringMeshCreator", {{"name", "loader"}, {"resolution", "3"}});

        oneMapping = simpleapi::createChild(root, "oneMapping");
        twoMappings = simpleapi::createChild(root, "twoMappings");

        for (const auto& node : {oneMapping, twoMappings})
        {
            simpleapi::createObject(node, "EulerImplicitSolver", {{"rayleighStiffness", "0.1"}, {"rayleighMass","0.1"}});
            simpleapi::createObject(node, "EdgeSetTopologyContainer",
                {{"position", "@../loader.position"}, {"edges", "@../loader.edges"}, {"name", "topology"}});
            simpleapi::createObject(node, "MechanicalObject", {{"name", "defoDOF"}, {"template", "Vec3"}});
            simpleapi::createObject(node, "EdgeSetGeometryAlgorithms");
            simpleapi::createObject(node, "FixedProjectiveConstraint", {{"indices", "0"}});
            simpleapi::createObject(node, "DiagonalMass", {{"totalMass", "1e-2"}});
        }

        const auto oneMappingExtension = simpleapi::createChild(oneMapping, "extensionsNode");
        const auto twoMappingsExtension = simpleapi::createChild(twoMappings, "extensionsNode");

        simpleapi::createObject(oneMappingExtension, "MechanicalObject", {{"name", "extensionsDOF"}, {"template", "Vec1"}});
        simpleapi::createObject(twoMappingsExtension, "MechanicalObject", {{"name", "extensionsDOF"}, {"template", "Vec1"}});

        simpleapi::createObject(oneMappingExtension, "SquareDistanceMapping",
                                {{"topology", "@../topology"}, {"input", "@../defoDOF"},
                                 {"output", "@extensionsDOF"}, {"geometricStiffness", "1"},
                                 {"applyRestPosition", "true"}});
        simpleapi::createObject(oneMappingExtension, "RestShapeSpringsForceField", {{"template", "Vec1"}, {"stiffness", "10000"}});




        simpleapi::createObject(twoMappingsExtension, "DistanceMapping",
                                {{"topology", "@../topology"}, {"input", "@../defoDOF"},
                                 {"output", "@extensionsDOF"}, {"geometricStiffness", "1"},
                                 {"applyRestPosition", "true"}, {"computeDistance", "true"}});
        const auto distanceMappingNode = simpleapi::createChild(twoMappingsExtension, "square");
        simpleapi::createObject(distanceMappingNode, "MechanicalObject", {{"name", "squaredDOF"}, {"template", "Vec1"}});
        simpleapi::createObject(distanceMappingNode, "SquareMapping",
                                        {{"input", "@../extensionsDOF"},
                                         {"output", "@squaredDOF"}, {"geometricStiffness", "1"},
                                         {"applyRestPosition", "true"}});
        simpleapi::createObject(distanceMappingNode, "RestShapeSpringsForceField", {{"template", "Vec1"}, {"stiffness", "10000"}});

    }

    void compareMechanicalObjects(unsigned int timeStepCount, SReal epsilon)
    {
        core::behavior::BaseMechanicalState* mstate0 = oneMapping->getMechanicalState();
        ASSERT_NE(mstate0, nullptr);

        core::behavior::BaseMechanicalState* mstate1 = twoMappings->getMechanicalState();
        ASSERT_NE(mstate1, nullptr);

        //position
        {
            sofa::type::vector<SReal> mstatex0(mstate0->getMatrixSize());
            mstate0->copyToBuffer(mstatex0.data(), core::ConstVecCoordId::position(), mstate0->getMatrixSize());

            sofa::type::vector<SReal> mstatex1(mstate1->getMatrixSize());
            mstate1->copyToBuffer(mstatex1.data(), core::ConstVecCoordId::position(), mstate1->getMatrixSize());

            EXPECT_LT(this->vectorMaxDiff(mstatex0, mstatex1), epsilon) << "Time step " << timeStepCount
                << "\n" << mstatex0 << "\n" << mstatex1;
        }

        //velocity
        {
            sofa::type::vector<SReal> mstatev0(mstate0->getMatrixSize());
            mstate0->copyToBuffer(mstatev0.data(), core::ConstVecDerivId::velocity(), mstate0->getMatrixSize());

            sofa::type::vector<SReal> mstatev1(mstate1->getMatrixSize());
            mstate1->copyToBuffer(mstatev1.data(), core::ConstVecDerivId::velocity(), mstate1->getMatrixSize());

            EXPECT_LT(this->vectorMaxDiff(mstatev0, mstatev1), epsilon) << "Time step " << timeStepCount
                << "\n" << mstatev0 << "\n" << mstatev1;
        }

        //force
        {
            sofa::type::vector<SReal> mstatef0(mstate0->getMatrixSize());
            mstate0->copyToBuffer(mstatef0.data(), core::ConstVecDerivId::force(), mstate0->getMatrixSize());

            sofa::type::vector<SReal> mstatef1(mstate1->getMatrixSize());
            mstate1->copyToBuffer(mstatef1.data(), core::ConstVecDerivId::force(), mstate1->getMatrixSize());

            EXPECT_LT(this->vectorMaxDiff(mstatef0, mstatef1), epsilon) << "Time step " << timeStepCount
                << "\n" << mstatef0 << "\n" << mstatef1;
        }
    }
};

TEST_F(SquareDistanceMappingCompare_test, compareToDistanceMappingAndSquareMappingCG)
{
    for (const auto& node : {oneMapping, twoMappings})
    {
        simpleapi::createObject(node, "CGLinearSolver", {{"iterations", "1e4"}, {"tolerance", "1.0e-9"}, {"threshold", "1.0e-9"}});
    }

    sofa::simulation::node::initRoot(root.get());

    for (unsigned int i = 0 ; i < 100; ++i)
    {
        sofa::simulation::node::animate(root.get(), 0.01_sreal);

        compareMechanicalObjects(i, 1e-7_sreal);
    }
}

TEST_F(SquareDistanceMappingCompare_test, compareToDistanceMappingAndSquareMappingLU)
{
    for (const auto& node : {oneMapping, twoMappings})
    {
        simpleapi::createObject(node, "EigenSparseLU", {{"template", "CompressedRowSparseMatrixMat3x3d"}});
    }

    sofa::simulation::node::initRoot(root.get());

    for (unsigned int i = 0 ; i < 100; ++i)
    {
        sofa::simulation::node::animate(root.get(), 0.01_sreal);

        compareMechanicalObjects(i, 1e-10_sreal);
    }
}



} // namespace sofa
