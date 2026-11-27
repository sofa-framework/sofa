/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <gtest/gtest.h>
#include <sofa/component/mass/FEMMass.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/testing/BaseSimulationTest.h>
#include <sofa/testing/LinearCongruentialRandomGenerator.h>

namespace sofa::component::mass::testing
{

using MeshTopology = sofa::component::topology::container::constant::MeshTopology;

namespace
{
struct MatrixDiagonalCheckerAccumulator : public sofa::core::behavior::MassMatrixAccumulator
{
    bool hasOffDiagonalNonZero { false };
    bool hasDiagonalNonZero { false };
    double tolerance { 1e-12 };

    void checkEntry(sofa::SignedIndex row, sofa::SignedIndex col, double value)
    {
        if (std::abs(value) > tolerance)
        {
            if (row == col)
            {
                hasDiagonalNonZero = true;
            }
            else
            {
                hasOffDiagonalNonZero = true;
            }
        }
    }

    void add(sofa::SignedIndex row, sofa::SignedIndex col, float value) override
    {
        checkEntry(row, col, static_cast<double>(value));
    }

    void add(sofa::SignedIndex row, sofa::SignedIndex col, double value) override
    {
        checkEntry(row, col, value);
    }

    void clear() override
    {
        hasOffDiagonalNonZero = false;
        hasDiagonalNonZero = false;
    }
};

struct MatrixSumAccumulator : public sofa::core::behavior::MassMatrixAccumulator
{
    double sum { 0 };
    sofa::SignedIndex maxIndex { -1 };

    void add(sofa::SignedIndex row, sofa::SignedIndex col, float value) override
    {
        add(row, col, static_cast<double>(value));
    }

    void add(sofa::SignedIndex row, sofa::SignedIndex col, double value) override
    {
        sum += value;
        maxIndex = std::max({maxIndex, row, col});
    }

    void clear() override
    {
        sum = 0;
        maxIndex = -1;
    }
};
} // namespace

template <typename MassParam>
struct FEMMassLumpingTest : public sofa::testing::BaseSimulationTest
{
    using DataTypes = typename MassParam::DataTypes;
    using SpatialTypes = typename MassParam::SpatialTypes;
    using ElementType = typename MassParam::ElementType;
    using FEMMassType = FEMMass<DataTypes, SpatialTypes, ElementType>;
    using DOF = sofa::component::statecontainer::MechanicalObject<DataTypes>;
    using SpatialDOF = sofa::component::statecontainer::MechanicalObject<SpatialTypes>;

    typename simulation::Node::SPtr m_root;
    typename simulation::Node::SPtr m_node; ///< node containing the field state and the mass
    typename DOF::SPtr m_dof;
    typename SpatialDOF::SPtr m_spatialDof;
    typename FEMMassType::SPtr m_mass;
    typename MeshTopology::SPtr m_topology;
    typename NodalMassDensity<sofa::Real_t<DataTypes>>::SPtr m_nodalDensity;

    void doSetUp() override
    {
        sofa::simulation::Simulation* simu = sofa::simulation::getSimulation();
        ASSERT_NE(simu, nullptr);

        m_root = simu->createNewGraph("root");
        m_node = m_root;

        if constexpr (!std::is_same_v<DataTypes, SpatialTypes>)
        {
            // The geometry is held by a separate state. Only one state is allowed per node, so
            // the spatial state is in the root node (initialized first), and the field in a child.
            m_spatialDof = sofa::core::objectmodel::New<SpatialDOF>();
            m_root->addObject(m_spatialDof);
            m_spatialDof->resize(8);
            setRandomPositions(*m_spatialDof);

            m_node = m_root->createChild("field");
        }

        m_dof = sofa::core::objectmodel::New<DOF>();
        m_node->addObject(m_dof);

        m_topology = sofa::core::objectmodel::New<MeshTopology>();
        m_node->addObject(m_topology);

        // Define a simple mesh element with distinct connected vertices
        m_topology->addEdge(0, 1);
        m_topology->addTriangle(0, 1, 2);
        m_topology->addQuad(0, 1, 2, 3);
        m_topology->addTetra(0, 1, 2, 3);
        m_topology->addHexa(0, 1, 2, 3, 4, 5, 6, 7);

        m_nodalDensity = sofa::core::objectmodel::New<NodalMassDensity<sofa::Real_t<DataTypes>>>();
        m_node->addObject(m_nodalDensity);

        m_mass = sofa::core::objectmodel::New<FEMMassType>();
        m_node->addObject(m_mass);

        m_dof->resize(8);

        if constexpr (std::is_same_v<DataTypes, SpatialTypes>)
        {
            setRandomPositions(*m_dof);
        }
        else
        {
            m_mass->l_spatialState.set(m_spatialDof.get());
        }
    }

    template<class State>
    static void setRandomPositions(State& state)
    {
        using StateDataTypes = typename State::DataTypes;
        auto x = state.writePositions();
        sofa::testing::LinearCongruentialRandomGenerator lcg(96547);
        for (std::size_t i = 0; i < 8; ++i)
        {
            StateDataTypes::set(x[i],
                           lcg.generateInUnitRange<sofa::Real_t<StateDataTypes>>(),
                           lcg.generateInUnitRange<sofa::Real_t<StateDataTypes>>(),
                           lcg.generateInUnitRange<sofa::Real_t<StateDataTypes>>());
        }
    }
};

template<class TDataTypes, class TSpatialTypes, class TElementType>
struct MassParam
{
    using DataTypes = TDataTypes;
    using SpatialTypes = TSpatialTypes;
    using ElementType = TElementType;
};

using ElementMassTypes = ::testing::Types<
    MassParam<defaulttype::Vec1Types, defaulttype::Vec1Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec2Types, defaulttype::Vec2Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec3Types, defaulttype::Vec3Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec2Types, defaulttype::Vec2Types, sofa::geometry::Triangle>,
    MassParam<defaulttype::Vec3Types, defaulttype::Vec3Types, sofa::geometry::Triangle>,
    MassParam<defaulttype::Vec2Types, defaulttype::Vec2Types, sofa::geometry::Quad>,
    MassParam<defaulttype::Vec3Types, defaulttype::Vec3Types, sofa::geometry::Quad>,
    MassParam<defaulttype::Vec3Types, defaulttype::Vec3Types, sofa::geometry::Tetrahedron>,
    MassParam<defaulttype::Vec3Types, defaulttype::Vec3Types, sofa::geometry::Hexahedron>,
    // MassParam<defaulttype::Vec3Types, defaulttype::Vec3Types, sofa::geometry::Prism>,
    // MassParam<defaulttype::Vec3Types, defaulttype::Vec3Types, sofa::geometry::Pyramid>

    // field type different from the spatial type
    MassParam<defaulttype::Vec1Types, defaulttype::Vec3Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec1Types, defaulttype::Vec2Types, sofa::geometry::Triangle>,
    MassParam<defaulttype::Vec1Types, defaulttype::Vec3Types, sofa::geometry::Triangle>,
    MassParam<defaulttype::Vec1Types, defaulttype::Vec3Types, sofa::geometry::Tetrahedron>,
    MassParam<defaulttype::Vec1Types, defaulttype::Vec3Types, sofa::geometry::Hexahedron>,
    MassParam<defaulttype::Vec2Types, defaulttype::Vec3Types, sofa::geometry::Tetrahedron>
>;

TYPED_TEST_SUITE(FEMMassLumpingTest, ElementMassTypes);

TYPED_TEST(FEMMassLumpingTest, ConsistentMassMatrixIsNotDiagonal)
{
    this->m_mass->d_lumping.setValue(false);
    sofa::simulation::node::initRoot(this->m_root.get());

    EXPECT_FALSE(this->m_mass->isDiagonal());

    MatrixDiagonalCheckerAccumulator accumulator;
    this->m_mass->buildMassMatrix(&accumulator);

    EXPECT_TRUE(accumulator.hasDiagonalNonZero) << "Consistent mass matrix should contain diagonal entries.";
    EXPECT_TRUE(accumulator.hasOffDiagonalNonZero) << "Consistent mass matrix must have non-zero off-diagonal entries.";
}

TYPED_TEST(FEMMassLumpingTest, LumpedMassMatrixIsDiagonal)
{
    this->m_mass->d_lumping.setValue(true);
    sofa::simulation::node::initRoot(this->m_root.get());

    EXPECT_TRUE(this->m_mass->isDiagonal());

    MatrixDiagonalCheckerAccumulator accumulator;
    this->m_mass->buildMassMatrix(&accumulator);

    EXPECT_TRUE(accumulator.hasDiagonalNonZero) << "Lumped mass matrix must contain diagonal entries.";
    EXPECT_FALSE(accumulator.hasOffDiagonalNonZero) << "Lumped mass matrix must not contain any off-diagonal entries.";
}

namespace
{
/// Builds a scene with a single tetrahedron (unit corner tetrahedron in 3D), and returns the sum of
/// all the entries of the mass matrix computed by a FEMMass<DataTypes, Vec3Types, Tetrahedron>.
template<class DataTypes>
MatrixSumAccumulator computeTetrahedronMassMatrixSum()
{
    using SpatialTypes = defaulttype::Vec3Types;
    using FEMMassType = FEMMass<DataTypes, SpatialTypes, sofa::geometry::Tetrahedron>;

    auto* simu = sofa::simulation::getSimulation();
    const auto root = simu->createNewGraph("root");

    // The spatial state is in the root node (initialized first). If the field type differs, the
    // field state is in a child node, as only one state is allowed per node.
    const auto spatialState = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<SpatialTypes>>();
    root->addObject(spatialState);
    spatialState->resize(4);
    {
        auto x = spatialState->writePositions();
        x[0] = {0, 0, 0};
        x[1] = {1, 0, 0};
        x[2] = {0, 1, 0};
        x[3] = {0, 0, 1};
    }

    auto fieldNode = root;
    if constexpr (!std::is_same_v<DataTypes, SpatialTypes>)
    {
        fieldNode = root->createChild("field");
        const auto fieldState = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<DataTypes>>();
        fieldNode->addObject(fieldState);
        fieldState->resize(4);
    }

    const auto topology = sofa::core::objectmodel::New<MeshTopology>();
    fieldNode->addObject(topology);
    topology->addTetra(0, 1, 2, 3);

    fieldNode->addObject(sofa::core::objectmodel::New<NodalMassDensity<sofa::Real_t<DataTypes>>>());

    const auto mass = sofa::core::objectmodel::New<FEMMassType>();
    fieldNode->addObject(mass);
    mass->l_spatialState.set(spatialState.get());

    sofa::simulation::node::initRoot(root.get());
    EXPECT_FALSE(mass->isComponentStateInvalid());

    MatrixSumAccumulator accumulator;
    mass->buildMassMatrix(&accumulator);

    return accumulator;
}
} // namespace

/**
 * A scalar field (Vec1) defined on a tetrahedron living in 3D: the sum of all the entries of the
 * mass matrix must be equal to the mass of the tetrahedron (density * volume).
 * The same tetrahedron with a Vec3 field has the same scalar mass matrix, repeated on each of the
 * 3 field components.
 */
TEST(FEMMass, scalarFieldOnTetrahedronIn3D)
{
    const auto scalar = computeTetrahedronMassMatrixSum<defaulttype::Vec1Types>();
    EXPECT_NEAR(scalar.sum, 1. / 6., 1e-10);
    EXPECT_EQ(scalar.maxIndex, 3);

    const auto vector = computeTetrahedronMassMatrixSum<defaulttype::Vec3Types>();
    EXPECT_NEAR(vector.sum, 3. / 6., 1e-10);
    EXPECT_EQ(vector.maxIndex, 11);
}

} // namespace sofa::component::mass::testing
