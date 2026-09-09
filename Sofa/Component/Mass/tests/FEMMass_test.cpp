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
} // namespace

template <typename MassParam>
struct FEMMassLumpingTest : public sofa::testing::BaseSimulationTest
{
    using DataTypes = typename MassParam::DataTypes;
    using ElementType = typename MassParam::ElementType;
    using FEMMassType = FEMMass<DataTypes, ElementType>;
    using DOF = sofa::component::statecontainer::MechanicalObject<DataTypes>;

    typename simulation::Node::SPtr m_node;
    typename DOF::SPtr m_dof;
    typename FEMMassType::SPtr m_mass;
    typename MeshTopology::SPtr m_topology;
    typename NodalMassDensity<sofa::Real_t<DataTypes>>::SPtr m_nodalDensity;

    void doSetUp() override
    {
        sofa::simulation::Simulation* simu = sofa::simulation::getSimulation();
        ASSERT_NE(simu, nullptr);

        m_node = simu->createNewGraph("root");
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
        typename DOF::WriteVecCoord x = m_dof->writePositions();
        sofa::testing::LinearCongruentialRandomGenerator lcg(96547);
        for (std::size_t i = 0; i < 8; ++i)
        {
            DataTypes::set(x[i],
                           lcg.generateInUnitRange<sofa::Real_t<DataTypes>>(),
                           lcg.generateInUnitRange<sofa::Real_t<DataTypes>>(),
                           lcg.generateInUnitRange<sofa::Real_t<DataTypes>>());
        }
    }
};

template<class TDataTypes, class TElementType>
struct MassParam
{
    using DataTypes = TDataTypes;
    using ElementType = TElementType;
};

using ElementMassTypes = ::testing::Types<
    MassParam<defaulttype::Vec1Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec2Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec2Types, sofa::geometry::Triangle>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Triangle>,
    MassParam<defaulttype::Vec2Types, sofa::geometry::Quad>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Quad>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Tetrahedron>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Hexahedron>
    // MassParam<defaulttype::Vec3Types, sofa::geometry::Prism>,
    // MassParam<defaulttype::Vec3Types, sofa::geometry::Pyramid>
>;

TYPED_TEST_SUITE(FEMMassLumpingTest, ElementMassTypes);

TYPED_TEST(FEMMassLumpingTest, ConsistentMassMatrixIsNotDiagonal)
{
    this->m_mass->d_lumping.setValue(false);
    sofa::simulation::node::initRoot(this->m_node.get());

    EXPECT_FALSE(this->m_mass->isDiagonal());

    MatrixDiagonalCheckerAccumulator accumulator;
    this->m_mass->buildMassMatrix(&accumulator);

    EXPECT_TRUE(accumulator.hasDiagonalNonZero) << "Consistent mass matrix should contain diagonal entries.";
    EXPECT_TRUE(accumulator.hasOffDiagonalNonZero) << "Consistent mass matrix must have non-zero off-diagonal entries.";
}

TYPED_TEST(FEMMassLumpingTest, LumpedMassMatrixIsDiagonal)
{
    this->m_mass->d_lumping.setValue(true);
    sofa::simulation::node::initRoot(this->m_node.get());

    EXPECT_TRUE(this->m_mass->isDiagonal());

    MatrixDiagonalCheckerAccumulator accumulator;
    this->m_mass->buildMassMatrix(&accumulator);

    EXPECT_TRUE(accumulator.hasDiagonalNonZero) << "Lumped mass matrix must contain diagonal entries.";
    EXPECT_FALSE(accumulator.hasOffDiagonalNonZero) << "Lumped mass matrix must not contain any off-diagonal entries.";
}

} // namespace sofa::component::mass::testing
