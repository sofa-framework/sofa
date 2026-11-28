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
#include <sofa/component/mass/FEMCapacity.h>
#include <sofa/component/mass/FEMMass.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>
#include <sofa/core/MechanicalParams.h>
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

/// Parameters of the tests: the component computing the matrix, and the component providing the
/// nodal coefficient (mass density or capacity)
template<class TComponent, template<class> class TCoefficient>
struct CapacityParam
{
    using Component = TComponent;
    using DataTypes = typename Component::DataTypes;
    using SpatialTypes = typename Component::SpatialTypes;
    using Coefficient = TCoefficient<sofa::Real_t<DataTypes>>;
};

template<class DataTypes, class ElementType>
using MassParam = CapacityParam<FEMMass<DataTypes, ElementType>, NodalMassDensity>;

template<class DataTypes, class SpatialTypes, class ElementType>
using CapacityOfFieldParam = CapacityParam<FEMCapacity<DataTypes, SpatialTypes, ElementType>, NodalCapacity>;

template <typename Param>
struct FEMCapacityLumpingTest : public sofa::testing::BaseSimulationTest
{
    using Component = typename Param::Component;
    using DataTypes = typename Param::DataTypes;
    using SpatialTypes = typename Param::SpatialTypes;
    using Coefficient = typename Param::Coefficient;
    using DOF = sofa::component::statecontainer::MechanicalObject<DataTypes>;
    using SpatialDOF = sofa::component::statecontainer::MechanicalObject<SpatialTypes>;

    typename simulation::Node::SPtr m_root;
    typename simulation::Node::SPtr m_node; ///< node containing the field state and the component
    typename DOF::SPtr m_dof;
    typename SpatialDOF::SPtr m_spatialDof;
    typename Component::SPtr m_component;
    typename MeshTopology::SPtr m_topology;

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

        m_node->addObject(sofa::core::objectmodel::New<Coefficient>());

        m_component = sofa::core::objectmodel::New<Component>();
        m_node->addObject(m_component);

        m_dof->resize(8);

        if constexpr (std::is_same_v<DataTypes, SpatialTypes>)
        {
            setRandomPositions(*m_dof);
        }
        else
        {
            m_component->l_spatialState.set(m_spatialDof.get());
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

using CapacityTypes = ::testing::Types<
    MassParam<defaulttype::Vec1Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec2Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Edge>,
    MassParam<defaulttype::Vec2Types, sofa::geometry::Triangle>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Triangle>,
    MassParam<defaulttype::Vec2Types, sofa::geometry::Quad>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Quad>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Tetrahedron>,
    MassParam<defaulttype::Vec3Types, sofa::geometry::Hexahedron>,
    // MassParam<defaulttype::Vec3Types, sofa::geometry::Prism>,
    // MassParam<defaulttype::Vec3Types, sofa::geometry::Pyramid>

    // field type different from the spatial type
    CapacityOfFieldParam<defaulttype::Vec1Types, defaulttype::Vec3Types, sofa::geometry::Edge>,
    CapacityOfFieldParam<defaulttype::Vec1Types, defaulttype::Vec2Types, sofa::geometry::Triangle>,
    CapacityOfFieldParam<defaulttype::Vec1Types, defaulttype::Vec3Types, sofa::geometry::Triangle>,
    CapacityOfFieldParam<defaulttype::Vec1Types, defaulttype::Vec3Types, sofa::geometry::Tetrahedron>,
    CapacityOfFieldParam<defaulttype::Vec1Types, defaulttype::Vec3Types, sofa::geometry::Hexahedron>,
    CapacityOfFieldParam<defaulttype::Vec2Types, defaulttype::Vec3Types, sofa::geometry::Tetrahedron>,
    // field type identical to the spatial type
    CapacityOfFieldParam<defaulttype::Vec3Types, defaulttype::Vec3Types, sofa::geometry::Tetrahedron>
>;

TYPED_TEST_SUITE(FEMCapacityLumpingTest, CapacityTypes);

TYPED_TEST(FEMCapacityLumpingTest, ConsistentMatrixIsNotDiagonal)
{
    this->m_component->d_lumping.setValue(false);
    sofa::simulation::node::initRoot(this->m_root.get());

    EXPECT_FALSE(this->m_component->isComponentStateInvalid());
    EXPECT_FALSE(this->m_component->isDiagonal());

    MatrixDiagonalCheckerAccumulator accumulator;
    this->m_component->buildMassMatrix(&accumulator);

    EXPECT_TRUE(accumulator.hasDiagonalNonZero) << "Consistent matrix should contain diagonal entries.";
    EXPECT_TRUE(accumulator.hasOffDiagonalNonZero) << "Consistent matrix must have non-zero off-diagonal entries.";
}

TYPED_TEST(FEMCapacityLumpingTest, LumpedMatrixIsDiagonal)
{
    this->m_component->d_lumping.setValue(true);
    sofa::simulation::node::initRoot(this->m_root.get());

    EXPECT_FALSE(this->m_component->isComponentStateInvalid());
    EXPECT_TRUE(this->m_component->isDiagonal());

    MatrixDiagonalCheckerAccumulator accumulator;
    this->m_component->buildMassMatrix(&accumulator);

    EXPECT_TRUE(accumulator.hasDiagonalNonZero) << "Lumped matrix must contain diagonal entries.";
    EXPECT_FALSE(accumulator.hasOffDiagonalNonZero) << "Lumped matrix must not contain any off-diagonal entries.";
}

namespace
{
/// A scene with a single tetrahedron (unit corner tetrahedron in 3D), with a gravity along -y
template<class Param>
struct TetrahedronScene
{
    using Component = typename Param::Component;
    using DataTypes = typename Param::DataTypes;
    using SpatialTypes = typename Param::SpatialTypes;
    static_assert(std::is_same_v<SpatialTypes, defaulttype::Vec3Types>);

    sofa::simulation::Node::SPtr root;
    typename Component::SPtr component;

    TetrahedronScene()
    {
        root = sofa::simulation::getSimulation()->createNewGraph("root");
        root->setGravity(sofa::type::Vec3(0, -10, 0));

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

        fieldNode->addObject(sofa::core::objectmodel::New<typename Param::Coefficient>());

        component = sofa::core::objectmodel::New<Component>();
        fieldNode->addObject(component);
        component->l_spatialState.set(spatialState.get());

        sofa::simulation::node::initRoot(root.get());
        EXPECT_FALSE(component->isComponentStateInvalid());
    }

    MatrixSumAccumulator computeMatrixSum() const
    {
        MatrixSumAccumulator accumulator;
        component->buildMassMatrix(&accumulator);
        return accumulator;
    }

    /// sum of all the components of the force vector computed by addForce
    SReal computeForceSum() const
    {
        sofa::Data<sofa::VecDeriv_t<DataTypes>> f;
        f.setValue(sofa::VecDeriv_t<DataTypes>(4));
        sofa::Data<sofa::VecCoord_t<DataTypes>> x;
        x.setValue(sofa::VecCoord_t<DataTypes>(4));
        sofa::Data<sofa::VecDeriv_t<DataTypes>> v;
        v.setValue(sofa::VecDeriv_t<DataTypes>(4));

        component->addForce(sofa::core::MechanicalParams::defaultInstance(), f, x, v);

        SReal sum = 0;
        for (const auto& fi : f.getValue())
        {
            for (sofa::Size d = 0; d < DataTypes::deriv_total_size; ++d)
            {
                sum += fi[d];
            }
        }
        return sum;
    }
};
} // namespace

/**
 * The sum of all the entries of the matrix must be equal to the integral of the coefficient over
 * the tetrahedron (coefficient * volume = 1/6), repeated on each of the field components.
 */
TEST(FEMCapacity, scalarFieldOnTetrahedronIn3D)
{
    const TetrahedronScene<CapacityOfFieldParam<defaulttype::Vec1Types, defaulttype::Vec3Types, sofa::geometry::Tetrahedron>> scene;
    const auto sum = scene.computeMatrixSum();
    EXPECT_NEAR(sum.sum, 1. / 6., 1e-10);
    EXPECT_EQ(sum.maxIndex, 3);
}

TEST(FEMMass, tetrahedronIn3D)
{
    const TetrahedronScene<MassParam<defaulttype::Vec3Types, sofa::geometry::Tetrahedron>> scene;
    const auto sum = scene.computeMatrixSum();
    EXPECT_NEAR(sum.sum, 3. / 6., 1e-10);
    EXPECT_EQ(sum.maxIndex, 11);
}

/// FEMMass applies the gravity: the total force is the total mass times the gravity
TEST(FEMMass, gravityIsApplied)
{
    const TetrahedronScene<MassParam<defaulttype::Vec3Types, sofa::geometry::Tetrahedron>> scene;
    EXPECT_NEAR(scene.computeForceSum(), -10. / 6., 1e-10);
}

/// FEMCapacity does not apply any body force, even if the field has the dimension of the space
TEST(FEMCapacity, gravityIsNotApplied)
{
    const TetrahedronScene<CapacityOfFieldParam<defaulttype::Vec1Types, defaulttype::Vec3Types, sofa::geometry::Tetrahedron>> scalar;
    EXPECT_EQ(scalar.computeForceSum(), 0);

    const TetrahedronScene<CapacityOfFieldParam<defaulttype::Vec3Types, defaulttype::Vec3Types, sofa::geometry::Tetrahedron>> vector;
    EXPECT_EQ(vector.computeForceSum(), 0);
}

} // namespace sofa::component::mass::testing
