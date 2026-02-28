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
#pragma once

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/DefaultMultiMatrixAccessor.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalResetForceVisitor.h>
#include <sofa/testing/BaseSimulationTest.h>
#include <sofa/testing/NumericTest.h>

namespace sofa::component::mass::testing
{

/**
 * @class Mass_test
 * @brief A test class to verify mass-related computations in a simulation environment.
 *
 * The class `Mass_test` is used to verify correctness of several key operations related to mass,
 * such as the mass matrix, computation of kinetic energy, and force/matrix interactions in
 * a simulation's mechanical state.
 *
 * @tparam _MassType Template parameter defining the mass type being tested. This type must define
 *        the associated `DataTypes` structure and methods for mass-related computations.
 */
template <typename _MassType>
requires std::is_base_of_v<core::behavior::BaseMass, _MassType>
struct Mass_test : public sofa::testing::BaseSimulationTest, public sofa::testing::NumericTest<typename _MassType::DataTypes::Real>
{
    using Mass = _MassType;
    using DataTypes = typename Mass::DataTypes;

    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;
    using Real = sofa::Real_t<DataTypes>;

    using DOF = sofa::component::statecontainer::MechanicalObject<DataTypes>;

    /// @name Scene elements
    /// {
    typename DOF::SPtr m_dof;
    typename Mass::SPtr m_mass;
    typename simulation::Node::SPtr m_node;
    /// }

    /// @name Precision and control parameters
    /// {
    SReal m_errorMax;       ///< tolerance in precision test. The actual value is this one times the epsilon of the Real numbers (typically float or double)
    std::pair<Real,Real> m_deltaRange;
    bool m_debug;           ///< Print debug messages. Default is false.
    /// }

    bool m_testAccFromF { true };
    bool m_testAddMToMatrix { true };
    bool m_testBuildMassMatrix { true };
    bool m_testKineticEnergy { true };

    Mass_test()
        : m_errorMax( 100 )
        , m_deltaRange( 1, 1000 )
        , m_debug( false )
    {
        simulation::Simulation* simu = sofa::simulation::getSimulation();
        assert(simu);

        m_node = simu->createNewGraph("root");
        m_dof = sofa::core::objectmodel::New<DOF>();
        m_node->addObject(m_dof);
        m_mass = sofa::core::objectmodel::New<Mass>();
        m_node->addObject(m_mass);
    }

    void setupState( const VecCoord& x, const VecDeriv& v)
    {
        std::size_t n = x.size();
        this->m_dof->resize(static_cast<sofa::Size>(n));
        typename DOF::WriteVecCoord xdof = this->m_dof->writePositions();
        sofa::testing::copyToData( xdof, x );
        typename DOF::WriteVecDeriv vdof = this->m_dof->writeVelocities();
        sofa::testing::copyToData( vdof, v );
    }

    void resetForce(core::MechanicalParams* mparams) const
    {
        simulation::mechanicalvisitor::MechanicalResetForceVisitor computeForce( mparams, core::vec_id::write_access::force );
        this->m_node->execute(computeForce);
    }

    void checkAddMToMatrix(const VecDeriv& v, const core::MechanicalParams& mparams, VecDeriv Mv,
                           sofa::SignedIndex matrixSize)
    {
        sofa::linearalgebra::EigenBaseSparseMatrix<SReal> testMatrix(matrixSize, matrixSize);

        core::behavior::DefaultMultiMatrixAccessor accessor;
        accessor.addMechanicalState(m_dof.get());
        accessor.setGlobalMatrix(&testMatrix);

        m_mass->addMToMatrix(&mparams, &accessor);
        testMatrix.compress();

        // Multiply by v
        sofa::type::vector<Real> eigenV, eigenMv;
        sofa::testing::data_traits<DataTypes>::VecDeriv_to_Vector(eigenV, v);
        eigenMv = testMatrix * eigenV;

        sofa::type::vector<Real> expectedMv;
        sofa::testing::data_traits<DataTypes>::VecDeriv_to_Vector(expectedMv, Mv);

        EXPECT_LT(this->vectorMaxDiff(eigenMv, expectedMv), m_errorMax * this->epsilon())
            << "addMToMatrix inconsistent with addMDx";
    }

    void checkBuildMassMatrix(const VecDeriv& v, const VecDeriv& Mv, sofa::SignedIndex matrixSize)
    {
        sofa::linearalgebra::EigenBaseSparseMatrix<SReal> testMatrix(matrixSize, matrixSize);

        struct MassMatrixAccumulatorTest : public core::behavior::MassMatrixAccumulator
        {
            MassMatrixAccumulatorTest(sofa::linearalgebra::EigenBaseSparseMatrix<SReal>& m)
                : matrix(m)
            {
            }
            void add(sofa::SignedIndex row, sofa::SignedIndex col, float value) override
            {
                matrix.add(row, col, (Real)value);
            }
            void add(sofa::SignedIndex row, sofa::SignedIndex col, double value) override
            {
                matrix.add(row, col, (Real)value);
            }
            sofa::linearalgebra::EigenBaseSparseMatrix<SReal>& matrix;
        } accumulator(testMatrix);

        m_mass->buildMassMatrix(&accumulator);
        testMatrix.compress();

        // Multiply by v
        sofa::type::vector<Real> eigenV, eigenMv;
        sofa::testing::data_traits<DataTypes>::VecDeriv_to_Vector(eigenV, v);
        eigenMv = testMatrix * eigenV;

        sofa::type::vector<Real> expectedMv;
        sofa::testing::data_traits<DataTypes>::VecDeriv_to_Vector(expectedMv, Mv);

        EXPECT_LT((SReal)this->vectorMaxDiff(eigenMv, expectedMv), (SReal)(m_errorMax * this->epsilon()))
            << "buildMassMatrix inconsistent with addMDx";
    }

    void checkKineticEnergy(const VecDeriv& v, std::size_t n, const core::MechanicalParams& mparams,
                            const VecDeriv& Mv)
    {
        SReal ke = m_mass->getKineticEnergy(&mparams);
        SReal vMv = 0;
        for (sofa::Size i = 0; i < static_cast<sofa::Size>(n); ++i)
        {
            vMv += static_cast<SReal>(sofa::type::dot(v[i], Mv[i]));
        }

        // v * Mv should be the 2 * kinetic energy
        EXPECT_LT(std::abs(vMv - 2 * ke), (SReal)(m_errorMax * this->epsilon()))
            << "Kinetic energy inconsistent with addMDx";
    }
    /**
     * @brief Given positions and velocities, checks mass methods.
     * @param x positions
     * @param v velocities
     */
    void run_test( const VecCoord& x, const VecDeriv& v, bool initScene = true )
    {
        if( m_deltaRange.second / m_errorMax <= sofa::testing::g_minDeltaErrorRatio )
            ADD_FAILURE() << "The comparison threshold is too large for the finite difference delta";

        ASSERT_EQ(x.size(), v.size());
        std::size_t n = x.size();

        // copy the position and velocities to the scene graph
        setupState(x, v);

        // init scene
        if (initScene)
        {
            sofa::simulation::node::initRoot(this->m_node.get());
        }

        core::MechanicalParams mparams;
        mparams.setMFactor(1.0);

        m_dof->vRealloc(&mparams, core::vec_id::write_access::dx); // dx is not allocated by default
        typename DOF::WriteVecDeriv dx = m_dof->writeDx();
        sofa::testing::copyToData ( dx, v );

        // Compute f = M * dx
        resetForce(&mparams);
        m_mass->addMDx(&mparams, core::vec_id::write_access::force, 1.0);

        VecDeriv Mv;
        sofa::testing::copyFromData(Mv, m_dof->readForces());

        // 1. Kinetic Energy
        if (m_testKineticEnergy)
        {
            checkKineticEnergy(v, n, mparams, Mv);
        }

        // 2. Test accFromF: a = M^-1 * f
        // a = M^-1 * (M * v) should be v
        if (m_testAccFromF)
        {
            m_mass->accFromF(&mparams, sofa::core::vec_id::write_access::force);

            VecDeriv a;
            sofa::testing::copyFromData(a, m_dof->readForces());

            EXPECT_LT((SReal)this->vectorMaxDiff(a, v), (SReal)(m_errorMax * this->epsilon())) << "accFromF inconsistent with addMDx (M^-1 * M * v != v)";
        }

        sofa::SignedIndex matrixSize = (sofa::SignedIndex)(n * DataTypes::deriv_total_size);

        // 3. Test buildMassMatrix
        if (m_testBuildMassMatrix)
        {
            checkBuildMassMatrix(v, Mv, matrixSize);
        }

        // 4. Test addMToMatrix
        if (m_testAddMToMatrix)
        {
            checkAddMToMatrix(v, mparams, Mv, matrixSize);
        }
    }
};

} // namespace sofa::component::mass::testing
