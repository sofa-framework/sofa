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
#pragma once
#include <sofa/component/linearsystem/config.h>

#include <sofa/core/behavior/BaseMatrixLinearSystem.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/component/linearsystem/MappingGraph.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsystem/LinearSystemData.h>
#include <sofa/core/MatrixAccumulator.h>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/core/behavior/BaseLocalMassMatrix.h>

namespace sofa::component::linearsystem
{

/**
 * Component storing a global matrix representing a linear system
 */
template<class TMatrix, class TVector>
class TypedMatrixLinearSystem : public sofa::core::behavior::BaseMatrixLinearSystem
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(TypedMatrixLinearSystem, TMatrix, TVector), sofa::core::behavior::BaseMatrixLinearSystem);

    template<class M, class V>
    friend class CompositeLinearSystem;

    using Matrix = TMatrix;
    using Vector = TVector;

    static std::string GetCustomTemplateName()
    {
        return TMatrix::Name();
    }

    /// Return the global matrix of the linear system
    virtual TMatrix* getSystemMatrix() const;

    /// Return the RHS of the linear system
    virtual TVector* getRHSVector() const;

    /// Return the solution of the linear system
    virtual TVector* getSolutionVector() const;

    linearalgebra::BaseMatrix* getSystemBaseMatrix() const override;

    /// Set the size of the matrix to n x n, and the size of RHS and solution to n
    virtual void resizeSystem(sofa::Size n);

    virtual void clearSystem();

    /// Assemble the right-hand side of the linear system, from the values contained in the (Mechanical/Physical)State objects
    /// Warning: it assumes m_mappingGraph is already built
    virtual void setRHS(core::MultiVecDerivId v);

    /// Set the initial estimate of the linear system solution vector, from the values contained in the (Mechanical/Physical)State objects
    /// This vector will be replaced by the solution of the system once the system is solved
    /// Warning: it assumes m_mappingGraph is already built
    virtual void setSystemSolution(core::MultiVecDerivId v);

    virtual void dispatchSystemSolution(core::MultiVecDerivId v);
    virtual void dispatchSystemRHS(core::MultiVecDerivId v);

    core::objectmodel::BaseContext* getSolveContext();

protected:

    LinearSystemData<TMatrix, TVector> m_linearSystem;

    /// Relationships between the mechanical states and their associated components
    MappingGraph m_mappingGraph;

    /// The list of force fields contributing to the matrix assembly
    sofa::type::vector<sofa::core::behavior::BaseForceField*> m_forceFields;
    /// The list of masses contributing to the matrix assembly
    sofa::type::vector<sofa::core::behavior::BaseMass*> m_masses;
    /// The list of mechanical mappings contributing to the matrix assembly
    sofa::type::vector<sofa::core::BaseMapping*> m_mechanicalMappings;
    /// The list of projective constraints contributing to the matrix assembly
    sofa::type::vector<sofa::core::behavior::BaseProjectiveConstraintSet*> m_projectiveConstraints;

    void preAssembleSystem(const core::MechanicalParams* /*mparams*/) override;

    virtual void associateLocalMatrixToComponents(const core::MechanicalParams* /*mparams*/) {}

    virtual void allocateSystem();

    /// Set the size of RHS and solution to n
    virtual void resizeVectors(sofa::Size n);

    void copyLocalVectorToGlobalVector(core::MultiVecDerivId v, TVector* globalVector);

    /**
     * Returns the factor to apply to the contributions depending on the contribution type
     */
    template<core::matrixaccumulator::Contribution c>
    static SReal getContributionFactor(
        const sofa::core::MechanicalParams* mparams,
        const sofa::core::matrixaccumulator::get_component_type<c>* object)
    {
        if constexpr (c == core::matrixaccumulator::Contribution::STIFFNESS)
        {
            return sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams,
                object->rayleighStiffness.getValue());
        }
        else if constexpr (c == core::matrixaccumulator::Contribution::MASS)
        {
            return sofa::core::mechanicalparams::mFactorIncludingRayleighDamping(mparams,
                object->rayleighMass.getValue());
        }
        else if constexpr (c == core::matrixaccumulator::Contribution::DAMPING)
        {
            return sofa::core::mechanicalparams::bFactor(mparams);
        }
        else if constexpr (c == core::matrixaccumulator::Contribution::GEOMETRIC_STIFFNESS)
        {
            return sofa::core::mechanicalparams::kFactor(mparams);
        }
        else
        {
            return 1_sreal;
        }
    }
};

} //namespace sofa::component::linearsystem
