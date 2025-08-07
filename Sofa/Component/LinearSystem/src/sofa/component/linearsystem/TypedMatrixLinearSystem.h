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

#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/DiagonalMatrix.h>
#include <sofa/linearalgebra/RotationMatrix.h>
#include <sofa/linearalgebra/BlockDiagonalMatrix.h>

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
    linearalgebra::BaseVector* getSystemRHSBaseVector() const override;
    linearalgebra::BaseVector* getSystemSolutionBaseVector() const override;

    /// Set the size of the matrix to n x n, and the size of RHS and solution to n
    void resizeSystem(sofa::Size n) override;

    void clearSystem() override;

    /// Assemble the right-hand side of the linear system, from the values contained in the (Mechanical/Physical)State objects
    /// Warning: it assumes m_mappingGraph is already built
    void setRHS(core::MultiVecDerivId v) override;

    /// Set the initial estimate of the linear system solution vector, from the values contained in the (Mechanical/Physical)State objects
    /// This vector will be replaced by the solution of the system once the system is solved
    /// Warning: it assumes m_mappingGraph is already built
    void setSystemSolution(core::MultiVecDerivId v) override;

    void dispatchSystemSolution(core::MultiVecDerivId v) override;
    void dispatchSystemRHS(core::MultiVecDerivId v) override;

    core::objectmodel::BaseContext* getSolveContext();

    /**
     * This Data is used only to notify other components that the system matrix changed (resize,
     * clear)
     */
    Data<bool> d_matrixChanged;

protected:

    TypedMatrixLinearSystem();

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
    template <core::matrixaccumulator::Contribution c>
    static SReal getContributionFactor(
        const sofa::core::MechanicalParams* mparams,
        const sofa::core::matrixaccumulator::get_component_type<c>* object)
    {
        if constexpr (c == core::matrixaccumulator::Contribution::STIFFNESS)
        {
            return sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(
                mparams, object->rayleighStiffness.getValue());
        }
        else if constexpr (c == core::matrixaccumulator::Contribution::MASS)
        {
            return sofa::core::mechanicalparams::mFactorIncludingRayleighDamping(
                mparams, object->rayleighMass.getValue());
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

#if !defined(SOFA_COMPONENT_LINEARSYSTEM_TYPEDMATRIXLINEARSYSTEM_CPP)
using namespace sofa::linearalgebra;

extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< FullMatrix<double>, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< FullMatrix<float>, FullVector<float> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< SparseMatrix<double>, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< SparseMatrix<float>, FullVector<float> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<double>, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<float>, FullVector<float> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<2,2,double> >, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<2,2,float> >, FullVector<float> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<3,3,double> >, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<3,3,float> >, FullVector<float> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<4,4,double> >, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<4,4,float> >, FullVector<float> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<6,6,double> >, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<6,6,float> >, FullVector<float> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<8,8,double> >, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< CompressedRowSparseMatrix<type::Mat<8,8,float> >, FullVector<float> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< DiagonalMatrix<double>, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< DiagonalMatrix<float>, FullVector<float> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< BlockDiagonalMatrix<3,double>, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< BlockDiagonalMatrix<3,float>, FullVector<float> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< RotationMatrix<double>, FullVector<double> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API TypedMatrixLinearSystem< RotationMatrix<float>, FullVector<float> >;
#endif


} //namespace sofa::component::linearsystem
