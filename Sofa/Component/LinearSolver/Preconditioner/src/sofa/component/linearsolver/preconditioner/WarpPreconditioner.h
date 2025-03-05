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
#include <sofa/component/linearsolver/preconditioner/config.h>

#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/type/Mat.h>
#include <sofa/linearalgebra/FullVector.h>
#include <cmath>
#include <sofa/linearalgebra/RotationMatrix.h>
#include <sofa/core/behavior/BaseRotationFinder.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>

#include <map>

namespace sofa::component::linearsolver::preconditioner
{

/// Linear system solver wrapping another (precomputed) linear solver by a per-node rotation matrix
template<class TMatrix, class TVector, class ThreadManager = NoThreadManager>
class WarpPreconditioner : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,ThreadManager>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(WarpPreconditioner,TMatrix,TVector,ThreadManager),SOFA_TEMPLATE3(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector,ThreadManager));
    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename TMatrix::Real Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,ThreadManager> Inherit;
    typedef sofa::type::MatNoInit<3, 3, Real> Transformation;
    typedef TMatrix TRotationMatrix;
    typedef typename Inherit::JMatrixType JMatrixType;
    using Index = typename TMatrix::Index;

    SingleLink<WarpPreconditioner, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_linearSolver; ///< Link towards the linear solver used to build the warp conditioner

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_LINEARSOLVER_PRECONDITIONER()
    Data<unsigned> f_useRotationFinder;

    Data<unsigned> d_useRotationFinder; ///< Which rotation Finder to use
    Data<unsigned> d_updateStep; ///< Number of steps before the next refresh of the system matrix in the main solver

protected:
    WarpPreconditioner();

public:

    ~WarpPreconditioner();

    void init() override;
    void bwdInit() override;

    void setSystemMBKMatrix(const core::MechanicalParams* mparams) override;

    void invert(Matrix& M) override;

    void solve(Matrix& M, Vector& solution, Vector& rh) override;

    bool addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact) override;

    bool addMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact) override;

    Index getSystemDimention(const sofa::core::MechanicalParams* mparams);

    void computeResidual(const core::ExecParams* params, linearalgebra::BaseVector* /*f*/) override;

    void updateSystemMatrix() override;

protected:

    void checkLinearSystem() override;

private :

    int updateSystemSize,currentSystemSize;

    int indexwork;
    bool first;
    unsigned nextRefreshStep {};

    TRotationMatrix * rotationWork[2];
    std::vector<sofa::core::behavior::BaseRotationFinder *> rotationFinders;

    JMatrixType j_local;
};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_WARPPRECONDITIONER_CPP)
template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API WarpPreconditioner< linearalgebra::RotationMatrix<SReal>, linearalgebra::FullVector<SReal>, NoThreadManager >;
#endif // !defined(SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_WARPPRECONDITIONER_CPP)

} // namespace sofa::component::linearsolver::preconditioner
