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

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/helper/map.h>

#include <cmath>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::linearsolver::preconditioner
{

/// Linear system solver / preconditioner based on Successive Over Relaxation (SSOR).
///
/// If the matrix is decomposed as $A = D + L + L^T$, this solver computes
//       $(1/(2-w))(D/w+L)(D/w)^{-1}(D/w+L)^T x = b$
//  , or $(D+L)D^{-1}(D+L)^T x = b$ if $w=1$
template<class TMatrix, class TVector, class TThreadManager = NoThreadManager>
class SSORPreconditioner : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(SSORPreconditioner,TMatrix,TVector,TThreadManager),SOFA_TEMPLATE3(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector,TThreadManager));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Index Index;
    typedef TThreadManager ThreadManager;
    typedef SReal Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager> Inherit;

    SOFA_ATTRIBUTE_DISABLED__PRECONDITIONER_VERBOSEDATA()
    sofa::core::objectmodel::lifecycle::RemovedData f_verbose{this, "v23.12", "v24.06", "verbose", "This Data is no longer used"};

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_LINEARSOLVER_PRECONDITIONER()
    sofa::core::objectmodel::lifecycle::RenamedData<double> f_omega;

    Data<double> d_omega; ///< Omega coefficient
protected:
    SSORPreconditioner();
public:
    void solve (Matrix& M, Vector& x, Vector& b) override;
    void invert(Matrix& M) override;

    MatrixInvertData * createInvertData() override
    {
        return new SSORPreconditionerInvertData();
    }

    void parse(core::objectmodel::BaseObjectDescription *arg) override;

protected :

    class SSORPreconditionerInvertData : public MatrixInvertData
    {
    public :
        unsigned bsize;
        std::vector<double> inv_diag;
    };

};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_SSORPRECONDITIONER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API SSORPreconditioner< linearalgebra::CompressedRowSparseMatrix<SReal>, linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API SSORPreconditioner< linearalgebra::CompressedRowSparseMatrix< type::Mat<3, 3, SReal> >, linearalgebra::FullVector<SReal> >;
#endif // !defined(SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_SSORPRECONDITIONER_CPP)

} // namespace sofa::component::linearsolver::preconditioner
