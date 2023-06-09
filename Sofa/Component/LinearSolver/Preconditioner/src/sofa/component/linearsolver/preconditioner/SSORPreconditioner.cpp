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
#define SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_SSORPRECONDITIONER_CPP
#include <sofa/component/linearsolver/preconditioner/SSORPreconditioner.inl>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa::component::linearsolver::preconditioner
{

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;
using namespace sofa::linearalgebra;

template<>
void SSORPreconditioner<linearalgebra::CompressedRowSparseMatrix<SReal>, linearalgebra::FullVector<SReal> >::solve (Matrix& M, Vector& z, Vector& r)
{
    const SSORPreconditionerInvertData * data = (SSORPreconditionerInvertData *) this->getMatrixInvertData(&M);

    const Index n = M.rowSize();
    const Real w = (Real)f_omega.getValue();

    const Matrix::VecIndex& colsIndex = M.getColsIndex();
    const Matrix::VecBlock& colsValue = M.getColsValue();
    // Solve (D/w+U) * t = r;
    for (Index j=n-1; j>=0; j--)
    {
        double temp = 0.0;
        Matrix::Range rowRange = M.getRowRange(j);
        Index xi = rowRange.begin();
        while (xi < rowRange.end() && (Index)colsIndex[xi] <= j) ++xi;
        for (; xi < rowRange.end(); ++xi)
        {
            const Index i = colsIndex[xi];
            const double e = colsValue[xi];
            temp += z[i] * e;
        }
        z[j] = (r[j] - temp) * w * data->inv_diag[j];
    }

    // Solve (I + w D^-1 * L) * z = t
    for (Index j=0; j<n; j++)
    {
        double temp = 0.0;
        Matrix::Range rowRange = M.getRowRange(j);
        Index xi = rowRange.begin();
        for (; xi < rowRange.end() && (Index)colsIndex[xi] < j; ++xi)
        {
            const Index i = colsIndex[xi];
            const double e = colsValue[xi];
            temp += z[i] * e;
        }
        z[j] -= temp * w * data->inv_diag[j];
        // we can reuse z because all values that we read are updated
    }

    if (w != (Real)1.0)
        for (Index j=0; j<M.rowSize(); j++)
            z[j] *= 2-w;
}

template<>
void SSORPreconditioner< linearalgebra::CompressedRowSparseMatrix< type::Mat<3,3, SReal> >, linearalgebra::FullVector<SReal> >::solve(Matrix& M, Vector& z, Vector& r)
{
    const SSORPreconditionerInvertData * data = (SSORPreconditionerInvertData *) this->getMatrixInvertData(&M);

    static constexpr sofa::Size BlocSize = 3;

    const Index nb = M.rowBSize();
    const Real w = (Real)f_omega.getValue();

    const typename Matrix::VecIndex& colsIndex = M.getColsIndex();
    const typename Matrix::VecBlock& colsValue = M.getColsValue();
    // Solve (D+U) * t = r;
    for (Index jb=nb-1; jb>=0; jb--)
    {
        const Index j0 = jb*BlocSize;
        type::Vec<BlocSize, SReal> temp;
        typename Matrix::Range rowRange = M.getRowRange(jb);
        Index xi = rowRange.begin();
        while (xi < rowRange.end() && static_cast<Index>(colsIndex[xi]) < jb) ++xi;
        // bloc on the diagonal
        const typename Matrix::Block& bdiag = colsValue[xi];
        // upper triangle matrix
        for (++xi; xi < rowRange.end(); ++xi)
        {
            const Index i0 = colsIndex[xi]*BlocSize;
            const typename Matrix::Block& b = colsValue[xi];
            for (Index j1=0; j1<static_cast<Index>(BlocSize); ++j1)
            {
                for (Index i1=0; i1<static_cast<Index>(BlocSize); ++i1)
                {
                    const Index i = i0+i1;
                    temp[j1] += z[i] * b[j1][i1];
                }
            }
        }
        // then the diagonal
        {
            const typename Matrix::Block& b = bdiag;
            for (Index j1=BlocSize-1; j1>=0; j1--)
            {
                const Index j = j0+j1;
                for (Index i1=j1+1; i1<static_cast<Index>(BlocSize); ++i1)
                {
                    const Index i = j0+i1;
                    temp[j1]+= z[i] * b[j1][i1];
                }
                z[j] = (r[j] - temp[j1]) * w * data->inv_diag[j];
            }
        }
    }

    // Solve (I + D^-1 * L) * z = t
    for (Index jb=0; jb<nb; jb++)
    {
        const Index j0 = jb*BlocSize;
        type::Vec<BlocSize, SReal> temp;
        typename Matrix::Range rowRange = M.getRowRange(jb);
        Index xi = rowRange.begin();
        // lower triangle matrix
        for (; xi < rowRange.end() && static_cast<Index>(colsIndex[xi]) < jb; ++xi)
        {
            const Index i0 = colsIndex[xi]*BlocSize;
            const typename Matrix::Block& b = colsValue[xi];
            for (Index j1=0; j1<static_cast<Index>(BlocSize); ++j1)
            {
                for (Index i1=0; i1<static_cast<Index>(BlocSize); ++i1)
                {
                    const Index i = i0+i1;
                    temp[j1] += z[i] * b[j1][i1];
                }
            }
        }
        // then the diagonal
        {
            const typename Matrix::Block& b = colsValue[xi];
            for (Index j1=0; j1<static_cast<Index>(BlocSize); ++j1)
            {
                const Index j = j0+j1;
                for (Index i1=0; i1<j1; ++i1)
                {
                    const Index i = j0+i1;
                    temp[j1] += z[i] * b[j1][i1];
                }
                // we can reuse z because all values that we read are updated
                z[j] -= temp[j1] * w * data->inv_diag[j];
            }
        }
    }
}

int SSORPreconditionerClass = core::RegisterObject("Linear system solver / preconditioner based on Symmetric Successive Over-Relaxation (SSOR). If the matrix is decomposed as $A = D + L + L^T$, this solver computes $(1/(2-w))(D/w+L)(D/w)^{-1}(D/w+L)^T x = b, or $(D+L)D^{-1}(D+L)^T x = b$ if $w=1$.")
        .add< SSORPreconditioner< CompressedRowSparseMatrix<SReal>, FullVector<SReal> > >(true)
        .add< SSORPreconditioner< CompressedRowSparseMatrix< type::Mat<3,3,SReal> >, FullVector<SReal> > >()
        .addAlias("SSORLinearSolver")
        .addAlias("SSORSolver")
        ;

template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API SSORPreconditioner< linearalgebra::CompressedRowSparseMatrix<SReal>, linearalgebra::FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API SSORPreconditioner< linearalgebra::CompressedRowSparseMatrix< type::Mat<3, 3, SReal> >, linearalgebra::FullVector<SReal> >;

} // namespace sofa::component::linearsolver::preconditioner
