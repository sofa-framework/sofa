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
#include <sofa/component/linearsolver/preconditioner/SSORPreconditioner.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <iostream>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <cmath>
#include <sofa/helper/system/thread/CTime.h>

namespace sofa::component::linearsolver::preconditioner
{

template<class TMatrix, class TVector, class TThreadManager>
SSORPreconditioner<TMatrix,TVector,TThreadManager>::SSORPreconditioner()
    : f_omega( initData(&f_omega,1.0, "omega","Omega coefficient") )
{
}

// solve (D+U) * D^-1 * ( D + U)
template<class TMatrix, class TVector, class TThreadManager>
void SSORPreconditioner<TMatrix,TVector,TThreadManager>::solve (Matrix& M, Vector& z, Vector& r)
{
    SSORPreconditionerInvertData * data = (SSORPreconditionerInvertData *) this->getMatrixInvertData(&M);

    const Index n = M.rowSize();
    const Real w = (Real)f_omega.getValue();
    //Solve (D/w+u) * u3 = r;
    for (Index j=n-1; j>=0; j--)
    {
        double temp = 0.0;
        for (Index i=j+1; i<n; i++)
        {
            temp += z[i] * M.element(i,j);
        }
        z[j] = (r[j] - temp) * w * data->inv_diag[j];
    }

    //Solve (I + w D^-1 * L) * z = u3
    for (Index j=0; j<n; j++)
    {
        double temp = 0.0;
        for (Index i=0; i<j; i++)
        {
            temp += z[i] * M.element(i,j);
        }
        z[j] = z[j] - temp * w * data->inv_diag[j];
        // we can reuse z because all values that we read are updated
    }

    if (w != (Real)1.0)
        for (Index j=0; j<M.rowSize(); j++)
            z[j] *= 2-w;

}

template<class TMatrix, class TVector, class TThreadManager>
void SSORPreconditioner<TMatrix,TVector,TThreadManager>::invert(Matrix& M)
{
    SSORPreconditionerInvertData * data = (SSORPreconditionerInvertData *) this->getMatrixInvertData(&M);

    Index n = M.rowSize();
    data->inv_diag.resize(n);
    for (Index j=0; j<n; j++) data->inv_diag[j] = 1.0 / M.element(j,j);
}

template <class TMatrix, class TVector, class TThreadManager>
void SSORPreconditioner<TMatrix, TVector, TThreadManager>::parse(
    core::objectmodel::BaseObjectDescription* arg)
{
    if (arg->getAttribute("verbose"))
    {
        msg_warning() << "Attribute 'verbose' has no use in this component. "
                         "To disable this warning, remove the attribute from the scene.";
    }

    Inherit::parse(arg);
}

} // namespace sofa::component::linearsolver::preconditioner
