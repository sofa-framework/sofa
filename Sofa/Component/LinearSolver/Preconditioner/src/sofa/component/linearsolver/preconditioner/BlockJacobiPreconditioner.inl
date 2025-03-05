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

#include <sofa/component/linearsolver/preconditioner/BlockJacobiPreconditioner.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <cmath>

namespace sofa::component::linearsolver::preconditioner
{

template<class TMatrix, class TVector>
BlockJacobiPreconditioner<TMatrix,TVector>::BlockJacobiPreconditioner()
{
    this->addAlias(&(this->f_printLog), "verbose");
}

template<class TMatrix, class TVector>
void BlockJacobiPreconditioner<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    M.mul(z,r);
}

template<class TMatrix, class TVector>
void BlockJacobiPreconditioner<TMatrix,TVector>::invert(Matrix& M)
{
    M.invert();
    msg_info() << M;
}

template <class TMatrix, class TVector>
void BlockJacobiPreconditioner<TMatrix, TVector>::parse(
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
