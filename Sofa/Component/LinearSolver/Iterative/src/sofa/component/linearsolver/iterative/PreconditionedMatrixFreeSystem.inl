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
#include <sofa/component/linearsolver/iterative/PreconditionedMatrixFreeSystem.h>

namespace sofa::component::linearsolver::iterative
{

template <class TMatrix, class TVector>
PreconditionedMatrixFreeSystem<TMatrix, TVector>::PreconditionedMatrixFreeSystem()
    : l_preconditionerSystem(initLink("preconditionerSystem", "Link toward the linear system of the preconditioner"))
    , d_assemblingRate(initData(&d_assemblingRate, 1u, "assemblingRate",
        "Rate of update of the preconditioner matrix, in number of time steps or Newton iterations"))
{
    this->addAlias(&d_assemblingRate, "update_step");
}

template <class TMatrix, class TVector>
void PreconditionedMatrixFreeSystem<TMatrix, TVector>::init()
{
    linearsystem::MatrixFreeSystem<TMatrix, TVector>::init();

    reinitAssemblyCounter();
}

template <class TMatrix, class TVector>
void PreconditionedMatrixFreeSystem<TMatrix, TVector>::reinitAssemblyCounter()
{
    m_assemblyCounter = d_assemblingRate.getValue();  // to assemble the first time
}

template <class TMatrix, class TVector>
void PreconditionedMatrixFreeSystem<TMatrix, TVector>::reset()
{
    reinitAssemblyCounter();
}

template <class TMatrix, class TVector>
void PreconditionedMatrixFreeSystem<TMatrix, TVector>::buildSystemMatrix(
    const core::MechanicalParams* mparams)
{
    //this component builds its own matrix...
    linearsystem::MatrixFreeSystem<TMatrix, TVector>::buildSystemMatrix(mparams);

    //... and also the one from the preconditioner
    if (l_preconditionerSystem)
    {
        if (++m_assemblyCounter >= d_assemblingRate.getValue())
        {
            l_preconditionerSystem->buildSystemMatrix(mparams);
            m_assemblyCounter = 0;
        }
    }
}

template <class TMatrix, class TVector>
void PreconditionedMatrixFreeSystem<TMatrix, TVector>::resizeSystem(sofa::Size n)
{
    linearsystem::MatrixFreeSystem<TMatrix, TVector>::resizeSystem(n);

    if (l_preconditionerSystem)
    {
        l_preconditionerSystem->resizeSystem(n);
        m_assemblyCounter = 0;
    }
}

template <class TMatrix, class TVector>
void PreconditionedMatrixFreeSystem<TMatrix, TVector>::clearSystem()
{
    linearsystem::MatrixFreeSystem<TMatrix, TVector>::clearSystem();

    if (l_preconditionerSystem)
    {
        l_preconditionerSystem->clearSystem();
        m_assemblyCounter = 0;
    }
}

}  // namespace sofa::component::linearsolver::iterative
