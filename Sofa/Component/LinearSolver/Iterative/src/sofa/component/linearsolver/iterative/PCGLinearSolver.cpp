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
#define SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_PCGLINEARSOLVER_CPP
#include <sofa/component/linearsolver/iterative/PCGLinearSolver.inl>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/linearsystem/MatrixFreeSystem.h>

namespace sofa::component::linearsolver::iterative
{

void registerPCGLinearSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Linear solver using the preconditioned conjugate gradient iterative algorithm.")
        .add< PCGLinearSolver<GraphScatteredMatrix, GraphScatteredVector> >());
}

template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PCGLinearSolver<GraphScatteredMatrix, GraphScatteredVector>;

} // namespace sofa::component::linearsolver::iterative
