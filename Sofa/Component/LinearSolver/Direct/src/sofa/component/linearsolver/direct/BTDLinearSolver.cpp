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
#define SOFA_COMPONENT_LINEARSOLVER_BTDLINEARSOLVER_CPP

#include <sofa/component/linearsolver/direct/BTDLinearSolver.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/linearalgebra/BTDMatrix.inl>
#include <sofa/linearalgebra/BlockFullMatrix.inl>
#include <sofa/linearalgebra/BlockVector.inl>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/linearalgebra/FullVector.h>

namespace sofa::component::linearsolver::direct
{
int BTDLinearSolverClass = core::RegisterObject("Linear system solver using Thomas Algorithm for Block Tridiagonal matrices")
    .add< BTDLinearSolver<linearalgebra::BTDMatrix<6, SReal>, linearalgebra::BlockVector<6, SReal> > >(true)
;

template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API BTDLinearSolver< linearalgebra::BTDMatrix<6, SReal>, linearalgebra::BlockVector<6, SReal> >;

} //namespace sofa::component::linearsolver::direct
