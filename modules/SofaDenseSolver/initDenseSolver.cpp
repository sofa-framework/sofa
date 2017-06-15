/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/system/config.h>
#include <SofaDenseSolver/initDenseSolver.h>



#include <SofaBaseLinearSolver/MatrixLinearSolver.inl>
#include "NewMatMatrix.h"
#include "NewMatVector.h"


namespace sofa
{

namespace component
{

namespace linearsolver
{

// template specialization on specific matrix types

template class SOFA_DENSE_SOLVER_API MatrixLinearSolver< NewMatMatrix, NewMatVector >;
template class SOFA_DENSE_SOLVER_API MatrixLinearSolver< NewMatSymmetricMatrix, NewMatVector >;
template class SOFA_DENSE_SOLVER_API MatrixLinearSolver< NewMatBandMatrix, NewMatVector >;
template class SOFA_DENSE_SOLVER_API MatrixLinearSolver< NewMatSymmetricBandMatrix, NewMatVector >;
}


void initDenseSolver()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

SOFA_LINK_CLASS(LULinearSolver)





} // namespace component

} // namespace sofa
