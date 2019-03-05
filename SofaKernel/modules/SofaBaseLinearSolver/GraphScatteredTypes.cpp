/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseLinearSolver/GraphScatteredTypes.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalMatrixVisitor.h>

#include <cstdlib>
#include <cmath>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using sofa::core::behavior::LinearSolver;
using sofa::core::objectmodel::BaseContext;

void GraphScatteredMatrix::apply(GraphScatteredVector& res, GraphScatteredVector& x)
{
    // matrix-vector product through visitors
    parent->propagateDxAndResetDf(x,res);
    parent->addMBKdx(res,parent->mparams.mFactor(),parent->mparams.bFactor(),parent->mparams.kFactor(), false); // df = (m M + b B + k K) dx

    // filter the product to take the constraints into account
    //
    parent->projectResponse(res);     // q is projected to the constrained space
}

unsigned int GraphScatteredMatrix::rowSize()
{
    unsigned int nbRow=0, nbCol=0;
    this->parent->getMatrixDimension(&nbRow, &nbCol);
    return nbRow;

}

unsigned int GraphScatteredMatrix::colSize()
{
    unsigned int nbRow=0, nbCol=0;
    this->parent->getMatrixDimension(&nbRow, &nbCol);
    return nbCol;
}


void GraphScatteredVector::operator=(const MultExpr<GraphScatteredMatrix,GraphScatteredVector>& expr)
{
    expr.a.apply(*this,expr.b);
}

} // namespace linearsolver

} // namespace component

} // namespace sofa
