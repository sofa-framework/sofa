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
#include <sofa/component/linearsolver/iterative/GraphScatteredTypes.h>

#include <sofa/simulation/MechanicalOperations.h>


namespace sofa::component::linearsolver
{

using sofa::core::behavior::LinearSolver;
using sofa::core::objectmodel::BaseContext;

void GraphScatteredMatrix::apply(GraphScatteredVector& res, GraphScatteredVector& x)
{
    // matrix-vector product through visitors
    parent->propagateDxAndResetDf(x,res);
    parent->addMBKdx(res,
        core::MatricesFactors::M(parent->mparams.mFactor()),
        core::MatricesFactors::B(parent->mparams.bFactor()),
        core::MatricesFactors::K(parent->mparams.kFactor()), false); // df = (m M + b B + k K) dx

    // filter the product to take the constraints into account
    //
    parent->projectResponse(res);     // q is projected to the constrained space
}

sofa::Size GraphScatteredMatrix::rowSize()
{
    sofa::Size nbRow=0, nbCol=0;
    this->parent->getMatrixDimension(&nbRow, &nbCol);
    return nbRow;

}

sofa::Size GraphScatteredMatrix::colSize()
{
    sofa::Size nbRow=0, nbCol=0;
    this->parent->getMatrixDimension(&nbRow, &nbCol);
    return nbCol;
}


void GraphScatteredVector::operator=(const MultExpr<GraphScatteredMatrix,GraphScatteredVector>& expr)
{
    expr.a.apply(*this,expr.b);
}

} // namespace sofa::component::linearsolver
