/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <SofaConstraint/ConstraintSolverImpl.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

ConstraintProblem::ConstraintProblem()
    : tolerance(0.00001), maxIterations(1000),
      dimension(0), problemId(0)
{
}

ConstraintProblem::~ConstraintProblem()
{
}

void ConstraintProblem::clear(int nbConstraints)
{
    dimension = nbConstraints;
    W.resize(nbConstraints, nbConstraints);
    dFree.resize(nbConstraints);
    f.resize(nbConstraints);

    static unsigned int counter = 0;
    problemId = ++counter;
}

unsigned int ConstraintProblem::getProblemId()
{
    return problemId;
}

} // namespace constraintset

} // namespace component

} // namespace sofa
