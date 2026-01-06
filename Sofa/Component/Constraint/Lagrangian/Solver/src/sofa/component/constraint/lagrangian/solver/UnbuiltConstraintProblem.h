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
#include <sofa/component/constraint/lagrangian/solver/config.h>

#include <sofa/component/constraint/lagrangian/solver/GenericConstraintProblem.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/linearalgebra/SparseMatrix.h>

namespace sofa::component::constraint::lagrangian::solver
{


/**
 *  \brief This class adds components needed for unbuilt solvers to the GenericConstraintProblem
 *  This needs to be used by unbuilt solvers.
 */
class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API UnbuiltConstraintProblem : public GenericConstraintProblem
{
public:
    typedef std::vector< core::behavior::BaseConstraintCorrection* > ConstraintCorrections;

    UnbuiltConstraintProblem(GenericConstraintSolver* solver)
    : GenericConstraintProblem(solver)
    {}

    linearalgebra::SparseMatrix<SReal> Wdiag; /** UNBUILT **/
    std::list<unsigned int> constraints_sequence; /** UNBUILT **/
    std::vector< ConstraintCorrections > cclist_elems; /** UNBUILT **/


};
}
