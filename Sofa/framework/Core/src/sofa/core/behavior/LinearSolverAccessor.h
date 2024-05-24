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

#include <sofa/core/config.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/LinearSolver.h>

namespace sofa::core::behavior
{

/**
 * Base class for components requiring access to a linear solver
 */
class SOFA_CORE_API LinearSolverAccessor : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(LinearSolverAccessor, objectmodel::BaseObject)

    void init() override;

protected:

    explicit LinearSolverAccessor(LinearSolver* linearSolver = nullptr);

    SingleLink<LinearSolverAccessor, LinearSolver, BaseLink::FLAG_STRONGLINK> l_linearSolver;
};

}
