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

#include <list>

namespace sofa::core::behavior
{

/**
 *  \brief Interface describing partial linear solvers API
 *
 */
class PartialLinearSolver : public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(PartialLinearSolver, objectmodel::BaseObject);

    /// Init the partial solve
    virtual void init_partial_solve() = 0;
    
    /// Init the partial solve
    /// partial solve :
    /// b is accumulated
    /// db is a sparse vector that is added to b
    /// partial_x is a sparse vector (with sparse map given) that provide the result of M x = b+db
    /// Solve Mx=b
    virtual void partial_solve(std::list<sofa::SignedIndex>& /*I_last_Disp*/, std::list<sofa::SignedIndex>& /*I_last_Dforce*/, bool /*NewIn*/) = 0;
};

} // namespace sofa::core::behavior
