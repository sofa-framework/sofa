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
#include <sofa/core/objectmodel/BaseComponent.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/map.h>
#include <sofa/type/vector.h>

#include <map>
#include <string>

namespace sofa::core::behavior
{

/**
 * Base class for components solving a problem through an iterative process.
 */
class SOFA_CORE_API IterativeSolver : public virtual objectmodel::BaseComponent
{
public:
    SOFA_ABSTRACT_CLASS(IterativeSolver, objectmodel::BaseComponent);

    /// Maximum number of iterations
    Data<unsigned> d_maxIter;

    /// Residual tolerance for accepting convergence
    Data<SReal> d_tolerance;

    /// Residual series recorded during the last solve, keyed by series name
    Data<std::map<std::string, sofa::type::vector<SReal>>> d_graph;

protected:
    IterativeSolver();
    ~IterativeSolver() override = default;
};

} // namespace sofa::core::behavior
