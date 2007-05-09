/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_BASEMAPPING_H
#define SOFA_CORE_BASEMAPPING_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/BehaviorModel.h>

namespace sofa
{

namespace core
{

/**
 *  \brief An interface to convert a model to an other model
 *
 *  This Interface is used for the Mappings. A Mapping can convert one model to an other.
 *  For example, we can have a mapping from BehaviorModel to a VisualModel.
 *
 */
class BaseMapping : public virtual objectmodel::BaseObject
{
public:
    virtual ~BaseMapping() { }

    virtual void init() = 0;

    /// Apply the transformation from a model to an other model (like apply displacement from BehaviorModel to VisualModel)
    virtual void updateMapping() = 0;

    /// Accessor to the input model of this mapping
    virtual objectmodel::BaseObject* getFrom() = 0;

    /// Accessor to the output model of this mapping
    virtual objectmodel::BaseObject* getTo() = 0;
};

} // namespace core

} // namespace sofa

#endif
