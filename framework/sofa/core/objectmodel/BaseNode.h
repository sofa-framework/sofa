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
#ifndef SOFA_CORE_OBJECTMODEL_BASENODE_H
#define SOFA_CORE_OBJECTMODEL_BASENODE_H

#include "BaseContext.h"

namespace sofa
{

namespace core

{
namespace objectmodel
{

class BaseObject;

/**
 *  \brief Base class for simulation nodes.
 *
 *  A Node is a class defining the main scene data structure of a simulation.
 *  It defined hierarchical relations between elements.
 *  Each node can have parent and child nodes (potentially defining a tree),
 *  as well as attached objects (the leaves of the tree).
 *
 * \author Jeremie Allard
 */
class BaseNode : public virtual Base
{
public:
    virtual ~BaseNode() {}

    /// @name Scene hierarchy
    /// @{

    /// Get parent node (or NULL if no hierarchy or for root node)
    virtual BaseNode* getParent() = 0;

    /// Get parent node (or NULL if no hierarchy or for root node)
    virtual const BaseNode* getParent() const = 0;

    /// Add a child node
    virtual void addChild(BaseNode* node) = 0;

    /// Remove a child node
    virtual void removeChild(BaseNode* node) = 0;

    /// Add a generic object
    virtual bool addObject(BaseObject* obj) = 0;

    /// Remove a generic object
    virtual bool removeObject(BaseObject* obj) = 0;

    /// Get this node context
    virtual BaseContext* getContext() = 0;

    /// Get this node context
    virtual const BaseContext* getContext() const = 0;

    /// @}
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
