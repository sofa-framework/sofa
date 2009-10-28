/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
 *  For example, we can have a mapping from a BehaviorModel to a VisualModel.
 *
 */
class BaseMapping : public virtual objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseMapping, objectmodel::BaseObject);
    /// Destructor
    virtual ~BaseMapping() { }


    /// Apply the transformation from the input model to the output model (like apply displacement from BehaviorModel to VisualModel)
    virtual void updateMapping() = 0;

    /// Accessor to the input model of this mapping
    virtual objectmodel::BaseObject* getFrom() = 0;

    /// Accessor to the output model of this mapping
    virtual objectmodel::BaseObject* getTo() = 0;

    /// Disable the mapping to get the original coordinates of the mapped model.
    virtual void disable()=0;


};

} // namespace core

} // namespace sofa

#endif
