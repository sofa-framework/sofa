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
#ifndef SOFA_CORE_BEHAVIORMODEL_H
#define SOFA_CORE_BEHAVIORMODEL_H

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace core
{

/*! \class BehaviorModel
 *  \brief An interface inherited by all BehaviorModel
 *  \author Fonteneau Sylvere
 *  \version 0.1
 *  \date    02/22/2004
 *
 *  <P>This Interface is used for MechanicalGroup and "black-box" BehaviorModel<BR>
 *  All behavior model inherit of this Interface, and each object has to implement the updatePosition method
 *  <BR>updatePosition corresponds to the computation of a new simulation step<BR>
 */

class BehaviorModel : public virtual sofa::core::objectmodel::BaseObject
{
public:
    virtual ~BehaviorModel() {}

    virtual void init() = 0;

    /// Computation of a new simulation step.
    virtual void updatePosition(double dt) = 0;

    /// Deprecated transform method. Replaced by local coordinates system in Context.
    virtual void applyTranslation(double /*dx*/, double /*dy*/, double /*dz*/) { }
    /// Deprecated transform method. Replaced by local coordinates system in Context.
    virtual void applyRotation(double /*ax*/, double /*ay*/, double /*az*/, double /*angle*/) { }

    virtual void applyScale(double /*sx*/, double /*sy*/, double /*sz*/, double /*smass*/) { }
};

} // namespace core

} // namespace sofa

#endif
