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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_ODESOLVER_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_ODESOLVER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>
#include <sofa/core/componentmodel/behavior/MultiVector.h>
#include <sofa/core/componentmodel/behavior/MultiMatrix.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

/**
 *  \brief Component responsible for timestep integration, i.e. advancing the state from time t to t+dt.
 *
 *  This class currently control both the integration scheme (explicit,
 *  implicit, static, etc), and the linear system resolution algorithm
 *  (conjugate gradient, matrix direct inversion, etc). Those two aspect will
 *  propably be separated in a future version.
 *
 *  While all computations required to do the integration step are handled by
 *  this object, they should not be implemented directly in it, but instead
 *  the solver propagates orders (or Visitor) to the other components in the
 *  scenegraph that will locally execute them. This allow for greater
 *  flexibility (the solver can just ask for the forces to be computed without
 *  knowing what type of forces are present), as well as performances
 *  (some computations can be executed in parallel).
 *
 */
class OdeSolver : public virtual objectmodel::BaseObject
{
public:
    typedef BaseMechanicalState::VecId VecId;

    OdeSolver();

    virtual ~OdeSolver();

    /// Main computation method.
    ///
    /// Specify and execute all computation for timestep integration, i.e.
    /// advancing the state from time t to t+dt.
    virtual void solve (double dt) = 0;

    /// Propagate the given state (time, position and velocity) through all mappings
    ///
    /// @TODO Why is this necessary in the OdeSolver API ? (Jeremie A. 03/02/2008)
    virtual void propagatePositionAndVelocity(double t, BaseMechanicalState::VecId x, BaseMechanicalState::VecId v) = 0;

    /// Given a displacement as computed by the linear system inversion, how much will it affect the velocity
    ///
    /// This method is used to compute the compliance for contact corrections
    /// For Euler methods, it is typically 1/dt.
    virtual double getVelocityIntegrationFactor() const
    {
        return 1.0/getContext()->getDt();
    }

    /// Given a displacement as computed by the linear system inversion, how much will it affect the position
    ///
    /// This method is used to compute the compliance for contact corrections
    /// For Euler methods, it is typically 1/dt².
    virtual double getPositionIntegrationFactor() const
    {
        return 1.0/(getContext()->getDt()*getContext()->getDt());
    }
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
