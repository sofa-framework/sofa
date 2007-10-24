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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEMASS_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEMASS_H

#include <sofa/core/objectmodel/BaseObject.h>
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
 *  \brief Component responsible for mass-related computations (gravity, acceleration).
 *
 *  Mass can be defined either as a scalar, vector, or a full mass-matrix.
 *  It is responsible for converting forces to accelerations (for explicit integrators),
 *  or displacements to forces (for implicit integrators).
 *
 *  It is often also a ForceField, computing gravity-related forces.
 */
class BaseMass : public virtual objectmodel::BaseObject
{
public:
    virtual ~BaseMass() { }

    /// @name Vector operations
    /// @{

    /// f += factor M dx
    virtual void addMDx(double factor = 1.0) = 0;

    /// dx = M^-1 f
    virtual void accFromF() = 0;

    /// vMv/2
    virtual double getKineticEnergy() = 0;

    /// Add Mass contribution to global Matrix assembling
    virtual void addMToMatrix(defaulttype::BaseMatrix * /*mat*/, double /*mFact*/, unsigned int &/*offset*/) = 0;

    /// Add Mass contribution to global Vector assembling
    virtual void addMDxToVector(defaulttype::BaseVector * /*resVect*/, double /*mFact*/, unsigned int& /*offset*/, bool /*dxNull*/) = 0;

    /// initialization to export kinetic and potential energy to gnuplot files format
    virtual void initGnuplot()=0;

    /// export kinetic and potential energy state at "time" to a gnuplot file
    virtual void exportGnuplot(double time)=0;
    /// @}

};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
