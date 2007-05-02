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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASECONSTRAINT_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASECONSTRAINT_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>

#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

class BaseConstraint : public virtual objectmodel::BaseObject
{
public:
    virtual ~BaseConstraint() { }

    virtual void projectResponse() = 0; ///< project dx to constrained space (dx models an acceleration)
    virtual void projectVelocity() = 0; ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition() = 0; ///< project x to constrained space (x models a position)

    virtual void projectResponse(double **) {}; ///< project the compliance Matrix to constrained space

    virtual void applyConstraint(unsigned int&, double&) {};

    virtual void applyConstraint(defaulttype::SofaBaseMatrix *, unsigned int &) {};
    virtual void applyConstraint(defaulttype::SofaBaseVector *, unsigned int &) {};

    virtual BaseMechanicalState* getDOFs() { return NULL; }

    virtual void getConstraintValue(double * /*, unsigned int &*/) {};
    // virtual void resetContactCpt(){};
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
