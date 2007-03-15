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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEFORCEFIELD_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEFORCEFIELD_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

class BaseForceField : public virtual objectmodel::BaseObject
{
public:
    virtual ~BaseForceField() {}

    virtual void addForce() = 0;

    virtual void addDForce() = 0;

    virtual double getPotentialEnergy() =0;


    virtual void computeMatrix(sofa::defaulttype::SofaBaseMatrix *, double , double , double, unsigned int &) {};

    virtual void contributeToMatrixDimension(unsigned int * const, unsigned int * const) {};

    virtual void computeVector(sofa::defaulttype::SofaBaseVector *, unsigned int &) {};

    virtual void matResUpdatePosition(sofa::defaulttype::SofaBaseVector *, unsigned int &) {};
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
