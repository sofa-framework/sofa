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
#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_DETECTIONOUTPUT_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_DETECTIONOUTPUT_H

#include <sofa/core/CollisionElement.h>
#include <sofa/defaulttype/Vec.h>
#include <iostream>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

using namespace sofa::defaulttype;

class DetectionOutput
{
public:
    std::pair<core::CollisionElementIterator, core::CollisionElementIterator> elem; ///< Pair of colliding elements
    Vector3 point[2]; ///< Point in contact on each element
    Vector3 freePoint[2]; ///< free Point in contact on each element
    Vector3 normal; ///< Normal of the contact, pointing outward from model 1
    //bool collision; ///< Are the elements interpenetrating
    double distance; ///< Distance between the elements (negative for interpenetration)
    double deltaT; ///< Time of contact (0 for non-continuous methods)
    DetectionOutput()
        : elem(NULL, NULL), distance(0.0), deltaT(0.0)
    {
    }
};

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
