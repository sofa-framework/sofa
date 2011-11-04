/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "CollisionModel.h"

namespace sofa
{

namespace core
{

std::vector<int> BaseCollisionElementIterator::emptyVector; ///< empty vector to be able to initialize the iterator to an empty pair


/// Get a color that can be used to display this CollisionModel
const float* CollisionModel::getColor4f()
{

    static float defaultColorSimulatedMovingActive[4] = {1, 0.5f, 0, 1};

    static float defaultColorSimulatedMoving[4] = {0.5f, 0.25f, 0, 1};

    static float defaultColorSimulatedActive[4] = {1, 0, 0, 1};

    static float defaultColorSimulated[4] = {0.5f, 0, 0, 1};

    static float defaultColorMovingActive[4] = {0, 1, 0.5f, 1};

    static float defaultColorMoving[4] = {0, 0.5f, 0.25f, 1};

    static float defaultColorActive[4] = {0.5f, 0.5f, 0.5f, 1};

    static float defaultColor[4] = {0.25f, 0.25f, 0.25f, 1};

    if (color.isSet())
        return color.getValue().ptr();
    else if (isSimulated())
        if (isMoving())
            if (isActive()) {setColor4f(defaultColorSimulatedMovingActive); return defaultColorSimulatedMovingActive;}
            else            {setColor4f(defaultColorSimulatedMoving); return defaultColorSimulatedMoving;}
        else if (isActive()) {setColor4f(defaultColorSimulatedActive); return defaultColorSimulatedActive;}
        else            {setColor4f(defaultColorSimulated); return defaultColorSimulated;}
    else if (isMoving())
        if (isActive()) {setColor4f(defaultColorMovingActive); return defaultColorMovingActive;}
        else            {setColor4f(defaultColorMoving); return defaultColorMoving;}
    else if (isActive()) {setColor4f(defaultColorActive); return defaultColorActive;}
    else            {setColor4f(defaultColor); return defaultColor;}
}

} // namespace core

} // namespace sofa

