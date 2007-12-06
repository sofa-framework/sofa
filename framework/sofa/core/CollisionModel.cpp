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
#include "CollisionModel.h"

namespace sofa
{

namespace core
{


/// Get a color that can be used to display this CollisionModel
const float* CollisionModel::getColor4f() const
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
            if (isActive()) return defaultColorSimulatedMovingActive;
            else            return defaultColorSimulatedMoving;
        else if (isActive()) return defaultColorSimulatedActive;
        else            return defaultColorSimulated;
    else if (isMoving())
        if (isActive()) return defaultColorMovingActive;
        else            return defaultColorMoving;
    else if (isActive()) return defaultColorActive;
    else            return defaultColor;
}

} // namespace core

} // namespace sofa

