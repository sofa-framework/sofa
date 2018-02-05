/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "CollisionModel.h"
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/helper/types/RGBAColor.h>

using sofa::helper::types::RGBAColor ;
namespace sofa
{

namespace core
{

std::vector<int> BaseCollisionElementIterator::emptyVector; ///< empty vector to be able to initialize the iterator to an empty pair

/// Get a color that can be used to display this CollisionModel
const float* CollisionModel::getColor4f()
{

    //TODO FIXME because of: https://github.com/sofa-framework/sofa/issues/64
    static const float defaultColorSimulatedMovingActive[4] = {1, 0.5f, 0, 1};

    static const float defaultColorSimulatedMoving[4] = {0.5f, 0.25f, 0, 1};

    static const float defaultColorSimulatedActive[4] = {1, 0, 0, 1};

    static const float defaultColorSimulated[4] = {0.5f, 0, 0, 1};

    static const float defaultColorMovingActive[4] = {0, 1, 0.5f, 1};

    static const float defaultColorMoving[4] = {0, 0.5f, 0.25f, 1};

    static const float defaultColorActive[4] = {0.5f, 0.5f, 0.5f, 1};

    static const float defaultColor[4] = {0.25f, 0.25f, 0.25f, 1};

    if (color.isSet())
        return color.getValue().data();
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



bool CollisionModel::insertInNode( objectmodel::BaseNode* node )
{
    node->addCollisionModel(this);
    Inherit1::insertInNode(node);
    return true;
}

bool CollisionModel::removeInNode( objectmodel::BaseNode* node )
{
    node->removeCollisionModel(this);
    Inherit1::removeInNode(node);
    return true;
}



} // namespace core

} // namespace sofa

