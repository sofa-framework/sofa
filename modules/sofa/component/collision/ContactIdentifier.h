/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_CONTACTIDENTIFIER_H
#define SOFA_COMPONENT_COLLISION_CONTACTIDENTIFIER_H

#include <sofa/core/collision/DetectionOutput.h>

#include <sofa/component/initConstraint.h>

#include <list>


namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_CONSTRAINT_API ContactIdentifier
{
public:
    ContactIdentifier()
    {
        if (!availableId.empty())
        {
            id = availableId.front();
            availableId.pop_front();
        }
        else
            id = cpt++;
    }

    virtual ~ContactIdentifier()
    {
        availableId.push_back(id);
    }

protected:
    static sofa::core::collision::DetectionOutput::ContactId cpt;
    sofa::core::collision::DetectionOutput::ContactId id;
    static std::list<sofa::core::collision::DetectionOutput::ContactId> availableId;
};

inline long cantorPolynomia(sofa::core::collision::DetectionOutput::ContactId x, sofa::core::collision::DetectionOutput::ContactId y)
{
    // Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
    return (long)(((x+y)*(x+y)+3*x+y)/2);
}

} // collision

} // component

} // sofa

#endif // SOFA_COMPONENT_COLLISION_CONTACTIDENTIFIER_H
