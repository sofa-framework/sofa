/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/collision/response/contact/config.h>

#include <sofa/core/collision/DetectionOutput.h>
#include <list>
#include <atomic>


namespace sofa::component::collision::response::contact
{

class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API ContactIdentifier
{
public:
    ContactIdentifier();
    virtual ~ContactIdentifier() = default;

protected:
    inline static std::atomic<sofa::core::collision::DetectionOutput::ContactId> cpt = 0;
    sofa::core::collision::DetectionOutput::ContactId id;
};

inline long cantorPolynomia(sofa::core::collision::DetectionOutput::ContactId x, sofa::core::collision::DetectionOutput::ContactId y)
{
    // Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
    return (long)(((x+y)*(x+y)+3*x+y)/2);
}

} // namespace sofa::component::collision::response::contact
