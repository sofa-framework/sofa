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
#ifndef RADIUSCONTAINER_H_
#define RADIUSCONTAINER_H_

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace component
{

namespace container
{

class RadiusContainer : public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(RadiusContainer,sofa::core::objectmodel::BaseObject);

    /// Get the radius around a given point
    virtual double getPointRadius(unsigned int index) = 0;
    /// Get the radius around a given edge
    virtual double getEdgeRadius(unsigned int index) = 0;
    /// Deprecated alias for getEdgeRadius
    double getRadius(unsigned int index) { return getEdgeRadius(index); }
};

}

}

}
#endif /*RADIUSCONTAINER_H_*/
