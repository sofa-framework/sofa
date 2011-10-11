/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/helper/system/config.h>
#include <sofa/component/initBaseCollision.h>


namespace sofa
{

namespace component
{


void initBaseCollision()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

SOFA_LINK_CLASS(DefaultPipeline)
SOFA_LINK_CLASS(Sphere)
SOFA_LINK_CLASS(Cube)
SOFA_LINK_CLASS(DiscreteIntersection)
SOFA_LINK_CLASS(DefaultContactManager)
SOFA_LINK_CLASS(Point)
SOFA_LINK_CLASS(Line)
SOFA_LINK_CLASS(Triangle)
SOFA_LINK_CLASS(TetrahedronModel)
SOFA_LINK_CLASS(SpatialGridPointModel)
SOFA_LINK_CLASS(SphereTreeModel)
SOFA_LINK_CLASS(LineLocalMinDistanceFilter)
SOFA_LINK_CLASS(PointLocalMinDistanceFilter)
SOFA_LINK_CLASS(Ray)


} // namespace component

} // namespace sofa
