/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_SPHEREGRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_SPHEREGRIDTOPOLOGY_H
#include "config.h"

#include <SofaBaseTopology/GridTopology.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace topology
{

class SOFA_GENERAL_TOPOLOGY_API SphereGridTopology : public GridTopology
{
public:
    SOFA_CLASS(SphereGridTopology,GridTopology);
    typedef sofa::defaulttype::Vector3 Vector3;
protected:
    SphereGridTopology(int nx, int ny, int nz);
    SphereGridTopology();
public:
    unsigned getIndex( int i, int j, int k ) const; ///< one-dimensional index of a grid point
    Vector3 getPoint(int i) const;
    Vector3 getPoint(int x, int y, int z) const;
    bool hasPos()  const { return true; }
    SReal getPX(int i)  const { return getPoint(i)[0]; }
    SReal getPY(int i) const { return getPoint(i)[1]; }
    SReal getPZ(int i) const { return getPoint(i)[2]; }

    void setCenter(SReal x, SReal y, SReal z);
    void setAxis(SReal x, SReal y, SReal z);
    void setRadius(SReal radius);

protected:
    Data< Vector3 > d_center;
    Data< Vector3 > d_axis;
    Data< SReal > d_radius;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_SPHEREGRIDTOPOLOGY_H
