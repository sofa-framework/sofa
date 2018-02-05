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
#ifndef SOFA_COMPONENT_TOPOLOGY_SPHEREQUADTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_SPHEREQUADTOPOLOGY_H
#include "config.h"

#include <SofaGeneralTopology/CubeTopology.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace topology
{

class SOFA_GENERAL_TOPOLOGY_API SphereQuadTopology : public CubeTopology
{
public:
    SOFA_CLASS(SphereQuadTopology,CubeTopology);
    typedef sofa::defaulttype::Vector3 Vector3;
protected:
    SphereQuadTopology(int nx, int ny, int nz);
    SphereQuadTopology();
public:
    Vector3 getPoint(int x, int y, int z) const override;

protected:
    Data< Vector3 > center;
    Data< SReal > radius;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
