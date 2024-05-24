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
#include <sofa/component/topology/container/constant/CubeTopology.h>

namespace sofa::component::topology::container::constant
{

class SOFA_COMPONENT_TOPOLOGY_CONTAINER_CONSTANT_API SphereQuadTopology : public CubeTopology
{
public:
    SOFA_CLASS(SphereQuadTopology,CubeTopology);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);

protected:
    SphereQuadTopology(int nx, int ny, int nz);
    SphereQuadTopology();
public:
    Vec3 getPoint(int x, int y, int z) const override;

protected:

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_CONSTANT()
    Data< Vec3 > center;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_CONTAINER_CONSTANT()
    Data< SReal > radius;


    Data< Vec3 > d_center; ///< Center of the sphere
    Data< SReal > d_radius; ///< Radius of the sphere
};

} // namespace sofa::component::topology::container::constant
