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
#include <sofa/component/topology/container/constant/SphereQuadTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::container::constant
{

using namespace sofa::type;
using namespace sofa::defaulttype;

int SphereQuadTopologyClass = core::RegisterObject("Sphere topology constructed with deformed quads")
        .addAlias("SphereQuad")
        .add< SphereQuadTopology >()
        ;

SphereQuadTopology::SphereQuadTopology(int nx, int ny, int nz)
    : CubeTopology(nx, ny, nz),
      center(initData(&center,Vector3(0.0_sreal,0.0_sreal,0.0_sreal),"center", "Center of the sphere")),
      radius(initData(&radius,1.0_sreal,"radius", "Radius of the sphere"))
{
}

SphereQuadTopology::SphereQuadTopology()
    : center(initData(&center,Vector3(0.0_sreal,0.0_sreal,0.0_sreal),"center", "Center of the sphere")),
      radius(initData(&radius,0_sreal,"radius", "Radius of the sphere"))
{
}

Vector3 SphereQuadTopology::getPoint(int x, int y, int z) const
{
    Vector3 p((2*x)/(SReal)(nx.getValue()-1) - 1, (2*y)/(SReal)(ny.getValue()-1) - 1, (2*z)/(SReal)(nz.getValue()-1) - 1);
    p.normalize();
    return center.getValue()+p*radius.getValue();
}

} // namespace sofa::component::topology::container::constant
