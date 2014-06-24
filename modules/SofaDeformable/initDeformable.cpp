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
#include <sofa/helper/system/config.h>
#include <SofaDeformable/initDeformable.h>


namespace sofa
{

namespace component
{


void initDeformable()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

SOFA_LINK_CLASS(QuadularBendingSprings);
SOFA_LINK_CLASS(RestShapeSpringsForceField);
SOFA_LINK_CLASS(TriangularBendingSprings);
SOFA_LINK_CLASS(TriangularBiquadraticSpringsForceField);
SOFA_LINK_CLASS(TriangularQuadraticSpringsForceField);
SOFA_LINK_CLASS(TriangularTensorMassForceField);
SOFA_LINK_CLASS(FrameSpringForceField);
SOFA_LINK_CLASS(MeshSpringForceField);
SOFA_LINK_CLASS(QuadBendingSprings);
SOFA_LINK_CLASS(RegularGridSpringForceField);
SOFA_LINK_CLASS(SpringForceField);
SOFA_LINK_CLASS(StiffSpringForceField);
SOFA_LINK_CLASS(TriangleBendingSprings);
SOFA_LINK_CLASS(VectorSpringForceField);


} // namespace component

} // namespace sofa
