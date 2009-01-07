/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDGEOMETRYALGORITHMS_INL

#include <sofa/component/topology/DynamicSparseGridGeometryAlgorithms.h>
#include <sofa/component/topology/CommonAlgorithms.h>

namespace sofa
{

namespace component
{

namespace topology
{

/// finds a hexahedron which is nearest to a given point. Computes barycentric coordinates and a distance measure.
template < class DataTypes >
int DynamicSparseGridGeometryAlgorithms<DataTypes>::findNearestElement ( const Coord& pos, defaulttype::Vector3& baryC, Real& distance ) const
{
    return HexahedronSetGeometryAlgorithms<DataTypes>::findNearestElement ( pos, baryC, distance );
}

/// given a vector of points, find the nearest hexa for each point. Computes barycentric coordinates and a distance measure.
template < class DataTypes >
void DynamicSparseGridGeometryAlgorithms<DataTypes>::findNearestElements ( const VecCoord& pos, helper::vector<int>& elem,
        helper::vector<defaulttype::Vector3>& baryC, helper::vector<Real>& dist ) const
{
    HexahedronSetGeometryAlgorithms<DataTypes>::findNearestElements ( pos, elem, baryC, dist );
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_HexahedronSetTOPOLOGY_INL
