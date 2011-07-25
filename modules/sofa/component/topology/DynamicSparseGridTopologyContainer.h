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
#ifndef SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYCONTAINER_H

#include <sofa/component/topology/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/HexahedronData.h>
#include <sofa/helper/map.h>

namespace sofa
{
namespace component
{
namespace topology
{
typedef defaulttype::Vec<3, int> Vec3i;

/** a class that stores a sparse regular grid of hexahedra and provides a better loading and access to neighbors than HexahedronSetTopologyContainer */
class SOFA_COMPONENT_CONTAINER_API DynamicSparseGridTopologyContainer : public HexahedronSetTopologyContainer
{
    friend class DynamicSparseGridTopologyModifier;

public:
    SOFA_CLASS(DynamicSparseGridTopologyContainer,HexahedronSetTopologyContainer);

    typedef Hexa Hexahedron;
    typedef EdgesInHexahedron EdgesInHexahedron;
    typedef QuadsInHexahedron QuadsInHexahedron;

    Data< Vec3i> resolution;

    Data< sofa::helper::vector<unsigned char> > valuesIndexedInRegularGrid;
    HexahedronData<unsigned char> valuesIndexedInTopology;

    Data< sofa::helper::vector<BaseMeshTopology::HexaID> > idxInRegularGrid;
    Data< std::map< unsigned int, BaseMeshTopology::HexaID> > idInRegularGrid2IndexInTopo;
    Data< defaulttype::Vector3 > voxelSize;

    DynamicSparseGridTopologyContainer();
    virtual ~DynamicSparseGridTopologyContainer() {}
    void init();

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
