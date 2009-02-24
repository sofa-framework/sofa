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
#ifndef SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_DYNAMICSPARSEGRIDTOPOLOGYCONTAINER_H

#include <sofa/component/topology/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/HexahedronData.h>
#include <map>

namespace sofa
{
namespace component
{
namespace topology
{
typedef defaulttype::Vec<3, int> Vec3i;

/** a class that stores a sparse regular grid of hexahedra and provides a better loading and access to neighbors than HexahedronSetTopologyContainer */
class DynamicSparseGridTopologyContainer : public HexahedronSetTopologyContainer
{
    friend class DynamicSparseGridTopologyModifier;

public:
    typedef Hexa Hexahedron;
    typedef HexaEdges HexahedronEdges;
    typedef HexaQuads HexahedronQuads;

    Data< Vec3i> resolution;

    sofa::helper::vector<unsigned char> valuesIndexedInRegularGrid; // dense. valeurs dans toute la grille.
    HexahedronData<unsigned char> valuesIndexedInTopology; // pas dense. uniquement la ou il y a des hexas.

    HexahedronData<BaseMeshTopology::HexaID> idxInRegularGrid;
    std::map< unsigned int, BaseMeshTopology::HexaID> idInRegularGrid2IndexInTopo;
    defaulttype::Vector3 voxelSize;

    DynamicSparseGridTopologyContainer();
    DynamicSparseGridTopologyContainer ( const sofa::helper::vector< Hexahedron > &hexahedra );
    virtual ~DynamicSparseGridTopologyContainer() {}

protected:
    virtual void loadFromMeshLoader ( sofa::component::MeshLoader* loader );
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
