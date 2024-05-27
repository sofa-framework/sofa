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
#include <sofa/component/topology/mapping/config.h>

#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/type/Vec.h>
#include <map>

#include <sofa/core/BaseMapping.h>

namespace sofa::component::topology::mapping
{

/**
 * This class is a specific implementation of TopologicalMapping where the destination topology should be kept identical to the source topology.
 * The implementation currently assumes that both topology have been initialized identically.
 */

class SOFA_COMPONENT_TOPOLOGY_MAPPING_API SubsetTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(SubsetTopologicalMapping,sofa::core::topology::TopologicalMapping);

    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;
    typedef core::topology::BaseMeshTopology::Index Index;

protected:
    SubsetTopologicalMapping();
    ~SubsetTopologicalMapping() override;
public:

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<bool> samePoints;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<bool> handleEdges;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<bool> handleTriangles;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<bool> handleQuads;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<bool> handleTetrahedra;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<bool> handleHexahedra;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> pointS2D;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> pointD2S;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> edgeS2D;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> edgeD2S;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> triangleS2D;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> triangleD2S;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> quadS2D;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> quadD2S;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> tetrahedronS2D;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> tetrahedronD2S;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> hexahedronS2D;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<SetIndex> hexahedronD2S;

    Data<bool> d_samePoints; ///< True if the same set of points is used in both topologies
    Data<bool> d_handleEdges; ///< True if edges events and mapping should be handled
    Data<bool> d_handleTriangles; ///< True if triangles events and mapping should be handled
    Data<bool> d_handleQuads; ///< True if quads events and mapping should be handled
    Data<bool> d_handleTetrahedra; ///< True if tetrahedra events and mapping should be handled
    Data<bool> d_handleHexahedra; ///< True if hexahedra events and mapping should be handled
    Data<SetIndex> d_pointS2D; ///< Internal source -> destination topology points map
    Data<SetIndex> d_pointD2S; ///< Internal destination -> source topology points map (link to SubsetMapping::indices to handle the mechanical-side of the mapping
    Data<SetIndex> d_edgeS2D; ///< Internal source -> destination topology edges map
    Data<SetIndex> d_edgeD2S; ///< Internal destination -> source topology edges map
    Data<SetIndex> d_triangleS2D; ///< Internal source -> destination topology triangles map
    Data<SetIndex> d_triangleD2S; ///< Internal destination -> source topology triangles map
    Data<SetIndex> d_quadS2D; ///< Internal source -> destination topology quads map
    Data<SetIndex> d_quadD2S; ///< Internal destination -> source topology quads map
    Data<SetIndex> d_tetrahedronS2D; ///< Internal source -> destination topology tetrahedra map
    Data<SetIndex> d_tetrahedronD2S; ///< Internal destination -> source topology tetrahedra map
    Data<SetIndex> d_hexahedronS2D; ///< Internal source -> destination topology hexahedra map
    Data<SetIndex> d_hexahedronD2S; ///< Internal destination -> source topology hexahedra map

    void init() override;

    /** \brief Translates the TopologyChange objects from the source to the target.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the first topology changes on the second topology.
     *
     */
    void updateTopologicalMappingTopDown() override;

    bool isTheOutputTopologySubdividingTheInputOne() override { return true; }
    Index getGlobIndex(Index ind) override;
    Index getFromIndex(Index ind) override;

};

} //namespace sofa::component::topology::mapping
