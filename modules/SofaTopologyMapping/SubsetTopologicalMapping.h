/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_SUBSETTOPOLOGICALMAPPING_H
#define SOFA_COMPONENT_TOPOLOGY_SUBSETTOPOLOGICALMAPPING_H
#include "config.h"

#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/defaulttype/Vec.h>
#include <map>

#include <sofa/core/BaseMapping.h>

namespace sofa
{

namespace component
{

namespace topology
{

/**
 * This class is a specific implementation of TopologicalMapping where the destination topology should be kept identical to the source topology.
 * The implementation currently assumes that both topology have been initialized identically.
 */

class SOFA_TOPOLOGY_MAPPING_API SubsetTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(SubsetTopologicalMapping,sofa::core::topology::TopologicalMapping);

    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;
    typedef core::topology::BaseMeshTopology::index_type Index;

protected:
    SubsetTopologicalMapping();
    virtual ~SubsetTopologicalMapping();
public:

    Data<bool> samePoints; ///< True if the same set of points is used in both topologies
    Data<bool> handleEdges; ///< True if edges events and mapping should be handled
    Data<bool> handleTriangles; ///< True if triangles events and mapping should be handled
    Data<bool> handleQuads; ///< True if quads events and mapping should be handled
    Data<bool> handleTetrahedra; ///< True if tetrahedra events and mapping should be handled
    Data<bool> handleHexahedra; ///< True if hexahedra events and mapping should be handled
    Data<SetIndex> pointS2D; ///< Internal source -> destination topology points map
    Data<SetIndex> pointD2S; ///< Internal destination -> source topology points map (link to SubsetMapping::indices to handle the mechanical-side of the mapping
    Data<SetIndex> edgeS2D; ///< Internal source -> destination topology edges map
    Data<SetIndex> edgeD2S; ///< Internal destination -> source topology edges map
    Data<SetIndex> triangleS2D; ///< Internal source -> destination topology triangles map
    Data<SetIndex> triangleD2S; ///< Internal destination -> source topology triangles map
    Data<SetIndex> quadS2D; ///< Internal source -> destination topology quads map
    Data<SetIndex> quadD2S; ///< Internal destination -> source topology quads map
    Data<SetIndex> tetrahedronS2D; ///< Internal source -> destination topology tetrahedra map
    Data<SetIndex> tetrahedronD2S; ///< Internal destination -> source topology tetrahedra map
    Data<SetIndex> hexahedronS2D; ///< Internal source -> destination topology hexahedra map
    Data<SetIndex> hexahedronD2S; ///< Internal destination -> source topology hexahedra map

    virtual void init() override;

    /** \brief Translates the TopologyChange objects from the source to the target.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the first topology changes on the second topology.
     *
     */
    virtual void updateTopologicalMappingTopDown() override;

    virtual bool isTheOutputTopologySubdividingTheInputOne() override { return true; }
    virtual unsigned int getGlobIndex(unsigned int ind) override;
    virtual unsigned int getFromIndex(unsigned int ind) override;

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TETRA2TRIANGLETOPOLOGICALMAPPING_H
