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

    Data<bool> samePoints;
    Data<bool> handleEdges;
    Data<bool> handleTriangles;
    Data<bool> handleQuads;
    Data<bool> handleTetrahedra;
    Data<bool> handleHexahedra;
    Data<SetIndex> pointS2D, pointD2S;
    Data<SetIndex> edgeS2D, edgeD2S;
    Data<SetIndex> triangleS2D, triangleD2S;
    Data<SetIndex> quadS2D, quadD2S;
    Data<SetIndex> tetrahedronS2D, tetrahedronD2S;
    Data<SetIndex> hexahedronS2D, hexahedronD2S;

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
