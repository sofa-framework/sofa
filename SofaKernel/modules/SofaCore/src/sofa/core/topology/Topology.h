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

#include <sofa/core/config.h>
#include <sofa/topology/Topology.h>
#include <sofa/topology/ElementType.h>
#include <sofa/topology/ElementInfo.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa::core::topology
{

using TopologyElementType 
[[deprecated("PR1xxx: sofa::core::topology::TopologyElementType has moved to sofa::topology::ElementType. This compatibility layer will be removed for the v21.12 release.")]] 
= sofa::topology::ElementType;

template<class TopologyElement>
using TopologyElementInfo 
[[deprecated("PR1xxx: sofa::core::topology::TopologyElementInfo has moved to sofa::topology::ElementInfo. This compatibility layer will be removed for the v21.12 release.")]]
= sofa::topology::ElementInfo<TopologyElement>;

// This class should be deprecated in the near future, and its only use is to be included in the Node topology Sequence.
// As for now, it is mainly used for compatibility reason (and its inheritance on BaseObject...) against BaseMeshTopology
class SOFA_CORE_API Topology : public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(Topology, core::objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(Topology)

    using Index = sofa::topology::Index;
    static constexpr Index InvalidID = sofa::topology::InvalidID;

    using ElemID = sofa::topology::ElemID;
    using PointID = sofa::topology::PointID;
    using EdgeID = sofa::topology::EdgeID;
    using TriangleID = sofa::topology::TriangleID;
    using QuadID = sofa::topology::QuadID;
    using TetraID = sofa::topology::TetraID;
    using TetrahedronID = sofa::topology::TetrahedronID;
    using HexaID = sofa::topology::HexaID;
    using HexahedronID = sofa::topology::HexahedronID;
    using PentahedronID = sofa::topology::PentahedronID;
    using PentaID = sofa::topology::PentaID;
    using PyramidID = sofa::topology::PyramidID;

    inline static auto InvalidSet = sofa::topology::InvalidSet;
    inline static auto InvalidEdge = sofa::topology::InvalidEdge;
    inline static auto InvalidTriangle = sofa::topology::InvalidTriangle;
    inline static auto InvalidQuad = sofa::topology::InvalidQuad;
    inline static auto InvalidTetrahedron = sofa::topology::InvalidTetrahedron;
    inline static auto InvalidPentahedron = sofa::topology::InvalidPentahedron;
    inline static auto InvalidHexahedron = sofa::topology::InvalidHexahedron;
    inline static auto InvalidPyramid = sofa::topology::InvalidPyramid;

    using SetIndex = sofa::topology::SetIndex;
    using SetIndices = sofa::topology::SetIndices;

    using Point = PointID;
    using Edge = sofa::topology::Edge;
    using Triangle = sofa::topology::Triangle;
    using Quad = sofa::topology::Quad;
    using Tetrahedron = sofa::topology::Tetrahedron;
    using Tetra = sofa::topology::Tetra;
    using Pentahedron = sofa::topology::Pentahedron;
    using Penta = sofa::topology::Penta;
    using Pyramid = sofa::topology::Pyramid;
    using Hexahedron = sofa::topology::Hexahedron;
    using Hexa = sofa::topology::Hexa;
        
    bool insertInNode(objectmodel::BaseNode* node) override;
    bool removeInNode(objectmodel::BaseNode* node) override;

protected:
    Topology() {}
    virtual ~Topology() {}
public:
    // Access to embedded position information (in case the topology is a regular grid for instance)
    // This is not very clean and is quit slow but it should only be used during initialization

    virtual bool hasPos() const { return false; }
    virtual Size getNbPoints() const { return 0; }
    virtual void setNbPoints(Size /*n*/) {}
    virtual SReal getPX(Index /*i*/) const { return 0.0; }
    virtual SReal getPY(Index /*i*/) const { return 0.0; }
    virtual SReal getPZ(Index /*i*/) const { return 0.0; }
};

} // namespace sofa::core::topology
