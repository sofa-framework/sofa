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
#include <sofa/topology/TopologyElementType.h>
#include <sofa/topology/TopologyElementInfo.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa::core::topology
{

using TopologyElementType 
[[deprecated("PR1xxx: sofa::core::topology::TopologyElementType has moved to sofa::topology::TopologyElementType. This compatibility layer will be removed for the v21.12 release.")]] 
= sofa::topology::TopologyElementType;

template<class TopologyElement>
using TopologyElementInfo 
[[deprecated("PR1xxx: sofa::core::topology::TopologyElementInfo has moved to sofa::topology::TopologyElementInfo. This compatibility layer will be removed for the v21.12 release.")]]
= sofa::topology::TopologyElementInfo<TopologyElement>;

// This class should be deprecated in the near future, and its only use is to be included in the Node topology Sequence.
// As for now, it is mainly used for compatibility reason (and its inheritance on BaseObject...)
class SOFA_CORE_API Topology : public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(Topology, core::objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(Topology)

    using Index = sofa::topology::geometry::Index;
    static constexpr Index InvalidID = sofa::topology::geometry::InvalidID;

    using ElemID = sofa::topology::geometry::ElemID;
    using PointID = sofa::topology::geometry::PointID;
    using EdgeID = sofa::topology::geometry::EdgeID;
    using TriangleID = sofa::topology::geometry::TriangleID;
    using QuadID = sofa::topology::geometry::QuadID;
    using TetraID = sofa::topology::geometry::TetraID;
    using TetrahedronID = sofa::topology::geometry::TetrahedronID;
    using HexaID = sofa::topology::geometry::HexaID;
    using HexahedronID = sofa::topology::geometry::HexahedronID;
    using PentahedronID = sofa::topology::geometry::PentahedronID;
    using PentaID = sofa::topology::geometry::PentaID;
    using PyramidID = sofa::topology::geometry::PyramidID;

    inline static auto InvalidSet = sofa::topology::geometry::InvalidSet;
    inline static auto InvalidEdge = sofa::topology::geometry::InvalidEdge;
    inline static auto InvalidTriangle = sofa::topology::geometry::InvalidTriangle;
    inline static auto InvalidQuad = sofa::topology::geometry::InvalidQuad;
    inline static auto InvalidTetrahedron = sofa::topology::geometry::InvalidTetrahedron;
    inline static auto InvalidPentahedron = sofa::topology::geometry::InvalidPentahedron;
    inline static auto InvalidHexahedron = sofa::topology::geometry::InvalidHexahedron;
    inline static auto InvalidPyramid = sofa::topology::geometry::InvalidPyramid;

    using SetIndex = sofa::topology::geometry::SetIndex;
    using SetIndices = sofa::topology::geometry::SetIndices;

    using Point = PointID;
    using Edge = sofa::topology::geometry::Edge;
    using Triangle = sofa::topology::geometry::Triangle;
    using Quad = sofa::topology::geometry::Quad;
    using Tetrahedron = sofa::topology::geometry::Tetrahedron;
    using Tetra = sofa::topology::geometry::Tetra;
    using Pentahedron = sofa::topology::geometry::Pentahedron;
    using Penta = sofa::topology::geometry::Penta;
    using Pyramid = sofa::topology::geometry::Pyramid;
    using Hexahedron = sofa::topology::geometry::Hexahedron;
    using Hexa = sofa::topology::geometry::Hexa;
        
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

// Specialization of the defaulttype::DataTypeInfo type traits template
