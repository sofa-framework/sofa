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
#include <sofa/geometry/ElementType.h>
#include <sofa/geometry/ElementInfo.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa::core::topology
{

// This class should be deprecated in the near future, and its only use is to be included in the Node topology Sequence.
// As for now, it is mainly used for compatibility reason (and its inheritance on BaseObject...) against BaseMeshTopology
class SOFA_CORE_API Topology : public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(Topology, core::objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(Topology)

    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index Index;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    static constexpr Index InvalidID = sofa::InvalidID;

    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index ElemID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index PointID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index Point;

    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index EdgeID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index TriangleID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index QuadID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index TetraID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index TetrahedronID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index PyramidID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index PentahedronID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index PentaID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index HexahedronID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index HexaID;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED__ALIASES_INDEX()
    typedef sofa::Index PrismID;

    inline static const auto InvalidSet = sofa::topology::InvalidSet;
    static constexpr auto InvalidEdge = sofa::topology::InvalidEdge;
    static constexpr auto InvalidTriangle = sofa::topology::InvalidTriangle;
    static constexpr auto InvalidQuad = sofa::topology::InvalidQuad;
    static constexpr auto InvalidTetrahedron = sofa::topology::InvalidTetrahedron;
    static constexpr auto InvalidPrism = sofa::topology::InvalidPrism;
    static constexpr auto InvalidHexahedron = sofa::topology::InvalidHexahedron;
    static constexpr auto InvalidPyramid = sofa::topology::InvalidPyramid;

    using SetIndex = sofa::topology::SetIndex;
    using SetIndices = sofa::topology::SetIndices;

    using Edge = sofa::topology::Element<sofa::geometry::Edge>;
    using Triangle = sofa::topology::Element<sofa::geometry::Triangle>;
    using Quad = sofa::topology::Element<sofa::geometry::Quad>;
    using Tetrahedron = sofa::topology::Element<sofa::geometry::Tetrahedron>;
    using Prism = sofa::topology::Element<sofa::geometry::Prism>;
    using Pyramid = sofa::topology::Element<sofa::geometry::Pyramid>;
    using Hexahedron = sofa::topology::Element<sofa::geometry::Hexahedron>;

    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED("Tetra alias has been deprecated, please use Tetrahedron instead")
    typedef Tetrahedron Tetra;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED("Penta alias has been deprecated, please use Prism instead")
    typedef Prism Penta;
    SOFA_CORE_TOPOLOGY_ATTRIBUTE_DEPRECATED("Hexa alias has been deprecated, please use Hexahedron instead")
    typedef Hexahedron Hexa;

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

/// List of pair of vertex indices (edge) in a tetrahedron
SOFA_CORE_API extern const unsigned int edgesInTetrahedronArray[6][2];
/// List of 3 vertex indices (triangle) in a tetrahedron
SOFA_CORE_API extern const unsigned int trianglesOrientationInTetrahedronArray[4][3];

/// List of pair of vertex indices (edge) in a hexahedron
SOFA_CORE_API extern const unsigned int edgesInHexahedronArray[12][2];

/// List of 4 vertex indices (quad) in a hexahedron
SOFA_CORE_API extern const unsigned int quadsOrientationInHexahedronArray[6][4];
// List of vertex indices in a hexahedron
SOFA_CORE_API extern const unsigned int verticesInHexahedronArray[2][2][2];

} // namespace sofa::core::topology

namespace sofa::geometry
{

// Specialization required because Topology::Point is not an alias of
// sofa::topology::Element, compared to the other aliases such as
// Edge, Triangle...
template<>
struct ElementInfo<sofa::core::topology::Topology::Point>
{
    static geometry::ElementType type()
    {
        return geometry::ElementType::POINT;
    }

    static const char* name()
    {
        static const char* n = elementTypeToString(type());
        return n;
    }
};
}
