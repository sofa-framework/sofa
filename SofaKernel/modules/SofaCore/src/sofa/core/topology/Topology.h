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
#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGY_H
#define SOFA_CORE_TOPOLOGY_TOPOLOGY_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/list.h>
#include <sofa/defaulttype/TopologyTypes.h>
#include <climits>

namespace sofa
{

namespace core
{

namespace topology
{

/// The enumeration used to give unique identifiers to Topological objects.
enum class TopologyElementType
{
    UNKNOWN,
    POINT,
    EDGE,
    TRIANGLE,
    QUAD,
    TETRAHEDRON,
    HEXAHEDRON,
    PENTAHEDRON,
    PYRAMID    
};


enum [[deprecated("This enum has been deprecated in PR #1593 and will be removed in release 21.06. Please use TopologyElementType instead.")]] TopologyObjectType
{
    POINT,
    EDGE,
    TRIANGLE,
    QUAD,
    TETRAHEDRON,
    HEXAHEDRON,
    PENTAHEDRON,
    PYRAMID
};



SOFA_CORE_API TopologyElementType parseTopologyElementTypeFromString(const std::string& s);
SOFA_CORE_API std::string parseTopologyElementTypeToString(TopologyElementType t);

class SOFA_CORE_API Topology : public virtual core::objectmodel::BaseObject
{
public:
    /// Topology global typedefs

    typedef sofa::Index Index;
    static constexpr Index InvalidID = sofa::InvalidID;

    typedef Index                 ElemID;
    typedef Index                 PointID;
    typedef Index                 EdgeID;
    typedef Index                 TriangleID;
    typedef Index                 QuadID;
    typedef Index                 TetraID;
    typedef Index                 TetrahedronID;
    typedef Index                 HexaID;
    typedef Index                 HexahedronID;
    typedef Index                 PentahedronID;
    typedef Index                 PentaID;
    typedef Index                 PyramidID;

    typedef sofa::helper::vector<Index>                  SetIndex;
    typedef sofa::helper::vector<Index>                  SetIndices;

    typedef PointID                             Point;
    // in the following types, we use wrapper classes to have different types for each element, otherwise Quad and Tetrahedron would be the same
    class Edge : public sofa::helper::fixed_array<PointID,2>
    {
    public:
        Edge(): sofa::helper::fixed_array<PointID,2>(Topology::InvalidID, Topology::InvalidID){}
        Edge(PointID a, PointID b) : sofa::helper::fixed_array<PointID,2>(a,b) {}
    };

    class Triangle : public sofa::helper::fixed_array<PointID,3>
    {
    public:
        Triangle(): sofa::helper::fixed_array<PointID,3>(Topology::InvalidID, Topology::InvalidID, Topology::InvalidID) {}
        Triangle(PointID a, PointID b, PointID c) : sofa::helper::fixed_array<PointID,3>(a,b,c) {}
    };

    class Quad : public sofa::helper::fixed_array<PointID,4>
    {
    public:
        Quad(): sofa::helper::fixed_array<PointID,4>(Topology::InvalidID, Topology::InvalidID, Topology::InvalidID, Topology::InvalidID) {}
        Quad(PointID a, PointID b, PointID c, PointID d) : sofa::helper::fixed_array<PointID,4>(a,b,c,d) {}
    };

    class Tetrahedron : public sofa::helper::fixed_array<PointID,4>
    {
    public:
        Tetrahedron(): sofa::helper::fixed_array<PointID,4>(Topology::InvalidID, Topology::InvalidID, Topology::InvalidID, Topology::InvalidID) {}
        Tetrahedron(PointID a, PointID b, PointID c, PointID d) : sofa::helper::fixed_array<PointID,4>(a,b,c,d) {}
    };
    typedef Tetrahedron                         Tetra;

    class Pyramid : public sofa::helper::fixed_array<PointID,5>
    {
    public:
        Pyramid(): sofa::helper::fixed_array<PointID,5>(Topology::InvalidID, Topology::InvalidID, Topology::InvalidID, Topology::InvalidID, Topology::InvalidID) {}
        Pyramid(PointID a, PointID b, PointID c, PointID d, PointID e) : sofa::helper::fixed_array<PointID,5>(a,b,c,d,e) {}
    };

    class Pentahedron : public sofa::helper::fixed_array<PointID,6>
    {
    public:
        Pentahedron(): sofa::helper::fixed_array<PointID,6>(Topology::InvalidID, Topology::InvalidID, Topology::InvalidID, Topology::InvalidID, Topology::InvalidID, Topology::InvalidID) {}
        Pentahedron(PointID a, PointID b, PointID c, PointID d, PointID e, PointID f) : sofa::helper::fixed_array<PointID,6>(a,b,c,d,e,f) {}
    };
    typedef Pentahedron                          Penta;

    class Hexahedron : public sofa::helper::fixed_array<PointID,8>
    {
    public:
        Hexahedron(): sofa::helper::fixed_array<PointID,8>(Topology::InvalidID, Topology::InvalidID, Topology::InvalidID, Topology::InvalidID,
                                                           Topology::InvalidID, Topology::InvalidID, Topology::InvalidID, Topology::InvalidID) {}
        Hexahedron(PointID a, PointID b, PointID c, PointID d,
                   PointID e, PointID f, PointID g, PointID h) : sofa::helper::fixed_array<PointID,8>(a,b,c,d,e,f,g,h) {}
    };
    typedef Hexahedron                          Hexa;



    SOFA_CLASS(Topology, core::objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(Topology)
protected:
    Topology():BaseObject() {}
    ~Topology() override
    {}
public:
    // Access to embedded position information (in case the topology is a regular grid for instance)
    // This is not very clean and is quit slow but it should only be used during initialization

    virtual bool hasPos() const { return false; }
    virtual Size getNbPoints() const { return 0; }
    virtual void setNbPoints(Size /*n*/) {}
    virtual SReal getPX(Index /*i*/) const { return 0.0; }
    virtual SReal getPY(Index /*i*/) const { return 0.0; }
    virtual SReal getPZ(Index /*i*/) const { return 0.0; }


    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

    // Declare invalid topology structures filled with Topology::InvalidID
    static const sofa::helper::vector<Topology::Index> InvalidSet;
    static const Edge                                       InvalidEdge;
    static const Triangle                                   InvalidTriangle;
    static const Quad                                       InvalidQuad;
    static const Tetrahedron                                InvalidTetrahedron;
    static const Pyramid                                    InvalidPyramid;
    static const Pentahedron                                InvalidPentahedron;
    static const Hexahedron                                 InvalidHexahedron;
};


template<class TopologyElement>
struct TopologyElementInfo;

template<>
struct TopologyElementInfo<Topology::Point>
{
    static TopologyElementType type() { return TopologyElementType::POINT; }
    static const char* name() { return "Point"; }
};

template<>
struct TopologyElementInfo<Topology::Edge>
{
    static TopologyElementType type() { return TopologyElementType::EDGE; }
    static const char* name() { return "Edge"; }
};

template<>
struct TopologyElementInfo<Topology::Triangle>
{
    static TopologyElementType type() { return TopologyElementType::TRIANGLE; }
    static const char* name() { return "Triangle"; }
};

template<>
struct TopologyElementInfo<Topology::Quad>
{
    static TopologyElementType type() { return TopologyElementType::QUAD; }
    static const char* name() { return "Quad"; }
};

template<>
struct TopologyElementInfo<Topology::Tetrahedron>
{
    static TopologyElementType type() { return TopologyElementType::TETRAHEDRON; }
    static const char* name() { return "Tetrahedron"; }
};

template<>
struct TopologyElementInfo<Topology::Pyramid>
{
    static TopologyElementType type() { return TopologyElementType::PYRAMID; }
    static const char* name() { return "Pyramid"; }
};

template<>
struct TopologyElementInfo<Topology::Pentahedron>
{
    static TopologyElementType type() { return TopologyElementType::PENTAHEDRON; }
    static const char* name() { return "Pentahedron"; }
};

template<>
struct TopologyElementInfo<Topology::Hexahedron>
{
    static TopologyElementType type() { return TopologyElementType::HEXAHEDRON; }
    static const char* name() { return "Hexahedron"; }
};


extern const unsigned int edgesInTetrahedronArray[6][2];

extern const unsigned int edgesInHexahedronArray[12][2];
extern const unsigned int quadsInHexahedronArray[6][4];
extern const unsigned int verticesInHexahedronArray[2][2][2];


} // namespace topology

} // namespace core

} // namespace sofa

// Specialization of the defaulttype::DataTypeInfo type traits template

namespace sofa
{

namespace defaulttype
{

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Edge > : public FixedArrayTypeInfo<sofa::helper::fixed_array<Index,2> >
{
    static std::string name() { return "Edge"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Triangle > : public FixedArrayTypeInfo<sofa::helper::fixed_array<Index,3> >
{
    static std::string name() { return "Triangle"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Quad > : public FixedArrayTypeInfo<sofa::helper::fixed_array<Index,4> >
{
    static std::string name() { return "Quad"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Tetrahedron > : public FixedArrayTypeInfo<sofa::helper::fixed_array<Index,4> >
{
    static std::string name() { return "Tetrahedron"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Pyramid > : public FixedArrayTypeInfo<sofa::helper::fixed_array<Index,5> >
{
    static std::string name() { return "Pyramid"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Pentahedron > : public FixedArrayTypeInfo<sofa::helper::fixed_array<Index,6> >
{
    static std::string name() { return "Pentahedron"; }
};

template<>
struct DataTypeInfo< sofa::core::topology::Topology::Hexahedron > : public FixedArrayTypeInfo<sofa::helper::fixed_array<Index,8> >
{
    static std::string name() { return "Hexahedron"; }
};





} // namespace defaulttype

} // namespace sofa

#endif
