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

#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/fwd.h>
#include <sofa/helper/list.h>

namespace sofa::core::topology
{

namespace 
{
    using TopologyElementType = sofa::geometry::ElementType;
}

/// Topology identification of a primitive element
struct TopologyElemID
{
    TopologyElemID() : type(TopologyElementType::POINT), index((Topology::ElemID)-1) {}

    TopologyElemID(TopologyElementType _type, Topology::ElemID _index)
        : type(_type)
        , index(_index)
    {}

    TopologyElementType type;
    Topology::ElemID index;
};

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const TopologyElemID& d);
SOFA_CORE_API std::istream& operator >> (std::istream& in, TopologyElemID& d);

/// Topology change informations related to the ancestor topology element of a point
struct PointAncestorElem
{
    typedef type::Vec3 LocalCoords;

    PointAncestorElem() : type(TopologyElementType::POINT), index(sofa::InvalidID) {}

    PointAncestorElem(TopologyElementType _type, Topology::ElemID _index, const LocalCoords& _localCoords)
        : type(_type)
        , index(_index)
        , localCoords(_localCoords)
    {}
    
    TopologyElementType type;
    Topology::ElemID index;
    LocalCoords localCoords;
};

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const PointAncestorElem& d);
SOFA_CORE_API std::istream& operator >> (std::istream& in, PointAncestorElem& d);

/// Topology change informations related to the ancestor topology element of an edge
template<int NV>
struct ElemAncestorElem
{

    ElemAncestorElem()
    {}

    ElemAncestorElem(const type::fixed_array<PointAncestorElem,NV>& _pointSrcElems,
        const type::vector<TopologyElemID>& _srcElems)
        : pointSrcElems(_pointSrcElems)
        , srcElems(_srcElems)
    {}
    
    ElemAncestorElem(const type::fixed_array<PointAncestorElem,NV>& _pointSrcElems,
        const TopologyElemID& _srcElem)
        : pointSrcElems(_pointSrcElems)
        , srcElems()
    {
        srcElems.push_back(_srcElem);
    }
    
    type::fixed_array<PointAncestorElem,NV> pointSrcElems;
    type::vector<TopologyElemID> srcElems;
};

template<int NV>
SOFA_CORE_API std::ostream& operator << (std::ostream& out, const ElemAncestorElem<NV>& d);
template<int NV>
SOFA_CORE_API std::istream& operator >> (std::istream& in, ElemAncestorElem<NV>& d);

typedef ElemAncestorElem<2> EdgeAncestorElem;
typedef ElemAncestorElem<3> TriangleAncestorElem;
typedef ElemAncestorElem<4> QuadAncestorElem;
typedef ElemAncestorElem<4> TetrahedronAncestorElem;
typedef ElemAncestorElem<8> HexahedronAncestorElem;

template<class TopologyElement>
struct TopologyChangeElementInfo;

template<>
struct TopologyChangeElementInfo<Topology::Point>
{
    enum { USE_EMOVED          = 1 };
    enum { USE_EMOVED_REMOVING = 0 };
    enum { USE_EMOVED_ADDING   = 0 };

    typedef PointsIndicesSwap    EIndicesSwap;
    typedef PointsRenumbering    ERenumbering;
    typedef PointsAdded          EAdded;
    typedef PointsRemoved        ERemoved;
    typedef PointsMoved          EMoved;
    /// This event is not used for this type of element
    class EMoved_Removing { }; 
    /// This event is not used for this type of element
    class EMoved_Adding { };

    typedef PointAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<Topology::Edge>
{
    enum { USE_EMOVED          = 0 };
    enum { USE_EMOVED_REMOVING = 1 };
    enum { USE_EMOVED_ADDING   = 1 };

    typedef EdgesIndicesSwap    EIndicesSwap;
    typedef EdgesRenumbering    ERenumbering;
    typedef EdgesAdded          EAdded;
    typedef EdgesRemoved        ERemoved;
    typedef EdgesMoved_Removing EMoved_Removing;
    typedef EdgesMoved_Adding   EMoved_Adding;
    /// This event is not used for this type of element
    class EMoved ;

    typedef EdgeAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<Topology::Triangle>
{
    enum { USE_EMOVED          = 0 };
    enum { USE_EMOVED_REMOVING = 1 };
    enum { USE_EMOVED_ADDING   = 1 };

    typedef TrianglesIndicesSwap    EIndicesSwap;
    typedef TrianglesRenumbering    ERenumbering;
    typedef TrianglesAdded          EAdded;
    typedef TrianglesRemoved        ERemoved;
    typedef TrianglesMoved_Removing EMoved_Removing;
    typedef TrianglesMoved_Adding   EMoved_Adding;
    /// This event is not used for this type of element
    class EMoved { };

    typedef TriangleAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<Topology::Quad>
{
    enum { USE_EMOVED          = 0 };
    enum { USE_EMOVED_REMOVING = 1 };
    enum { USE_EMOVED_ADDING   = 1 };

    typedef QuadsIndicesSwap    EIndicesSwap;
    typedef QuadsRenumbering    ERenumbering;
    typedef QuadsAdded          EAdded;
    typedef QuadsRemoved        ERemoved;
    typedef QuadsMoved_Removing EMoved_Removing;
    typedef QuadsMoved_Adding   EMoved_Adding;
    /// This event is not used for this type of element
    class EMoved { };

    typedef QuadAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<Topology::Tetrahedron>
{
    enum { USE_EMOVED          = 0 };
    enum { USE_EMOVED_REMOVING = 1 };
    enum { USE_EMOVED_ADDING   = 1 };

    typedef TetrahedraIndicesSwap    EIndicesSwap;
    typedef TetrahedraRenumbering    ERenumbering;
    typedef TetrahedraAdded          EAdded;
    typedef TetrahedraRemoved        ERemoved;
    typedef TetrahedraMoved_Removing EMoved_Removing;
    typedef TetrahedraMoved_Adding   EMoved_Adding;
    /// This event is not used for this type of element
    class EMoved { };

    typedef TetrahedronAncestorElem AncestorElem;
};

template<>
struct TopologyChangeElementInfo<Topology::Hexahedron>
{
    enum { USE_EMOVED          = 0 };
    enum { USE_EMOVED_REMOVING = 1 };
    enum { USE_EMOVED_ADDING   = 1 };

    typedef HexahedraIndicesSwap    EIndicesSwap;
    typedef HexahedraRenumbering    ERenumbering;
    typedef HexahedraAdded          EAdded;
    typedef HexahedraRemoved        ERemoved;
    typedef HexahedraMoved_Removing EMoved_Removing;
    typedef HexahedraMoved_Adding   EMoved_Adding;
    /// This event is not used for this type of element
    class EMoved { };

    typedef HexahedronAncestorElem AncestorElem;
};

} // namespace sofa::core::topology


