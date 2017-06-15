/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <sofa/core/topology/TopologyChange.h>

namespace sofa
{

namespace core
{

namespace topology
{

SOFA_CORE_API TopologyObjectType parseTopologyObjectTypeFromString(const std::string& s)
{
    std::string sUP = s;
    std::transform( sUP.begin(), sUP.end(), sUP.begin(), ::tolower);
#define SOFA_ENUM_CASE(e) if (sUP == sofa_tostring(e)) return e
    SOFA_ENUM_CASE(POINT);
    SOFA_ENUM_CASE(EDGE);
    SOFA_ENUM_CASE(TRIANGLE);
    SOFA_ENUM_CASE(QUAD);
    SOFA_ENUM_CASE(TETRAHEDRON);
    SOFA_ENUM_CASE(HEXAHEDRON);
#undef SOFA_ENUM_CASE
    msg_error("TopologyObjectType")
            << "unable to parse '" << s << "' as TopologyObjectType, defaulting to POINT" ;
    return POINT;
}

SOFA_CORE_API std::string parseTopologyObjectTypeToString(TopologyObjectType t)
{
    switch (t)
    {
#define SOFA_ENUM_CASE(e) case e: return sofa_tostring(e);
    SOFA_ENUM_CASE(POINT);
    SOFA_ENUM_CASE(EDGE);
    SOFA_ENUM_CASE(TRIANGLE);
    SOFA_ENUM_CASE(QUAD);
    SOFA_ENUM_CASE(TETRAHEDRON);
    SOFA_ENUM_CASE(HEXAHEDRON);
#undef SOFA_ENUM_CASE
    default: return std::string("UNKNOWN");
    }
}
/*
SOFA_CORE_API std::ostream& operator << (std::ostream& out, const TopologyObjectType& d)
{
    return out << parseTopologyObjectTypeToString(d);
}

SOFA_CORE_API std::istream& operator >> (std::istream& in, TopologyObjectType& d)
{
    std::string s;
    in >> s;
    d = parseTopologyObjectTypeFromString(s);
    return in;
}
*/
SOFA_CORE_API TopologyChangeType parseTopologyChangeTypeFromString(const std::string& s)
{
    std::string sUP = s;
    std::transform( sUP.begin(), sUP.end(), sUP.begin(), ::tolower);
#define SOFA_ENUM_CASE(e) if (sUP == sofa_tostring(e)) return e
    SOFA_ENUM_CASE(BASE);
    SOFA_ENUM_CASE(ENDING_EVENT);

    SOFA_ENUM_CASE(POINTSINDICESSWAP);
    SOFA_ENUM_CASE(POINTSADDED);
    SOFA_ENUM_CASE(POINTSREMOVED);
    SOFA_ENUM_CASE(POINTSMOVED);
    SOFA_ENUM_CASE(POINTSRENUMBERING);

    SOFA_ENUM_CASE(EDGESINDICESSWAP);
    SOFA_ENUM_CASE(EDGESADDED);
    SOFA_ENUM_CASE(EDGESREMOVED);
    SOFA_ENUM_CASE(EDGESMOVED_REMOVING);
    SOFA_ENUM_CASE(EDGESMOVED_ADDING);
    SOFA_ENUM_CASE(EDGESRENUMBERING);

    SOFA_ENUM_CASE(TRIANGLESINDICESSWAP);
    SOFA_ENUM_CASE(TRIANGLESADDED);
    SOFA_ENUM_CASE(TRIANGLESREMOVED);
    SOFA_ENUM_CASE(TRIANGLESMOVED_REMOVING);
    SOFA_ENUM_CASE(TRIANGLESMOVED_ADDING);
    SOFA_ENUM_CASE(TRIANGLESRENUMBERING);

    SOFA_ENUM_CASE(TETRAHEDRAINDICESSWAP);
    SOFA_ENUM_CASE(TETRAHEDRAADDED);
    SOFA_ENUM_CASE(TETRAHEDRAREMOVED);
    SOFA_ENUM_CASE(TETRAHEDRAMOVED_REMOVING);
    SOFA_ENUM_CASE(TETRAHEDRAMOVED_ADDING);
    SOFA_ENUM_CASE(TETRAHEDRARENUMBERING);

    SOFA_ENUM_CASE(QUADSINDICESSWAP);
    SOFA_ENUM_CASE(QUADSADDED);
    SOFA_ENUM_CASE(QUADSREMOVED);
    SOFA_ENUM_CASE(QUADSMOVED_REMOVING);
    SOFA_ENUM_CASE(QUADSMOVED_ADDING);
    SOFA_ENUM_CASE(QUADSRENUMBERING);

    SOFA_ENUM_CASE(HEXAHEDRAINDICESSWAP);
    SOFA_ENUM_CASE(HEXAHEDRAADDED);
    SOFA_ENUM_CASE(HEXAHEDRAREMOVED);
    SOFA_ENUM_CASE(HEXAHEDRAMOVED_REMOVING);
    SOFA_ENUM_CASE(HEXAHEDRAMOVED_ADDING);
    SOFA_ENUM_CASE(HEXAHEDRARENUMBERING);
    SOFA_ENUM_CASE(TOPOLOGYCHANGE_LASTID);
#undef SOFA_ENUM_CASE
    msg_warning("TopologyChange") << "problem while parsing '" << s << "' as TopologyChangeType, defaulting to TOPOLOGYCHANGE_LASTID" ;
    return TOPOLOGYCHANGE_LASTID;
}

SOFA_CORE_API std::string parseTopologyChangeTypeToString(TopologyChangeType t)
{
    switch (t)
    {
#define SOFA_ENUM_CASE(e) case e: return sofa_tostring(e)
    SOFA_ENUM_CASE(BASE);
    SOFA_ENUM_CASE(ENDING_EVENT);

    SOFA_ENUM_CASE(POINTSINDICESSWAP);
    SOFA_ENUM_CASE(POINTSADDED);
    SOFA_ENUM_CASE(POINTSREMOVED);
    SOFA_ENUM_CASE(POINTSMOVED);
    SOFA_ENUM_CASE(POINTSRENUMBERING);

    SOFA_ENUM_CASE(EDGESINDICESSWAP);
    SOFA_ENUM_CASE(EDGESADDED);
    SOFA_ENUM_CASE(EDGESREMOVED);
    SOFA_ENUM_CASE(EDGESMOVED_REMOVING);
    SOFA_ENUM_CASE(EDGESMOVED_ADDING);
    SOFA_ENUM_CASE(EDGESRENUMBERING);

    SOFA_ENUM_CASE(TRIANGLESINDICESSWAP);
    SOFA_ENUM_CASE(TRIANGLESADDED);
    SOFA_ENUM_CASE(TRIANGLESREMOVED);
    SOFA_ENUM_CASE(TRIANGLESMOVED_REMOVING);
    SOFA_ENUM_CASE(TRIANGLESMOVED_ADDING);
    SOFA_ENUM_CASE(TRIANGLESRENUMBERING);

    SOFA_ENUM_CASE(TETRAHEDRAINDICESSWAP);
    SOFA_ENUM_CASE(TETRAHEDRAADDED);
    SOFA_ENUM_CASE(TETRAHEDRAREMOVED);
    SOFA_ENUM_CASE(TETRAHEDRAMOVED_REMOVING);
    SOFA_ENUM_CASE(TETRAHEDRAMOVED_ADDING);
    SOFA_ENUM_CASE(TETRAHEDRARENUMBERING);

    SOFA_ENUM_CASE(QUADSINDICESSWAP);
    SOFA_ENUM_CASE(QUADSADDED);
    SOFA_ENUM_CASE(QUADSREMOVED);
    SOFA_ENUM_CASE(QUADSMOVED_REMOVING);
    SOFA_ENUM_CASE(QUADSMOVED_ADDING);
    SOFA_ENUM_CASE(QUADSRENUMBERING);

    SOFA_ENUM_CASE(HEXAHEDRAINDICESSWAP);
    SOFA_ENUM_CASE(HEXAHEDRAADDED);
    SOFA_ENUM_CASE(HEXAHEDRAREMOVED);
    SOFA_ENUM_CASE(HEXAHEDRAMOVED_REMOVING);
    SOFA_ENUM_CASE(HEXAHEDRAMOVED_ADDING);
    SOFA_ENUM_CASE(HEXAHEDRARENUMBERING);
    SOFA_ENUM_CASE(TOPOLOGYCHANGE_LASTID);
#undef SOFA_ENUM_CASE
    default: return std::string("UNKNOWN");
    }
}
/*
SOFA_CORE_API std::ostream& operator << (std::ostream& out, const TopologyChangeType& d)
{
    return out << parseTopologyChangeTypeToString(d);
}

SOFA_CORE_API std::istream& operator >> (std::istream& in, TopologyChangeType& d)
{
    std::string s;
    in >> s;
    d = parseTopologyChangeTypeFromString(s);
    return out;
}
*/

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const TopologyElemID& d)
{
    out << parseTopologyObjectTypeToString(d.type) << " " << d.index;
    return out;
}

SOFA_CORE_API std::istream& operator >> (std::istream& in, TopologyElemID& /* d */)
{/*
    std::string tstr;
    in >> tstr;
    d.type = parseTopologyObjectTypeFromString(tstr);
    in >> d.index;*/
    return in;
}

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const PointAncestorElem& d)
{
    out << parseTopologyObjectTypeToString(d.type) << " " << d.index << " " << d.localCoords;
    return out;
}

SOFA_CORE_API std::istream& operator >> (std::istream& in, PointAncestorElem& /* d */)
{/*
    std::string tstr;
    in >> tstr;
    d.type = parseTopologyObjectTypeFromString(tstr);
    in >> d.index;
    in >> d.localCoords;*/
    return in;
}

template<int NV>
SOFA_CORE_API std::ostream& operator << (std::ostream& out, const ElemAncestorElem<NV>& d)
{
    out << d.pointSrcElems << " " << d.srcElems.size() << " " << d.srcElems << "\n";
    return out;
}

template<int NV>
SOFA_CORE_API std::istream& operator >> (std::istream& in, ElemAncestorElem<NV>& /*d*/)
{
    /*
    in >> d.pointSrcElems;
    int nsrc = 0;
    in >> nsrc;
    if (nsrc > 0)
    {
        d.srcElems.resize(nsrc);
        for (unsigned int i = 0; i < nsrc; ++i)
            in >> d.srcElems[i];
    }
    */
    return in;
}

template SOFA_CORE_API std::ostream& operator<< (std::ostream& out, const ElemAncestorElem<2>& d);
template SOFA_CORE_API std::ostream& operator<< (std::ostream& out, const ElemAncestorElem<3>& d);
template SOFA_CORE_API std::ostream& operator<< (std::ostream& out, const ElemAncestorElem<4>& d);
template SOFA_CORE_API std::ostream& operator<< (std::ostream& out, const ElemAncestorElem<8>& d);

TopologyChange::~TopologyChange()
{
}

bool TopologyChange::write(std::ostream& out) const
{
    out << parseTopologyChangeTypeToString(getChangeType());
    return true;
}
bool TopologyChange::read(std::istream& /* in */)
{
    return false;
}

EndingEvent::~EndingEvent()
{
}

PointsIndicesSwap::~PointsIndicesSwap()
{
}

PointsAdded::~PointsAdded()
{
}

PointsRemoved::~PointsRemoved()
{
}

PointsRenumbering::~PointsRenumbering()
{
}

PointsMoved::~PointsMoved()
{
}

EdgesIndicesSwap::~EdgesIndicesSwap()
{
}

EdgesAdded::~EdgesAdded()
{
}

EdgesRemoved::~EdgesRemoved()
{
}

EdgesMoved_Removing::~EdgesMoved_Removing()
{
}

EdgesMoved_Adding::~EdgesMoved_Adding()
{
}

EdgesRenumbering::~EdgesRenumbering()
{
}

TrianglesIndicesSwap::~TrianglesIndicesSwap()
{
}

TrianglesAdded::~TrianglesAdded()
{
}

TrianglesRemoved::~TrianglesRemoved()
{
}

TrianglesMoved_Removing::~TrianglesMoved_Removing()
{
}

TrianglesMoved_Adding::~TrianglesMoved_Adding()
{
}

TrianglesRenumbering::~TrianglesRenumbering()
{
}

QuadsIndicesSwap::~QuadsIndicesSwap()
{
}

QuadsAdded::~QuadsAdded()
{
}

QuadsRemoved::~QuadsRemoved()
{
}

QuadsMoved_Removing::~QuadsMoved_Removing()
{
}

QuadsMoved_Adding::~QuadsMoved_Adding()
{
}

QuadsRenumbering::~QuadsRenumbering()
{
}

TetrahedraIndicesSwap::~TetrahedraIndicesSwap()
{
}

TetrahedraAdded::~TetrahedraAdded()
{
}

TetrahedraRemoved::~TetrahedraRemoved()
{
}

TetrahedraMoved_Removing::~TetrahedraMoved_Removing()
{
}

TetrahedraMoved_Adding::~TetrahedraMoved_Adding()
{
}

TetrahedraRenumbering::~TetrahedraRenumbering()
{
}

HexahedraIndicesSwap::~HexahedraIndicesSwap()
{
}

HexahedraAdded::~HexahedraAdded()
{
}

HexahedraRemoved::~HexahedraRemoved()
{
}

HexahedraMoved_Removing::~HexahedraMoved_Removing()
{
}

HexahedraMoved_Adding::~HexahedraMoved_Adding()
{
}

HexahedraRenumbering::~HexahedraRenumbering()
{
}

} // namespace topology

} // namespace core

} // namespace sofa
