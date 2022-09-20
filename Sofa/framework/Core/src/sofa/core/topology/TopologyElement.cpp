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
#define SOFA_CORE_TOPOLOGY_TOPOLOGYCHANGE_DEFINITION

#include <sofa/core/topology/TopologyElement.h>
#include <sofa/core/topology/TopologyChange.h>

namespace std
{
    template class SOFA_CORE_API list<const sofa::core::topology::TopologyChange*>;
}


namespace sofa::core::topology
{


SOFA_CORE_API TopologyElementType parseTopologyElementTypeFromString(const std::string& s)
{
    std::string sUP = s;
    std::transform( sUP.begin(), sUP.end(), sUP.begin(), ::tolower);
#define SOFA_ENUM_CASE(e) if (sUP == sofa_tostring(e)) return e
    SOFA_ENUM_CASE(TopologyElementType::POINT);
    SOFA_ENUM_CASE(TopologyElementType::EDGE);
    SOFA_ENUM_CASE(TopologyElementType::TRIANGLE);
    SOFA_ENUM_CASE(TopologyElementType::QUAD);
    SOFA_ENUM_CASE(TopologyElementType::TETRAHEDRON);
    SOFA_ENUM_CASE(TopologyElementType::HEXAHEDRON);
#undef SOFA_ENUM_CASE
    msg_error("TopologyElementType")
            << "unable to parse '" << s << "' as TopologyElementType, defaulting to POINT" ;
    return TopologyElementType::POINT;
}

SOFA_CORE_API std::string parseTopologyElementTypeToString(TopologyElementType t)
{
    switch (t)
    {
#define SOFA_ENUM_CASE(e) case e: return sofa_tostring(e);
    SOFA_ENUM_CASE(TopologyElementType::POINT);
    SOFA_ENUM_CASE(TopologyElementType::EDGE);
    SOFA_ENUM_CASE(TopologyElementType::TRIANGLE);
    SOFA_ENUM_CASE(TopologyElementType::QUAD);
    SOFA_ENUM_CASE(TopologyElementType::TETRAHEDRON);
    SOFA_ENUM_CASE(TopologyElementType::HEXAHEDRON);
#undef SOFA_ENUM_CASE
    default: return std::string("UNKNOWN");
    }
}

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const TopologyElemID& d)
{
    out << parseTopologyElementTypeToString(d.type) << " " << d.index;
    return out;
}

SOFA_CORE_API std::istream& operator >> (std::istream& in, TopologyElemID& /* d */)
{
    return in;
}

SOFA_CORE_API std::ostream& operator << (std::ostream& out, const PointAncestorElem& d)
{
    out << parseTopologyElementTypeToString(d.type) << " " << d.index << " " << d.localCoords;
    return out;
}

SOFA_CORE_API std::istream& operator >> (std::istream& in, PointAncestorElem& /* d */)
{
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
    return in;
}

template SOFA_CORE_API std::ostream& operator<< (std::ostream& out, const ElemAncestorElem<2>& d);
template SOFA_CORE_API std::ostream& operator<< (std::ostream& out, const ElemAncestorElem<3>& d);
template SOFA_CORE_API std::ostream& operator<< (std::ostream& out, const ElemAncestorElem<4>& d);
template SOFA_CORE_API std::ostream& operator<< (std::ostream& out, const ElemAncestorElem<8>& d);

} // namespace sofa::core::topology
