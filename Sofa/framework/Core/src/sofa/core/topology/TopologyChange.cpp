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
#include <sofa/core/topology/TopologyElement.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/list.h>

namespace std
{
    template class SOFA_CORE_API list<const sofa::core::topology::TopologyChange*>;
}

namespace sofa::core::topology
{

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

} // namespace sofa::core::topology
