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
#include <iostream>

namespace sofa::core::topology
{

/// The enumeration used to give unique identifiers to TopologyChange objects.
enum TopologyChangeType
{
    BASE,                      ///< For TopologyChange class, should never be used.
    ENDING_EVENT,              ///< To notify the end for the current sequence of topological change events

    POINTSINDICESSWAP,         ///< For PointsIndicesSwap class.
    POINTSADDED,               ///< For PointsAdded class.
    POINTSREMOVED,             ///< For PointsRemoved class.
    POINTSMOVED,               ///< For PointsMoved class.
    POINTSRENUMBERING,         ///< For PointsRenumbering class.

    EDGESINDICESSWAP,          ///< For EdgesIndicesSwap class.
    EDGESADDED,                ///< For EdgesAdded class.
    EDGESREMOVED,              ///< For EdgesRemoved class.
    EDGESMOVED_REMOVING,       ///< For EdgesMoved class (event before changing state).
    EDGESMOVED_ADDING,         ///< For EdgesMoved class.
    EDGESRENUMBERING,          ///< For EdgesRenumbering class.

    TRIANGLESINDICESSWAP,      ///< For TrianglesIndicesSwap class.
    TRIANGLESADDED,            ///< For TrianglesAdded class.
    TRIANGLESREMOVED,          ///< For TrianglesRemoved class.
    TRIANGLESMOVED_REMOVING,   ///< For TrianglesMoved class (event before changing state).
    TRIANGLESMOVED_ADDING,     ///< For TrianglesMoved class.
    TRIANGLESRENUMBERING,      ///< For TrianglesRenumbering class.

    TETRAHEDRAINDICESSWAP,     ///< For TetrahedraIndicesSwap class.
    TETRAHEDRAADDED,           ///< For TetrahedraAdded class.
    TETRAHEDRAREMOVED,         ///< For TetrahedraRemoved class.
    TETRAHEDRAMOVED_REMOVING,  ///< For TetrahedraMoved class (event before changing state).
    TETRAHEDRAMOVED_ADDING,    ///< For TetrahedraMoved class.
    TETRAHEDRARENUMBERING,     ///< For TetrahedraRenumbering class.

    QUADSINDICESSWAP,          ///< For QuadsIndicesSwap class.
    QUADSADDED,                ///< For QuadsAdded class.
    QUADSREMOVED,              ///< For QuadsRemoved class.
    QUADSMOVED_REMOVING,       ///< For QuadsMoved class (event before changing state).
    QUADSMOVED_ADDING,         ///< For QuadsMoved class.
    QUADSRENUMBERING,          ///< For QuadsRenumbering class.

    HEXAHEDRAINDICESSWAP,      ///< For HexahedraIndicesSwap class.
    HEXAHEDRAADDED,            ///< For HexahedraAdded class.
    HEXAHEDRAREMOVED,          ///< For HexahedraRemoved class.
    HEXAHEDRAMOVED_REMOVING,   ///< For HexahedraMoved class (event before changing state).
    HEXAHEDRAMOVED_ADDING,     ///< For HexahedraMoved class.
    HEXAHEDRARENUMBERING,      ///< For HexahedraRenumbering class.

    TOPOLOGYCHANGE_LASTID      ///< user defined topology changes can start here
};

SOFA_CORE_API TopologyChangeType parseTopologyChangeTypeFromString(const std::string& s);
SOFA_CORE_API std::string parseTopologyChangeTypeToString(TopologyChangeType t);

/** \brief Base class to indicate a topology change occurred.
*
* All topological changes taking place in a given BaseTopology will issue a TopologyChange in the
* BaseTopology's changeList, so that BasicTopologies mapped to it can know what happened and decide how to
* react.
* Classes inheriting from this one describe a given topolopy change (e.g. RemovedPoint, AddedEdge, etc).
* The exact type of topology change is given by member changeType.
*/
class SOFA_CORE_API TopologyChange
{
public:
    /** \ brief Destructor.
        *
        * Must be virtual for TopologyChange to be a Polymorphic type.
        */
    virtual ~TopologyChange();

    /** \brief Returns the code of this TopologyChange. */
    TopologyChangeType getChangeType() const { return m_changeType;}

    virtual bool write(std::ostream& out) const;
    virtual bool read(std::istream& in);

    /// Output  stream
    SOFA_CORE_API friend std::ostream& operator<< ( std::ostream& out, const TopologyChange* t );

    /// Input (empty) stream
    SOFA_CORE_API friend std::istream& operator>> ( std::istream& in, TopologyChange*& t );

    /// Input (empty) stream
    SOFA_CORE_API friend std::istream& operator>> ( std::istream& in, const TopologyChange*& );

protected:
    TopologyChange( TopologyChangeType changeType = BASE )
        : m_changeType(changeType)
    {}

    TopologyChangeType m_changeType; ///< A code that tells the nature of the Topology modification event (could be an enum).
};

} // sofa::core::topology
