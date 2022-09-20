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

namespace sofa::core::topology
{

// forward declarations
class SOFA_CORE_API TopologyChange;
class SOFA_CORE_API EndingEvent;
class SOFA_CORE_API PointsIndicesSwap;
class SOFA_CORE_API PointsAdded;
class SOFA_CORE_API PointsRemoved;
class SOFA_CORE_API PointsMoved;
class SOFA_CORE_API PointsRenumbering;
class SOFA_CORE_API EdgesIndicesSwap;
class SOFA_CORE_API EdgesAdded;
class SOFA_CORE_API EdgesRemoved;
class SOFA_CORE_API EdgesMoved_Removing;
class SOFA_CORE_API EdgesMoved_Adding;
class SOFA_CORE_API EdgesRenumbering;
class SOFA_CORE_API TrianglesIndicesSwap;
class SOFA_CORE_API TrianglesAdded;
class SOFA_CORE_API TrianglesRemoved;
class SOFA_CORE_API TrianglesMoved_Removing;
class SOFA_CORE_API TrianglesMoved_Adding;
class SOFA_CORE_API TrianglesRenumbering;
class SOFA_CORE_API TetrahedraIndicesSwap;
class SOFA_CORE_API TetrahedraAdded;
class SOFA_CORE_API TetrahedraRemoved;
class SOFA_CORE_API TetrahedraMoved_Removing;
class SOFA_CORE_API TetrahedraMoved_Adding;
class SOFA_CORE_API TetrahedraRenumbering;
class SOFA_CORE_API QuadsIndicesSwap;
class SOFA_CORE_API QuadsAdded;
class SOFA_CORE_API QuadsRemoved;
class SOFA_CORE_API QuadsMoved_Removing;
class SOFA_CORE_API QuadsMoved_Adding;
class SOFA_CORE_API QuadsRenumbering;
class SOFA_CORE_API HexahedraIndicesSwap;
class SOFA_CORE_API HexahedraAdded;
class SOFA_CORE_API HexahedraRemoved;
class SOFA_CORE_API HexahedraMoved_Removing;
class SOFA_CORE_API HexahedraMoved_Adding;
class SOFA_CORE_API HexahedraRenumbering;

} // sofa::core::topology
