/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaCombinatorialMaps/Core/CMTopologyChange.h>

namespace sofa
{

namespace core
{

namespace cm_topology
{

TopologyChange::~TopologyChange()
{
}

bool TopologyChange::write(std::ostream& out) const
{
//	out << parseTopologyChangeTypeToString(getChangeType()); // TODO
	return true;
}
bool TopologyChange::read(std::istream& /* in */)
{
	return false;
}

std::ostream& operator<< ( std::ostream& out, const core::cm_topology::TopologyChange* t )
{
	if (t)
	{
		t->write(out);
	}
	return out;
}

std::istream& operator>> ( std::istream& in, const core::cm_topology::TopologyChange*& )
{
	return in;
}

/// Input (empty) stream
std::istream& operator>> ( std::istream& in, core::cm_topology::TopologyChange*& t )
{
	if (t)
	{
		t->read(in);
	}
	return in;
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

FacesIndicesSwap::~FacesIndicesSwap()
{
}

FacesAdded::~FacesAdded()
{
}

FacesRemoved::~FacesRemoved()
{
}

FacesMoved_Removing::~FacesMoved_Removing()
{
}

FacesMoved_Adding::~FacesMoved_Adding()
{
}

FacesRenumbering::~FacesRenumbering()
{
}



VolumesIndicesSwap::~VolumesIndicesSwap()
{
}

VolumesAdded::~VolumesAdded()
{
}

VolumesRemoved::~VolumesRemoved()
{
}

VolumesMoved_Removing::~VolumesMoved_Removing()
{
}

VolumesMoved_Adding::~VolumesMoved_Adding()
{
}

VolumesRenumbering::~VolumesRenumbering()
{
}

} // namespace topology

} // namespace core

} // namespace sofa
