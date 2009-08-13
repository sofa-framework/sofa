/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/component/collision/LocalMinDistanceFilter.inl>

#include <sofa/core/ObjectFactory.h>

#include <limits>


namespace sofa
{

namespace component
{

namespace collision
{


LocalMinDistanceFilter::LocalMinDistanceFilter()
    : m_coneExtension(initData(&m_coneExtension, 0.5, "coneExtension", "Filtering cone extension angle."))
    , m_coneMinAngle(initData(&m_coneMinAngle, 0.0, "coneMinAngle", "Minimal filtering cone angle value, independent from geometry."))
    , m_revision(0)
{

}



LocalMinDistanceFilter::~LocalMinDistanceFilter()
{

}



void LocalMinDistanceFilter::invalidate()
{
    m_revision = m_revision < std::numeric_limits< unsigned int >::max() ? m_revision++ : 0;
}


/*
template<>
bool LocalMinDistanceFilters::validate(const Point &p, const defaulttype::Vector3 &PQ)
{
	PointInfoMap::iterator it = m_pointInfoMap.find(p.getIndex());
	if (it != m_pointInfoMap.end())
	{
		return it->second->validate(p, PQ);
	}

	std::pair< PointInfoMap::iterator, bool > ret = m_pointInfoMap.insert(std::make_pair(p.getIndex(), new PointInfo(this)));

	return ret.first->second->validate(p, PQ);
}



template<>
bool LocalMinDistanceFilters::validate(const Line &l, const defaulttype::Vector3 &PQ)
{
	LineInfoMap::iterator it = m_lineInfoMap.find(l.getIndex());
	if (it != m_lineInfoMap.end())
	{
		return it->second->validate(l, PQ);
	}

	std::pair< LineInfoMap::iterator, bool > ret = m_lineInfoMap.insert(std::make_pair(l.getIndex(), new LineInfo(this)));

	return ret.first->second->validate(l, PQ);
}



template<>
bool LocalMinDistanceFilters::validate(const Triangle &t, const defaulttype::Vector3 &PQ)
{
	TriangleInfoMap::iterator it = m_triangleInfoMap.find(t.getIndex());
	if (it != m_triangleInfoMap.end())
	{
		return it->second->validate(t, PQ);
	}

	std::pair< TriangleInfoMap::iterator, bool > ret = m_triangleInfoMap.insert(std::make_pair(t.getIndex(), new TriangleInfo(this)));

	return ret.first->second->validate(t, PQ);
}
*/

} // namespace collision

} // namespace component

} // namespace sofa
