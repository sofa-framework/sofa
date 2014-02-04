/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

namespace CGoGN
{

namespace Geom
{

template <typename VEC>
BoundingBox<VEC>::BoundingBox() :
	m_initialized(false),
	m_pMin(0),
	m_pMax(0)
{}

template <typename VEC>
BoundingBox<VEC>::BoundingBox(const VEC& p) :
	m_initialized(true),
	m_pMin(p),
	m_pMax(p)
{}

template <typename VEC>
VEC& BoundingBox<VEC>::min()
{
	assert(m_initialized || !"Bounding box not initialized");
	return m_pMin ;
}

template <typename VEC>
const VEC& BoundingBox<VEC>::min() const
{
	assert(m_initialized || !"Bounding box not initialized");
	return m_pMin ;
}

template <typename VEC>
VEC& BoundingBox<VEC>::max()
{
	assert(m_initialized || !"Bounding box not initialized");
	return m_pMax ;
}

template <typename VEC>
const VEC& BoundingBox<VEC>::max() const
{
	assert(m_initialized);
	return m_pMax ;
}

template <typename VEC>
typename VEC::DATA_TYPE BoundingBox<VEC>::size(unsigned int coord) const
{
	assert(m_initialized && coord < m_pMax.dimension()) ;
	return m_pMax[coord] - m_pMin[coord] ;
}

template <typename VEC>
typename VEC::DATA_TYPE BoundingBox<VEC>::maxSize() const
{
	assert(m_initialized || !"Bounding box not initialized");

	typename VEC::DATA_TYPE max = m_pMax[0] - m_pMin[0] ;
	for(unsigned int i = 1; i < m_pMax.dimension(); ++i)
	{
		typename VEC::DATA_TYPE size = m_pMax[i] - m_pMin[i] ;
		if(size > max)
			max = size ;
	}
	return max ;
}

template <typename VEC>
typename VEC::DATA_TYPE BoundingBox<VEC>::minSize() const
{
	assert(m_initialized || !"Bounding box not initialized");

	typename VEC::DATA_TYPE min = m_pMax[0] - m_pMin[0] ;
	for(unsigned int i = 1; i < m_pMax.dimension(); ++i)
	{
		typename VEC::DATA_TYPE size = m_pMax[i] - m_pMin[i] ;
		if(size < min)
			min = size ;
	}
	return min ;
}

template <typename VEC>
VEC BoundingBox<VEC>::diag() const
{
	assert(m_initialized || !"Bounding box not initialized");
	return m_pMax - m_pMin ;
}

template <typename VEC>
typename VEC::DATA_TYPE BoundingBox<VEC>::diagSize() const
{
	assert(m_initialized || !"Bounding box not initialized");
	return (m_pMax - m_pMin).norm() ;
}

template <typename VEC>
VEC BoundingBox<VEC>::center() const
{
	assert(m_initialized || !"Bounding box not initialized");
	VEC center = (m_pMax + m_pMin) / typename VEC::DATA_TYPE(2) ;
	return center ;
}

template <typename VEC>
bool BoundingBox<VEC>::isInitialized() const
{
	return m_initialized;
}

/**********************************************/
/*                 FUNCTIONS                  */
/**********************************************/

template <typename VEC>
void BoundingBox<VEC>::reset()
{
	m_initialized = false;
}

template <typename VEC>
void BoundingBox<VEC>::addPoint(const VEC& p)
{
	if(!m_initialized)
	{
		m_pMin = p ;
		m_pMax = p ;
		m_initialized = true ;
	}
	else
	{
		for(unsigned int i = 0; i < p.dimension(); ++i)
		{
			if(p[i] < m_pMin[i])
				m_pMin[i] = p[i] ;
			if(p[i] > m_pMax[i])
				m_pMax[i] = p[i] ;
		}
	}
}

template <typename VEC>
bool BoundingBox<VEC>::intersects(const BoundingBox<VEC>& bb)
{
	assert(m_initialized || !"Bounding box not initialized");
	VEC bbmin = bb.min() ;
	VEC bbmax = bb.max() ;
	for(unsigned int i = 0; i < bbmin.dimension(); ++i)
	{
		if(m_pMax[i] < bbmin[i])
			return false ;
		if(m_pMin[i] > bbmax[i])
			return false ;
	}
	return true ;
}

template <typename VEC>
void BoundingBox<VEC>::fusion(const BoundingBox<VEC>& bb)
{
	assert(m_initialized || !"Bounding box not initialized");
	VEC bbmin = bb.min() ;
	VEC bbmax = bb.max() ;
	for(unsigned int i = 0; i < bbmin.dimension(); ++i)
	{
		if(bbmin[i] < m_pMin[i])
			m_pMin[i] = bbmin[i] ;
		if(bbmax[i] > m_pMax[i])
			m_pMax[i] = bbmax[i] ;
	}
}

template <typename VEC>
bool BoundingBox<VEC>::contains(const VEC& p)
{
	assert(m_initialized || !"Bounding box not initialized");
	for(unsigned int i = 0; i < m_pMin.dimension(); ++i)
	{
		if(m_pMin[i] > p[i])
			return false ;
		if(p[i] > m_pMax[i])
			return false ;
	}

	return true;
}

template <typename VEC>
bool BoundingBox<VEC>::contains(const VEC& a, const VEC& b)
{
	assert(m_initialized || !"Bounding box not initialized");

#define LEFT 'l'
#define RIGHT 'r'
#define MIDDLE 'm'

	//Algorithm from Graphic Gems
	//modified to test segment
	VEC dir(b-a);		/*ray */

	bool inside = true;
	char quadrant[m_pMin.dimension()];

	VEC candidatePlane;

	/* Find candidate planes; this loop can be avoided if
	rays cast all from the eye(assume perpsective view) */
	for (unsigned int i=0; i< m_pMin.dimension(); i++)
	{
		if(a[i] < m_pMin[i])
		{
			quadrant[i] = LEFT;
			candidatePlane[i] = m_pMin[i];
			inside = false;
		}
		else if (a[i] > m_pMax[i])
		{
			quadrant[i] = RIGHT;
			candidatePlane[i] = m_pMax[i];
			inside = false;
		}
		else
		{
			quadrant[i] = MIDDLE;
		}
	}

	/* Ray origin inside bounding box */
	if(inside)
		return true;

	VEC maxT;
	VEC coord;			/* hit point */
	/* Calculate T distances to candidate planes */
	for(unsigned int i = 0; i < m_pMin.dimension(); i++)
	{
		if (quadrant[i] != MIDDLE && dir[i] !=0)
			maxT[i] = (candidatePlane[i]-a[i]) / dir[i];
		else
			maxT[i] = -1.;
	}

#undef LEFT
#undef RIGHT
#undef MIDDLE

	/* Get largest of the maxT's for final choice of intersection */
	unsigned int whichPlane = 0;
	for(unsigned int i = 1; i < m_pMin.dimension(); i++)
		if (maxT[whichPlane] < maxT[i])
			whichPlane = i;

	/* Check final candidate actually inside box */
	if (maxT[whichPlane] < 0.)
		return false;
	for (unsigned int i = 0; i < m_pMin.dimension(); i++)
	{
		if (whichPlane != i)
		{
			coord[i] = a[i] + maxT[whichPlane] *dir[i];
			if (coord[i] < m_pMin[i] || coord[i] > m_pMax[i])
				return false;
			else
				coord[i] = candidatePlane[i];
		}
	}

	return VEC(coord-b)*VEC(a-b); /* intersection in segment ?*/
}

template <typename VEC>
bool BoundingBox<VEC>::contains(const BoundingBox<VEC>& bb)
{
	assert(m_initialized || !"Bounding box not initialized");
	return this->contains(bb.min()) && this->contains(bb.max());
}

template <typename VEC>
void BoundingBox<VEC>::scale(typename VEC::DATA_TYPE size)
{
	assert(m_initialized || !"Bounding box not initialized");
	m_pMin *= size ;
	m_pMax *= size ;
}

template <typename VEC>
void BoundingBox<VEC>::centeredScale(typename VEC::DATA_TYPE size)
{
	VEC center = (m_pMin + m_pMax) / typename VEC::DATA_TYPE(2) ;
	m_pMin = ((m_pMin - center) * size) + center ;
	m_pMax = ((m_pMax - center) * size) + center ;
}



/// test if bb is intersected by a ray
template <typename VEC>
bool BoundingBox<VEC>::rayIntersect(const VEC& P, const VEC& V) const
{
	static float EPSILON = 0.000001f;

	if (fabs(V[2]) > EPSILON)
	{
		VEC Q = P + ((m_pMin[2] - P[2])/V[2])*V;
		if ((Q[0]<m_pMax[0]) && (Q[0]>m_pMin[0]) && (Q[1]<m_pMax[1]) && (Q[1]>m_pMin[1]))
			return true;
		Q = P + ((m_pMax[2] - P[2])/V[2])*V;
				if ((Q[0]<m_pMax[0]) && (Q[0]>m_pMin[0]) && (Q[1]<m_pMax[1]) && (Q[1]>m_pMin[1]))
					return true;
	}

	if (fabs(V[1]) > EPSILON)
	{
		VEC Q = P + ((m_pMin[1] - P[1])/V[1])*V;
		if ((Q[0]<m_pMax[0]) && (Q[0]>m_pMin[0]) && (Q[2]<m_pMax[2]) && (Q[2]>m_pMin[2]))
			return true;
		Q = P + ((m_pMax[1] - P[1])/V[1])*V;
				if ((Q[0]<m_pMax[0]) && (Q[0]>m_pMin[0]) && (Q[2]<m_pMax[2]) && (Q[2]>m_pMin[2]))
					return true;
	}


	if (fabs(V[0]) > EPSILON)
	{
		VEC Q = P + ((m_pMin[0] - P[0])/V[0])*V;
		if ((Q[1]<m_pMax[1]) && (Q[1]>m_pMin[1]) && (Q[2]<m_pMax[2]) && (Q[2]>m_pMin[2]))
			return true;
		Q = P + ((m_pMax[0] - P[0])/V[0])*V;
				if ((Q[1]<m_pMax[1]) && (Q[1]>m_pMin[1]) && (Q[2]<m_pMax[2]) && (Q[2]>m_pMin[2]))
					return true;
	}

	return false;
}




} // namespace Geom

} // namespace CGoGN
