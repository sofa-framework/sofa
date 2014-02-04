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

#ifndef __BOUNDING_BOX__
#define __BOUNDING_BOX__

namespace CGoGN
{

namespace Geom
{

/*
 * Class for the computation of bounding boxes
 */
template <typename VEC>
class BoundingBox
{
	public:
		/**********************************************/
		/*                CONSTRUCTORS                */
		/**********************************************/

		BoundingBox() ;

		// initialize the bounding box with one first point
		BoundingBox(const VEC& p) ;

		/**********************************************/
		/*                 ACCESSORS                  */
		/**********************************************/

		VEC& min() ;

		const VEC& min() const ;

		VEC& max() ;

		const VEC& max() const ;

		typename VEC::DATA_TYPE size(unsigned int coord) const ;

		typename VEC::DATA_TYPE maxSize() const ;

		typename VEC::DATA_TYPE minSize() const ;

		VEC diag() const ;

		typename VEC::DATA_TYPE diagSize() const ;

		VEC center() const ;

		bool isInitialized() const ;

		/**********************************************/
		/*                 FUNCTIONS                  */
		/**********************************************/

		// reinitialize the bounding box
		void reset() ;

		// add a point to the bounding box
		void addPoint(const VEC& p) ;

		// return true if bb intersects the bounding box
		bool intersects(const BoundingBox<VEC>& bb) ;

		// fusion with the given bounding box
		void fusion(const BoundingBox<VEC>& bb) ;

		// return true if the point belongs strictly to a bounding box
		bool contains(const VEC& p);

		// return true if the segment belongs strictly to a bounding box
		bool contains(const VEC& a, const VEC& b);

		// return true if the bounding box belongs strictly to a bounding box
		bool contains(const BoundingBox<VEC> & bb);

		// scale the bounding box
		void scale(typename VEC::DATA_TYPE size);

		// 0-centered scale of the bounding box
		void centeredScale(typename VEC::DATA_TYPE size);

		/// test if bb is intersected by a ray
		bool rayIntersect(const VEC& P, const VEC& V) const;

		/**********************************************/
		/*             STREAM OPERATORS               */
		/**********************************************/

		friend std::ostream& operator<<(std::ostream& out, const BoundingBox<VEC>& bb)
		{
			out << bb.min() << " " << bb.max() ;
			return out ;
		}

		friend std::istream& operator>>(std::istream& in, BoundingBox<VEC>& bb)
		{
			in >> bb.min() >> bb.max() ;
			return in ;
		}

	private:
		bool m_initialized ;
		VEC m_pMin, m_pMax ;
} ;

} // namespace Geom

} // namespace CGoGN

#include "Geometry/bounding_box.hpp"
#endif
