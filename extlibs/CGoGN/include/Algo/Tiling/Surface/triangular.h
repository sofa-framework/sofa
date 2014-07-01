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

#include "Algo/Tiling/tiling.h"

#ifndef _TILING_TRIANGULAR_H_
#define _TILING_TRIANGULAR_H_

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Tilings
{

namespace Triangular
{

/*! \brief The class of regular grid square tiling
 */
template <typename PFP>
class Grid : public Tiling<PFP>
{
	typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;

public:
    Grid(MAP& map, unsigned int x, unsigned int y, bool close):
		Tiling<PFP>(map, x, y, -1)
    {
		grid(x, y, close);
    }

    Grid(MAP& map, unsigned int x, unsigned int y):
		Tiling<PFP>(map, x, y, -1)
	{
		grid(x, y, true);
	}

    /*! @name Embedding Operators
     * Tiling creation
     *************************************************************************/

    //@{
    //! Embed a topological grid
    /*! @param position Attribute used to store vertices positions
     *  @param x size in X
     *  @param x size in Y
     *  @param y position in Z (centered on 0 by default)
     */
	void embedIntoGrid(VertexAttribute<VEC3, MAP>& position, float x, float y, float z = 0.0f);

    //! Embed a topological grid into a twister open ribbon with turns=PI it is a Moebius strip, needs only to be closed (if model allow it)
    /*! @param position Attribute used to store vertices positions
     *  @param radius_min
     *  @param radius_max
     *  @param turns number of turn multiplied by 2*PI
     */
	void embedIntoTwistedStrip(VertexAttribute<VEC3, MAP>& position, float radius_min, float radius_max, float turns);

    //! Embed a topological grid into a helicoid
    /*! @param position Attribute used to store vertices positions
     *  @param radius_min
     *  @param radius_max
     *  @param maxHeight height to reach
     *  @param turns number of turn
     */
	void embedIntoHelicoid(VertexAttribute<VEC3, MAP>& position, float radius_min,  float radius_max, float maxHeight, float nbTurn, int orient = 1);
    //@}

protected:
    /*! @name Topological Operators
     * Tiling creation
     *************************************************************************/

    //@{
    //! Create a 2D grid
    /*! @param x nb of squares in x
     *  @param y nb of squares in y
     *  @param closed close the boundary face of the 2D grid
     */
    void grid(unsigned int x, unsigned int y, bool close);
    //@}

};

/*! \brief The class of regular cylinder square tiling or subdivided sphere (with pole)
 */
template <typename PFP>
class Cylinder : public Tiling<PFP>
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

private:
	bool m_top_closed, m_bottom_closed;
	bool m_top_triangulated, m_bottom_triangulated;
	Dart m_topVertDart, m_bottomVertDart;

public:
	Cylinder(MAP& map, unsigned int n, unsigned int z, bool close_top, bool close_bottom):
		Tiling<PFP>(map, n, -1, z),
		m_top_closed(close_top),
		m_bottom_closed(close_bottom),
		m_top_triangulated(false),
		m_bottom_triangulated(false)
	{
		cylinder(n,z);
	}

	Cylinder(MAP& map, unsigned int n, unsigned int z):
	  Tiling<PFP>(map, n, -1, z),
	  m_top_closed(true),
	  m_bottom_closed(true),
	  m_top_triangulated(false),
	  m_bottom_triangulated(false)
	{
		cylinder(n,z);
	}

	/*! @name Embedding Operators
	 * Tiling creation
	 *************************************************************************/

	//@{
	//! Embed a topological cylinder
	/*! @param position Attribute used to store vertices positions
	 *  @param bottom_radius
	 *  @param top_radius
	 *  @param height
	 */
	void embedIntoCylinder(VertexAttribute<VEC3, MAP>& position, float bottom_radius, float top_radius, float height);

	//! Embed a topological sphere
	/*! @param position Attribute used to store vertices positions
	 *  @param radius
	 *  @param height
	 */
	void embedIntoSphere(VertexAttribute<VEC3, MAP>& position, float radius);

	//! Embed a topological cone
	/*! @param position Attribute used to store vertices positions
	 *  @param radius
	 *  @param height
	 */
	void embedIntoCone(VertexAttribute<VEC3, MAP>& position, float radius, float height);
	//@}

	/*! @name Topological Operators
	 * Tiling creation
	 *************************************************************************/

	//@{
	//! Close the top with a n-sided face
	void closeTop();

	//! Triangulate the top face with triangles fan
	void triangleTop();

	//! Close the bottom with a n-sided face
	void closeBottom();

	//! Triangulate the bottom face with triangles fan
	void triangleBottom();

protected:
	//! Create a subdivided 2D cylinder
	/*! @param n nb of squares around circumference
	 *  @param z nb of squares in height
	 *  @param top_closed close the top (with triangles fan)
	 *  @param bottom_closed close the bottom (with triangles fan)
	 */
	//square -> cylinder -> grid + finish sewing -> open or closed (closeHole) -> triangule face ?
	void cylinder(unsigned int n, unsigned int z);
	//@}
};

/*! \brief The class of regular cube square tiling
 */
template <typename PFP>
class Cube : public Cylinder<PFP>
{
	typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;

public:
    Cube(MAP& map, unsigned int x, unsigned int y, unsigned int z):
        Cylinder<PFP>(map,2*(x+y),z, false, false)
    {
        cube(x,y,z);
    }

    /*! @name Embedding Operators
     * Tiling creation
     *************************************************************************/

    //@{
    //! Embed a topological cube
    /*! @param position Attribute used to store vertices positions
     *  @param x
     *  @param y
     *  @param z
     */
	void embedIntoCube(VertexAttribute<VEC3, MAP>& position, float x, float y, float z);
    //@}

protected:
    /*! @name Topological Operators
     * Tiling creation
     *************************************************************************/

    //@{
    //! Create a subdivided 2D cube
    /*! @param x nb of squares in x
     *  @param y nb of squares in y
     *  @param z nb of squares un z
     */
    void cube(unsigned int x, unsigned int y, unsigned int z);
    //@}

};

/*! \brief The class of regular tore square tiling
 */
template <typename PFP>
class Tore : public Cylinder<PFP>
{
	typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;

public:
    Tore(MAP& map, unsigned int n, unsigned int m):
        Cylinder<PFP>(map,n,m)
    {
        tore(n,m);
    }

    /*! @name Embedding Operators
     * Tiling creation
     *************************************************************************/

    //@{
    //! Embed a topological tore
    /*! @param position Attribute used to store vertices positions
     *  @param big_radius
     *  @param small_radius
     */
	void embedIntoTore(VertexAttribute<VEC3, MAP>& position, float big_radius, float small_radius);
    //@}

    /*! @name Topological Operators
     * Tiling creation
     *************************************************************************/

    //@{
protected:
    //! Create a subdivided 2D tore
    /*! @param n nb of squares around big circumference
     *  @param m nb of squares around small circumference
     */
    //square -> tore -> cylinder + finish sewing
    void tore(unsigned int n, unsigned int m);
    //@}
};

} // namespace Triangular

} // namespace Tilings

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Tiling/Surface/triangular.hpp"

#endif //_TILING_TRIANGULAR_H_
