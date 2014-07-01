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

#ifndef _TILING_HEXAGONAL_H_
#define _TILING_HEXAGONAL_H_

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Tilings
{

namespace Hexagonal
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

} // namespace Hexagonal

} // namespace Tilings

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Tiling/Surface/hexagonal.hpp"

#endif // _TILING_HEXAGONAL_H_
