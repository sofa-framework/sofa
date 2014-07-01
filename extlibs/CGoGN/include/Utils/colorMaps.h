/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009, IGG Team, LSIIT, University of Strasbourg                *
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
* Web site: http://cgogn.unistra.fr/                                  *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef __COLOR_MAPS_H__
#define __COLOR_MAPS_H__

namespace CGoGN
{

namespace Utils
{

/**
 *
 * @param x
 */
Geom::Vec3f color_map_blue_green_red(float x);

/**
 *
 * @param x
 */
Geom::Vec3f color_map_BCGYR(float x);

/**
 *
 * @param x
 */
Geom::Vec3f color_map_blue_white_red(float x);

/**
 *
 * @param x
 */
Geom::Vec3f color_map_cyan_white_red(float x);

/**
 * Create a table of color using function (param between 0 & 1)
 * @param table table of color to fill
 * @param nb size of table
 * @param f the function (color_map_xxxx)
 */
template <typename FUNC>
void createTableColor01(std::vector<Geom::Vec3f>& table, unsigned int nb, FUNC f);


/**
 *
 * @param x
 * @param n
 */
float scale_expand_within_0_1(float x, int n);

/**
 *
 * @param x
 * @param n
 */
float scale_expand_towards_1(float x, int n);

/**
 *
 * @param min
 * @param max
 */
float scale_to_0_1(float x, float min, float max);

/**
 *
 * @param x
 * @param min
 * @param max
 */
float scale_and_clamp_to_0_1(float x, float min, float max);

/**
 *
 * @param min
 * @param max
 */
void scale_centering_around_0(float& min, float& max);

/**
 *
 * @param x
 * @param min
 * @param max
 */
float scale_to_0_1_around_one_half(float x, float min, float max);

} // namespace Utils

} // namespace CGoGN

#include "Utils/colorMaps.hpp"

#endif

