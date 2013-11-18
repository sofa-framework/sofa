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

#ifndef _VBO_VECTOR_ATTRIBUTE_RENDER_
#define _VBO_VECTOR_ATTRIBUTE_RENDER_

#include <GL/gl.h>
#include <vector>

#include "Topology/generic/functor.h"
#include "Container/convert.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace VBO
{

template <typename PFP>
class VectorAttributeRender
{
protected:
	typedef typename PFP::MAP MAP ;

	/**
	* the map to draw
	*/
	MAP& m_map ;

	/**
	* darts selector
	*/
	FunctorType& m_good ;

	/**
	 * vbo buffer
	 */
	GLuint m_VBOBuffer ;

	/**
	* number of indices of lines
	*/
	GLuint m_nbIndices ;

	/**
	* creation of indices table of lines
	* @param tableIndices the table where indices are stored
	*/
	void initLines(std::vector<GLuint>& tableIndices) ;

public:
	/**
	* Constructor
	* @param map the map to draw
	* @param good functor that return true for darts of part to draw
	*/
	VectorAttributeRender(MAP& map, FunctorType& good) ;

	/**
	* Destructor
	*/
	~VectorAttributeRender() ;

	/**
	 * update the data
	 * @param uptype what have ot be update: POSITIONS, NORMALS, COLORS, TEXCOORDS, ???
	 * @param attribId id of attribute where data are stored
	 * @param conv Callback of attribute conversion (NULL if direct copy, default value)
	 */
	void updateData(int upType, unsigned int attribId, ConvertAttrib* conv = NULL);

	/**
	* draw the VBO (function to call in the drawing callback)
	*/
	void draw() ;
} ;

} // namespace VBO

} // namespace Render

} // namespace Algo

} // namespace CGoGN

#include "Algo/Render/vbo_VectorAttributeRender.hpp"

#endif
