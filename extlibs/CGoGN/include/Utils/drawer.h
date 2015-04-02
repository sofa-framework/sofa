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

#ifndef _CGOGN_DRAWER_H_
#define _CGOGN_DRAWER_H_

#include <vector>
#include "Utils/gl_def.h"
#include "Geometry/vector_gen.h"
#include "Utils/Shaders/shaderColorPerVertex.h"

namespace CGoGN { namespace Utils { class VBO; } }
namespace CGoGN { namespace Utils { namespace SVG { class SVGOut ; } } }

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

/**
 * A class that allow easy drawing, allmost as easy as
 * display lists in old school OpenGL 1.1
 * Allow only position and color
 * Drawing (callList) efficient after data has been finalized (endList)
 *
 */
class CGoGN_UTILS_API Drawer
{
	struct PrimParam
	{
		unsigned int begin;
		GLenum mode;
		float width;
		unsigned int nb;
		PrimParam(unsigned int b, GLenum m, float w): begin(b),mode(m),width(w),nb(0) {}
	};

protected:
	Utils::VBO* m_vboPos;
	Utils::VBO* m_vboCol;
	std::vector<Geom::Vec3f> m_dataPos;
	std::vector<Geom::Vec3f> m_dataCol;
	std::vector<PrimParam> m_begins;
	float m_currentWidth;
	float m_currentSize;
	GLenum m_compile;
	Utils::ShaderColorPerVertex* m_shader;

public:

	/**
	 * constructor, init all buffers (data and OpenGL) and shader
	 * @Warning need OpenGL context
	 */
	Drawer();

	/**
	 * release buffers and shader
	 */
	~Drawer();

	Utils::ShaderColorPerVertex* getShader();

	/**
	 * init the data structure
	 * @param comp say if compile only, or compile and execute (GL_COMPILE / GL_COMPILE_AND_EXECUTE)
	 */
	void newList(GLenum comp = GL_COMPILE);

	/**
	 * as glBegin, but need a newList call before
	 * @param mode: POINTS, LINES, LINE_LOOP, TRIANGLES, QUADS, POLYGON, etc..
	 * @return index of subdrawing possible call
	 */
	int begin(GLenum mode);

	/**
	 * as glEnd
	 */
	void end();

	/**
	 * finalize the data initialization
	 * drawn is done if newList called with GL_COMPILE_AND_EXECUTE
	 */
	void endList();

	/**
	 * use as glVertex
	 * @return index of position for update
	 */
	unsigned int vertex(const Geom::Vec3f& v);

	/**
	 * use as glVertex
	 * @return index of position for update
	 */
	unsigned int vertex3f(float x, float y, float z);

	/**
	 * use as glColor
	 */
	void color(const Geom::Vec3f& col);

	/**
	 * use as glColor3f
	 */
	void color3f(float r, float g, float b);

	/**
	 * use as a glCallList
	 */
	void callList(float opacity = 1.0f);

	/**
	 * call a sub-list (index return by begin (0,1,...)
	 */
	void callSubList(int index, float opacity = 1.0f);

	/**
	 * call a set of sub-lists
	 */
	void callSubLists(std::vector<int> indices, float opacity);

	/**
	 * call a set of sub-lists
	 */
	void callSubLists(int first, int nb, float opacity);


	/**
	 * use as glLineWidth
	 */
	void lineWidth(float lw);

	/**
	 * usr as glPointSize
	 */
	void pointSize(float ps);

	/**
	 * update position of VBO of drawer
	 * @param first index of vertex to update
	 * @param nb number of vertices to update
	 * @param P ptr to table of vertices
	 */
	void updatePositions(unsigned int first, unsigned int nb, const Geom::Vec3f* P);

	/**
	 * update position of VBO of drawer
	 * @param first index of vertex to update
	 * @param nb number of vertices to update
	 * @param P ptr to table of vertices
	 */
	void updatePositions(unsigned int first, unsigned int nb, const float* P);

	/**
	 * rendering in svg struct
	 */
	void toSVG(Utils::SVG::SVGOut& svg);
};

} // namespace Utils

} // namespace CGoGN

#endif
