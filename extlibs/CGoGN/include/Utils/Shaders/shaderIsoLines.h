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

#ifndef __CGOGN_SHADER_ISO_LINES__
#define __CGOGN_SHADER_ISO_LINES__

#include "Utils/GLSLShader.h"
#include "Geometry/vector_gen.h"

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{
/**
 * Shader to draw iso-lines on a triangles mesh.
 * Iso-line are computed on a data vertex attribute (float)
 * nb iso-lines, min/max attributes value and colors can be changed on the fly
 * For better rendering result use glEnable(GL_LINE_SMOOTH)
 */
class CGoGN_UTILS_API ShaderIsoLines : public GLSLShader
{
protected:
	/// shader sources
    static std::string vertexShaderText;
    static std::string fragmentShaderText;
    static std::string geometryShaderText;

    /// uniform locations
	CGoGNGLuint m_unif_colorMin;
	CGoGNGLuint m_unif_colorMax;
	CGoGNGLuint m_unif_vmin;
	CGoGNGLuint m_unif_vmax;
	CGoGNGLuint m_unif_vnb;

	// colors of iso-lines
	Geom::Vec4f m_colorMin;
	Geom::Vec4f m_colorMax;

	/// min/max of data attribute
	float m_vmin;
	float m_vmax;

	/// number of iso-line to draw
	int m_vnb;

	VBO* m_vboPos;
	VBO* m_vboData;

	void getLocations();

public:
	/**
	 * constructor
	 * @param max number of isolines drawable per triangle (as low as possible for performance)
	 */
	ShaderIsoLines(int maxNbIsoPerTriangle=6);

	/**
	 * set colors for min and max isoLine, interpolated between
//	 * @param colorMin color for minimum iso-line value
	 * @param colorMax color for maximum iso-line value
	 */
	void setColors(const Geom::Vec4f& colorMin, const Geom::Vec4f& colorMax);

	/**
	 * Set min and max value of used atribute.
	 * @param attMin minimun of attribute
	 * @param attMax maximun of attribute
	 */
	void setDataBound(float attMin, float attMax);

	/**
	 * set the number of iso-lines (default is 32)
	 */
	void setNbIso(int nb);

	/**
	 * set max number of isolines per triangle
	 * If to small risk of missing lines
	 * if to big performance problem
	 */
	void setNbMaxIsoLinePerTriangle(int nb) { changeNbMaxVertices(2*nb); }

	/**
	 * Position attribute
	 */
	unsigned int setAttributePosition(VBO* vbo);

	/**
	 * Data attribute for iso-lines must be of type float
	 */
	unsigned int setAttributeData(VBO* vbo);
};

} // namespace Utils

} // namespace CGoGN

#endif
