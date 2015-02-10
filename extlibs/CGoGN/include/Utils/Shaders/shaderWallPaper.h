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

#ifndef __CGOGN_SHADER_WALLPAPER__
#define __CGOGN_SHADER_WALLPAPER__


#include "Geometry/vector_gen.h"
#include "Utils/GLSLShader.h"
#include "Utils/clippingShader.h"
#include "Utils/textures.h"
#include "Utils/gl_def.h"

namespace CGoGN
{

namespace Utils
{

class ShaderWallPaper : public ClippingShader
{
protected:
	// shader sources
	static std::string vertexShaderText;
	static std::string fragmentShaderText;

	CGoGNGLuint m_unif_unit;
	CGoGNGLuint m_unif_pos;
	CGoGNGLuint m_unif_sz;
	int m_unit;

	Utils::GTexture* m_tex_ptr;
	VBO* m_vboPos;
	VBO* m_vboTexCoord;

	void restoreUniformsAttribs();

public:
	ShaderWallPaper();

	~ShaderWallPaper();

	/**
	 * choose the texture unit engine to use for this texture
	 */
	void setTextureUnit(GLenum texture_unit);

	/**
	 * set the texture to use
	 */
	void setTexture(Utils::GTexture* tex);

	/**
	 * activation of texture unit with set texture
	 */
	void activeTexture();
	
	/**
	 * activation of texture unit with texture id
	 */
	void activeTexture(CGoGNGLuint texId);

	/**
	 * @brief draw the quad as wallpaper
	 */
	void draw();

	/**
	 * @brief draw a specific texture at a specific place in a window
	 * @param window_w
	 * @param window_h
	 * @param x
	 * @param y
	 * @param w
	 * @param h
	 * @param button
	 */
	void drawBack(int window_w, int window_h, int x, int y, int w, int h, Utils::GTexture* button);

	void drawFront(int window_w, int window_h, int x, int y, int w, int h, Utils::GTexture* button);


};

} // namespace Utils

} // namespace CGoGN


#endif /* __CGOGN_SHADER_SIMPLETEXTURE__ */
