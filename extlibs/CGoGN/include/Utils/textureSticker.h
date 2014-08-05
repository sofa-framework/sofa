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

#ifndef _CGOGN_TEXTURESTICKER_H_
#define _CGOGN_TEXTURESTICKER_H_

#include "Utils/vbo_base.h"
#include "Utils/Shaders/shaderSimpleTexture.h"
#include "Utils/Shaders/shaderTextureDepth.h"
#include "Utils/gl_def.h"

namespace CGoGN
{

namespace Utils
{

/**
 * Static class that can be used to "stick" textures easily on the screen.
 */
class TextureSticker
{

private :

	/**
	 * Constructor (not accesssible since the class is static).
	 */
	TextureSticker();
	
	/**
	 * Destructor (not accesssible since the class is static).
	 */
	~TextureSticker();
	
public :

	/**
	 * Sticks a texture on the whole screen.
	 * \param  texId  Id of the texture
	 */
	static void fullScreenTexture(CGoGNGLuint texId);
	
	/**
	 * Sticks a texture on the whole screen.
	 * \param  texId  Id of the texture
	 * \param  dtexId  Id of the depth texture
	 */
	static void fullScreenTextureDepth(CGoGNGLuint texId, CGoGNGLuint dtexId);


	/**
	 * Draw a fullscreen quad with the specified shader.\n
	 * Uniforms other than matrices must have been sent to the shader before the call to this function.\n
	 * \param  shader  Shader that will be used to render the quad
	 */
	static void fullScreenShader(Utils::GLSLShader* shader);
	
	/**
	 * Get the Vbo of the vertices positions of the fullscreen quad.
	 * \returns  Fullscreen quad vertices positions Vbo
	 */
	static VBO* getPositionsVbo();
	
	/**
	 * Get the Vbo of the vertices tex coords of the fullscreen quad.
	 * \returns  Fullscreen quad vertices tex coords Vbo
	 */
	static VBO* getTexCoordsVbo();
	
private :

	/**
	 * Initializes static elements of the class.\n
	 * Automatically called when it's needed.\n
	 */
	static void initializeElements();
	
	/// Indicates wether the static elements of the class were already initialized or not.
	static bool sm_isInitialized;
	
	/// Vbo of the vertices positions of the fullscreen quad.
	static Utils::VBO* sm_quadPositionsVbo;
	
	/// Vbo of the vertices texture coords of the fullscreen quad.
	static Utils::VBO* sm_quadTexCoordsVbo;
	
	/// Shader for mapping the texture on the fullscreen quad.
	static Utils::ShaderSimpleTexture* sm_textureMappingShader;
	static Utils::ShaderTextureDepth* sm_depthtextureMappingShader;

};

} // namespace Utils

} // namespace CGoGN

#endif

