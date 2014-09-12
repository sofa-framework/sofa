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

#ifndef __TEXT_3D__
#define __TEXT_3D__

#include "Utils/GLSLShader.h"
#include "Geometry/vector_gen.h"

namespace CGoGN { namespace Utils { namespace SVG { class SVGOut ; } } }

namespace CGoGN
{
namespace Utils
{

class Strings3D : public GLSLShader
{
protected:
    static const unsigned int WIDTHFONT = 32;

    static const unsigned int REALWIDTHFONT = 18;

    static const unsigned int WIDTHTEXTURE = 512;

    static const unsigned int HEIGHTTEXTURE = 256;

    static const unsigned int CHARSPERLINE = WIDTHTEXTURE/WIDTHFONT;

    static const unsigned int CHARSPERCOL = HEIGHTTEXTURE/WIDTHFONT;

    static std::string vertexShaderText;

    static std::string fragmentShaderText1;
    static std::string fragmentShaderText2;
    static std::string fragmentShaderText3;

    static Strings3D*  m_instance0;

	std::vector<std::string> m_strings;

	unsigned int m_nbChars;

    std::vector< std::pair<unsigned int, unsigned int> > m_strpos;

    std::vector< Geom::Vec3f > m_strTranslate;

    Utils::VBO* m_vbo1;

    float m_scale;

    Geom::Vec4f m_color;

    unsigned int sendOneStringToVBO(const std::string& str, float **buffer);

    CGoGNGLuint m_idTexture;

    CGoGNGLuint m_uniform_texture;

    CGoGNGLuint m_uniform_scale;

    CGoGNGLuint m_uniform_position;

    CGoGNGLuint m_uniform_color;
	
	CGoGNGLuint m_uniform_planeX;
	
	CGoGNGLuint m_uniform_planeY;

public:
	Strings3D(bool withBackground = true, const Geom::Vec3f& bgc = Geom::Vec3f(0.0f, 0.0f, 0.0f), bool with_plane=false);

	~Strings3D();

	/**
	 * add astring
	 * @param string to add to pool of string
	 * @return the id of string for rendering
	 */
	unsigned int addString(const std::string& str);

	/**
	 * add astring with its position
	 * @param string to add to pool of string
	 * @param pos the position of text
	 * @return the id of string (future use:updating data)
	 */
	unsigned int addString(const std::string& str, const Geom::Vec3f& pos);

	/**
	 * clear the string and position database
	 */
	void clear();

	/**
	 * once all string are stored, we must send it to the graphic card
	 */
	void sendToVBO();

	/**
	 * draw one string
	 * @param idSt the id of string
	 * @param pos the position of text
	 */
	void draw(unsigned int idSt, const Geom::Vec3f& pos);

	/**
	* Draw all text stored with their position
	* @param color the color of text
	*/
	void drawAll(const Geom::Vec4f& color);

	void drawAll(const Geom::Vec3f& color){ drawAll( Geom::Vec4f(color[0],color[1],color[2],m_color[3])); }


	/**
	* call once before several draw(id,pos)
	* @param color the color of text
	*/
	void predraw(const Geom::Vec4f& color);

	void predraw(const Geom::Vec3f& color){ predraw( Geom::Vec4f(color[0],color[1],color[2],m_color[3])); }


	/**
	* call just before a draw
	* @param color the color of text
	*/
	void changeColor(const Geom::Vec4f& color);

	void changeColor(const Geom::Vec3f& color){ changeColor( Geom::Vec4f(color[0],color[1],color[2],m_color[3])); }

	/**
	 * change opacity but keep color (0 is totally transparent 1 is totally opaque)
	 * @param opacity
	 */
	void changeOpacity(float op);

	/**
	* call once after several draw(id,pos)
	*/
	void postdraw();

	/**
	 * set the scale for font rendering
	 * @param scale
	 */
	void setScale(float scale);
	
	/**
	* set the plane of rendering for VR rendering
	*/
	void setPlane(const Geom::Vec3f& ox, const Geom::Vec3f& oy);

	/**
	 * Update string content
	 * @warning size can not be changed
	 * @param idSt string id
	 * @param str new string content
	 */
	void updateString(unsigned int idSt, const std::string& str);


	void toSVG(Utils::SVG::SVGOut& svg);
};

} // namespace Utils

} // namespace CGoGN

#endif
