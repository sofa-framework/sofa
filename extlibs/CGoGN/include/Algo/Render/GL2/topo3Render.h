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

#ifndef _TOPO3_VBO_RENDER_
#define _TOPO3_VBO_RENDER_

#include <vector>
#include <list>

#include "Topology/generic/dart.h"
#include "Topology/generic/attributeHandler.h"
#include "Topology/generic/functor.h"
#include "Geometry/vector_gen.h"

#include "Utils/GLSLShader.h"
#include "Utils/Shaders/shaderSimpleColor.h"
#include "Utils/Shaders/shaderColorPerVertex.h"

#include "Utils/vbo_base.h"
#include "Utils/svg.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{

template <typename PFP>
class Topo3Render
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

protected:
	/**
	* vbo buffers
	* 0: vertices darts
	* 1: vertices phi1 / beta1
	* 2: vertices phi2 / beta2
	* 3: vertices phi3 / beta3
	* 4: colors
	*/
	Utils::VBO* m_vbo0;
	Utils::VBO* m_vbo1;
	Utils::VBO* m_vbo2;
	Utils::VBO* m_vbo3;
	Utils::VBO* m_vbo4;

	unsigned int m_vaId;

	Utils::ShaderSimpleColor* m_shader1;
	Utils::ShaderColorPerVertex* m_shader2;

	/**
	*number of darts to draw
	*/
	GLuint m_nbDarts;

	/**
	* number of relations 2 to draw
	*/
	GLuint m_nbRel1;

	/**
	* number of relations 2 to draw
	*/
	GLuint m_nbRel2;

	/**
	* number of relations 3 to draw
	*/
	GLuint m_nbRel3;

	/**
	 * width of lines use to draw darts
	 */
	float m_topo_dart_width;

	/**
	 * width of lines use to draw phi
	 */
	float m_topo_relation_width;

	/**
	 * pointer for saved colorvbo (in picking)
	 */
	float *m_color_save;

	/**
	 * initial darts color (set in update)
	 */
	Geom::Vec3f m_dartsColor;

	/**
	 * attribute index to get easy correspondence dart/color
	 */
	DartAttribute<unsigned int, MAP> m_attIndex;

	Geom::Vec3f* m_bufferDartPosition;

	/**
	 * save colors
	 */
	void pushColors();

	/**
	 * restore colors
	 */
	void popColors();

	/**
	 * pick dart with color set by setDartsIdColor
	 * @param x position of mouse (x)
	 * @param y position of mouse (pass H-y, classic pb of origin)
	 * @return the dart or NIL
	 */
	Dart pickColor(unsigned int x, unsigned int y);

public:

	/**
	* Constructor
	*/
	Topo3Render();

	/**
	* Destructor
	*/
	~Topo3Render();

	Utils::GLSLShader* shader1() { return static_cast<Utils::GLSLShader*>(m_shader1); }
	Utils::GLSLShader* shader2() { return static_cast<Utils::GLSLShader*>(m_shader2); }

	/**
	 * set the with of line use to draw darts (default val is 2)
	 * @param dw width
	 */
	void setDartWidth(float dw);

	/**
	 * set the with of line use to draw phi (default val is ")
	 * @param pw width
	 */
	void setRelationWidth(float pw);

	/**
	* Drawing function for darts only
	*/
	void drawDarts();

	/**
	* Drawing function for phi1 only
	*/
	void drawRelation1();

	/**
	* Drawing function for phi2 only
	*/
	void drawRelation2();

	/**
	* Drawing function for phi2 only
	*/
	void drawRelation3(Geom::Vec4f c);

	/**
	 * draw all topo
	 * \warning DO NOT FORGET TO DISABLE CULLFACE BEFORE CALLING
	 */
	void drawTopo();

//	void drawDart(Dart d, float R, float G, float B, float width);

	/**
	 * change dart drawing color
	 * @param d the dart
	 * @param r red !
	 * @param g green !
	 * @param b blue !
	 */
	void setDartColor(Dart d, float r, float g, float b);

	/**
	 * change all darts drawing color
	 * @param r red !
	 * @param g green !
	 * @param b blue !
	 */
	void setAllDartsColor(float r, float g, float b);

	/**
	 * change dart initial color (used when calling updateData)
	 * @param r red !
	 * @param g green !
	 * @param b blue !
	 */
	void setInitialDartsColor(float r, float g, float b);

	/**
	 * overdraw a dart with given width and color
	 * @param d the dart
	 * @param width drawing width
	 * @param r red !
	 * @param g green !
	 * @param b blue !
	 */
	void overdrawDart(Dart d, float width, float r, float g, float b);

	/**
	 * store darts in color for picking
	 */
	void setDartsIdColor(MAP& map);

	/**
	 * pick dart with color set bey setDartsIdColor
	 * Do not forget to apply same transformation to scene before picking than before drawing !
	 * @param map the map

	 * @param x position of mouse (x)
	 * @param y position of mouse (pass H-y, classic pb of origin)
	 * @return the dart or NIL
	 */
	Dart picking(MAP& map, int x, int y);

	/**
	 * compute dart from color (for picking)
	 */
	Dart colToDart(float* color);

	/**
	 * compute color from dart (for picking)
	 */
	void dartToCol(Dart d, float& r, float& g, float& b);

	/**
	* update all drawing buffers to render a dual map
	* @param map the map
	* @param positions  attribute of position vertices
	* @param ke exploding coef for edge
	* @param kf exploding coef for face
 	* @param kv exploding coef for face
	*/
	virtual void updateData(MAP& map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, float kv) = 0;

//	template<typename PFP>
//	void updateData(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& positions, float ke, float kf, float kv);

	/**
	* update color buffer with color attribute handler
	* @param map the map
	* @param colors  attribute of dart's colors
	*/
//	template<typename PFP, typename EMBV>
//	void updateColorsGen(typename PFP::MAP& map, const EMBV& colors);

	void updateColors(MAP& map, const VertexAttribute<Geom::Vec3f, MAP>& colors);

	/**
	 * Get back middle position of drawn darts
	 * @param map the map
	 * @param posExpl the output positions
	 */
	void computeDartMiddlePositions(MAP& map, DartAttribute<VEC3, MAP>& posExpl);

	/**
	 * render to svg struct
	 */
	void toSVG(Utils::SVG::SVGOut& svg);

	/**
	 * render svg into svg file
	 */
	void svgout2D(const std::string& filename, const glm::mat4& model, const glm::mat4& proj);

	Dart coneSelection(MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float angle);

	Dart raySelection(MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float distmax);

//protected:
//	/**
//	* update all drawing buffers to render a dual map
//	* @param map the map
//	* @param positions  attribute of position vertices
//	* @param ke exploding coef for edge
//	* @param kf exploding coef for face
// 	* @param kv exploding coef for face
//	*/
//	void updateDataMap3(MAP& map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, float kv);

//	/**
//	* update all drawing buffers to render a gmap
//	* @param map the map
//	* @param positions  attribute of position vertices
//	* @param ke exploding coef for edge
//	* @param kf exploding coef for face
// 	* @param kv exploding coef for face
//	*/
//	void updateDataGMap3(MAP& map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, float kv);
};

template <typename PFP>
class Topo3RenderMap : public Topo3Render<PFP>
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

public:
	void updateData(MAP &map, const VertexAttribute<VEC3, MAP> &positions, float ke, float kf, float kv);
};

template <typename PFP>
class Topo3RenderGMap : public Topo3Render<PFP>
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;

public:
	void updateData(MAP &map, const VertexAttribute<VEC3, MAP> &positions, float ke, float kf, float kv);
};

}//end namespace GL2

}//end namespace Algo

}//end namespace Render

}//end namespace CGoGN

#include "Algo/Render/GL2/topo3Render.hpp"

#endif
