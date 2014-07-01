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

#ifndef _GL2_TOPO_RENDER_
#define _GL2_TOPO_RENDER_

#include <vector>
#include <list>

#include "Topology/generic/dart.h"
#include "Topology/generic/attributeHandler.h"
#include "Topology/generic/functor.h"
#include "Geometry/vector_gen.h"

#include "Utils/vbo_base.h"
#include "Utils/svg.h"

#include "Utils/Shaders/shaderSimpleColor.h"
#include "Utils/Shaders/shaderColorPerVertex.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{

template <typename PFP>
class TopoRender
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

protected:
	/**
	* vbo buffers
	* 0: vertices darts
	* 1: vertices relation 1
	* 2: vertices relation 2
	* 3: color
	*/
	Utils::VBO* m_vbo0;
	Utils::VBO* m_vbo1;
	Utils::VBO* m_vbo2;
	Utils::VBO* m_vbo3;

	unsigned int m_vaId;

	/**
	*number of darts to draw
	*/
	GLuint m_nbDarts;

	/**
	* number of relations 1 to draw
	*/
	GLuint m_nbRel1;

	/**
	* number of relations 2 to draw
	*/
	GLuint m_nbRel2;

	/**
	 * width of lines use to draw darts
	 */
	float m_topo_dart_width;

	/**
	 * width of lines use to draw phi
	 */
	float m_topo_relation_width;

	/// shifting along normals for 3-map boundary drawing
	float m_normalShift;

	float m_boundShift;

	/**
	 * initial darts color (set in update)
	 */
	Geom::Vec3f m_dartsColor;

	/**
	 * initial darts color (set in update)
	 */
	Geom::Vec3f m_dartsBoundaryColor;

	float *m_color_save;

	/**
	 * attribut d'index dans le VBO
	 */
	DartAttribute<unsigned int, MAP> m_attIndex;

	Geom::Vec3f* m_bufferDartPosition;

	Utils::ShaderSimpleColor* m_shader1;
	Utils::ShaderColorPerVertex* m_shader2;

	/**
	 * compute color from dart index (for color picking)
	 */
	Dart colToDart(float* color);

	/**
	 * compute dart  from color (for color picking)
	 */
	void dartToCol(Dart d, float& r, float& g, float& b);

	/**
	 * pick the color in the rendered image
	 */
	Dart pickColor(unsigned int x, unsigned int y);

	/**
	 * affect a color to each dart
	 */
	void setDartsIdColor(MAP& map, bool withBoundary);

	/**
	 * save colors before picking
	 */
	void pushColors();

	/**
	 * restore colors after picking
	 */
	void popColors();

public:
	/**
	* Constructor
	* @param bs shift for boundary drawing
	*/
	TopoRender(float bs = 0.01f);

	/**
	* Destructor
	*/
	~TopoRender();

	/**
	 * set the with of line use to draw darts (default val is 2)
	 * @param dw width
	 */
	void setDartWidth(float dw);

	/**
	 * set the with of line use to draw phi (default val is 3)
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
	 * draw all topo
	 */
	void drawTopo();

	/**
	 * get shader objects
	 */
	Utils::GLSLShader* shader1() { return static_cast<Utils::GLSLShader*>(m_shader1); }
	Utils::GLSLShader* shader2() { return static_cast<Utils::GLSLShader*>(m_shader2); }

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
	 * @param d the dart
	 * @param r red !
	 * @param g green !
	 * @param b blue !
	 */
	void setAllDartsColor(float r, float g, float b);

	void setInitialDartsColor(float r, float g, float b);

	void setInitialBoundaryDartsColor(float r, float g, float b);

	/**
	 * redraw one dart with specific width and color (not efficient use only for debug with small amount of call)
	 * @param d the dart
	 * @param width the drawing width
	 * @param r red !
	 * @param g green !
	 * @param b blue !
	 */
	void overdrawDart(Dart d, float width, float r, float g, float b);

	/**
	 * pick dart with color set by setDartsIdColor
	 * Do not forget to apply same transformation to scene before picking than before drawing !
	 * @param map the map in which we pick (same as drawn !)
	 * @param x position of mouse (x)
	 * @param y position of mouse (pass H-y, classic pb of origin)
	 * @return the dart or NIL
	 */
	Dart picking(MAP& map, int x, int y, bool withBoundary=false);

	Dart coneSelection(MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float angle);

	Dart raySelection(MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float distmax);

	virtual void updateData(MAP& map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, bool withBoundary = false) = 0;

	/**
	 * Special update function used to draw boundary of map3
	 */
	void updateDataBoundary(MAP& map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, float ns);

	/**
	 * render to svg struct
	 */
	void toSVG(Utils::SVG::SVGOut& svg);

	/**
	 * render svg into svg file
	 */
	void svgout2D(const std::string& filename, const glm::mat4& model, const glm::mat4& proj);

	/**
	 * @brief set normal shift for boundary of dim 3 drawing
	 * @param ns distance shift along normals (use BB.diagSize()/100 is good approximation)
	 */
	void setNormalShift(float ns);

	/**
	 * @brief set boundary shift for boundary of dim 2 drawing
	 * @param ns distance shift
	 */
	void setBoundaryShift(float bs);
};

template <typename PFP>
class TopoRenderMap : public TopoRender<PFP>
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

public:
	TopoRenderMap(float bs = 0.01f) : TopoRender<PFP>(bs) {}
	void updateData(MAP &map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, bool withBoundary = false);
};

template <typename PFP>
class TopoRenderGMap : public TopoRender<PFP>
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

public:
	TopoRenderGMap(float bs = 0.01f) : TopoRender<PFP>(bs) {}
	void updateData(MAP &map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, bool withBoundary = false);
};

} // namespace GL2

} // namespace Render

} // namespace Algo

} // namespace CGoGN

#include "Algo/Render/GL2/topoRender.hpp"

#endif
