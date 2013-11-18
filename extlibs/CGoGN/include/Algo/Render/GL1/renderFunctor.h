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

#ifndef __RENDER_FUNCTOR_GL_H
#define __RENDER_FUNCTOR_GL_H

#include <vector>

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL1
{

template<typename PFP>
class FunctorGLFace: public FunctorMap<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP;
	
protected:
	/**
	* functor used in GL smooth rendering ? (normal per vertex or per face)
	*/
	bool m_smooth;

	/**
	* functor used in lighted rendering ? (use normal or not)
	*/
	bool m_lighted;

	/**
	* Nb edges of primitive (3 for triangle and 4 for quad, -1 for polygon)
	*/
	unsigned m_nbEdges;

	/**
	* coefficient for exploding faces
	*/
	float m_explode;

	/**
	* storing faces that have not been rendered
	*/
	bool m_storing;

	/**
	 * positions of vertices
	 */
	const VertexAttribute<typename PFP::VEC3>& m_positions;

	/**
	 * normals of vertices
	 */
	const VertexAttribute<typename PFP::VEC3>& m_normals;

	/**
	* vector of darts (one for each polygon)
	*/
	std::vector<Dart> m_poly;


public:
	/**
	* @param lighted use normal (face or vertex)
	* @param smooth use per vertex normal
	* @param nbe number of vertex per primitive (3 for triangles, 4 for quads, -1 for polygons)
	* @param expl exploding coefficient
	* @param stor shall we store faces that are not of the good primitive type
	* @param vn the vertex normal vector (indiced by dart label)
	*/
	FunctorGLFace(MAP& map, bool lighted, bool smooth, int nbe, float expl, bool stor,
		const VertexAttribute<typename PFP::VEC3>& posi, const VertexAttribute<typename PFP::VEC3>& normals );

	/**
	* get back the vector of darts of faces that have not been treated
	*/
	std::vector<Dart>& getPolyDarts();

	/**
	* operator applied on each face:
	* if the face has the right number of edges
	* it is rendered (glVertex). Other are stored
	* if needed.
	*/
	bool operator() (Dart d);

	/**
	* Render a face without exploding
	*/
	void renderFace(Dart d);

	/**
	* Render a face with exploding
	*/
	void renderFaceExplode(Dart d);
};


/**
 * Fonctor for color rendering
 */

template<typename PFP>
class FunctorGLFaceColor: public FunctorMap<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP;

protected:
	/**
	* functor used in GL smooth rendering ? (normal per vertex or per face)
	*/
	bool m_smooth;

	/**
	* functor used in lighted rendering ? (use normal or not)
	*/
	bool m_lighted;

	/**
	* Nb edges of primitive (3 for triangle and 4 for quad, -1 for polygon)
	*/
	unsigned m_nbEdges;

	/**
	* coefficient for exploding faces
	*/
	float m_explode;

	/**
	* storing faces that have not been rendered
	*/
	bool m_storing;

	/**
	 * positions of vertices
	 */
	const VertexAttribute<typename PFP::VEC3>& m_positions;

	/**
	 * normals of vertices
	 */
	const VertexAttribute<typename PFP::VEC3>& m_normals;

	/**
	 * colors of vertices
	 */
	const VertexAttribute<typename PFP::VEC3>& m_colors;

	/**
	* vector of darts (one for each polygon)
	*/
	std::vector<Dart> m_poly;


public:

	FunctorGLFaceColor(MAP& map, bool lighted, bool smooth, int nbe, float expl, bool stor,
		const VertexAttribute<typename PFP::VEC3>& posi, const VertexAttribute<typename PFP::VEC3>& normals , const VertexAttribute<typename PFP::VEC3>& colors);

	std::vector<Dart>& getPolyDarts();

	bool operator() (Dart d);

	void renderFace(Dart d);

	void renderFaceExplode(Dart d);
};

template<typename PFP>
class FunctorGLNormal : public  CGoGN::FunctorMap<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP;


protected:
	/**
	 * positions of vertices
	 */
	const VertexAttribute<typename PFP::VEC3>& m_positions;

	/**
	 * normals of vertices
	 */
	const VertexAttribute<typename PFP::VEC3>& m_normals;


	float m_scale;

public:

	FunctorGLNormal(MAP& map, const VertexAttribute<typename PFP::VEC3>& posi, const VertexAttribute<typename PFP::VEC3>& normals, float scale);

	bool operator() (Dart d);
};

template<typename PFP>
class FunctorGLFrame : public  CGoGN::FunctorMap<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP;


protected:
	/**
	 * positions of vertices
	 */
	const VertexAttribute<typename PFP::VEC3>& m_positions;

	/**
	 * frame of vertices
	 */
	const VertexAttribute<typename PFP::VEC3> *m_frames;


	float m_scale;

public:

	FunctorGLFrame (MAP& map, const VertexAttribute<typename PFP::VEC3>& posi, const VertexAttribute<typename PFP::VEC3> frames[3], float scale);

	bool operator() (Dart d);
};

} // namespace GL1

} // namespace Render

} // namespace Algo

} // namespace CGoGN

#include "Algo/Render/GL1/renderFunctor.hpp"

#endif
