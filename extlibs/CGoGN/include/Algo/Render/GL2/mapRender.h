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

#ifndef _GL2_MAP_RENDER_
#define _GL2_MAP_RENDER_

#define _USE_MATH_DEFINES
#include <cmath>

#include <vector>
#include <list>
#include <set>
#include <utility>

#include "Utils/gl_def.h"
#include "Topology/generic/dart.h"
#include "Topology/generic/functor.h"
#include "Topology/generic/attributeHandler.h"
#include "Container/convert.h"
#include "Geometry/vector_gen.h"

// forward definition
namespace CGoGN { namespace Utils { class GLSLShader; } }

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{

enum drawingType
{
	POINTS = 0,
	LINES = 1,
	TRIANGLES = 2,
	FLAT_TRIANGLES = 3,
	BOUNDARY = 4,
	EXPLODED = 5,
	SIZE_BUFFER
} ;

class MapRender
{
protected:
	/**
	 * vbo buffers
	 */
	GLuint m_indexBuffers[SIZE_BUFFER] ;
	bool m_indexBufferUpToDate[SIZE_BUFFER];

	GLuint m_currentSize[SIZE_BUFFER] ;

	/**
	 * nb indices
	 */
	GLuint m_nbIndices[SIZE_BUFFER] ;

	typedef std::pair<GLuint*, unsigned int> buffer_array;

	// forward declaration
	class VertexPoly;

	// comparaison function for multiset
	static bool cmpVP(VertexPoly* lhs, VertexPoly* rhs);

	// multiset typedef for simple writing
	typedef std::multiset<VertexPoly*, bool(*)(VertexPoly*,VertexPoly*)> VPMS;

	class VertexPoly
	{
	public:
		int id;
		float value;
		float length;
		VertexPoly* prev;
		VertexPoly* next;
		VPMS::iterator ear;

		VertexPoly(int i, float v, float l, VertexPoly* p = NULL) : id(i), value(v), length(l), prev(p), next(NULL)
		{
			if (prev != NULL)
				prev->next = this;
		}

		static void close(VertexPoly* first, VertexPoly* last)
		{
			last->next = first;
			first->prev = last;
		}

		static VertexPoly* erase(VertexPoly* vp)
		{
			VertexPoly* tmp = vp->prev;
			tmp->next = vp->next;
			vp->next->prev = tmp;
			delete vp;
			return tmp;
		}
	};

public:
	/**
	 * Constructor
	 */
	MapRender() ;

	/**
	 * Constructor that share vertices attributes vbo (position/normals/colors...)
	 */
	MapRender(const MapRender& mrvbo);

	/**
	 * Destructor
	 */
	~MapRender() ;

	buffer_array get_index_buffer() { return std::make_pair(m_indexBuffers, SIZE_BUFFER); }
	buffer_array get_nb_index_buffer() { return std::make_pair(m_nbIndices, SIZE_BUFFER); }

protected:
	/**
	 * addition of indices table of one triangle
	 * @param d a dart of the triangle
	 * @param tableIndices the indices table
	 */
	template <typename PFP>
	void addTri(typename PFP::MAP& map, Face f, std::vector<GLuint>& tableIndices) ;

	template<typename PFP>
	inline void addEarTri(typename PFP::MAP& map, Face f, std::vector<GLuint>& tableIndices, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>* position);

	template<typename PFP>
	float computeEarAngle(const typename PFP::VEC3& P1, const typename PFP::VEC3& P2, const typename PFP::VEC3& P3, const typename PFP::VEC3& normalPoly);

	template<typename PFP>
	bool computeEarIntersection(const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexPoly* vp, const typename PFP::VEC3& normalPoly);

	template<typename PFP>
	void recompute2Ears(const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexPoly* vp, const typename PFP::VEC3& normalPoly, VPMS& ears, bool convex);

	template<typename VEC3>
	bool inTriangle(const VEC3& P, const VEC3& normal, const VEC3& Ta, const VEC3& Tb, const VEC3& Tc);

public:
	/**
	 * creation of indices table of triangles (optimized order)
	 * @param tableIndices the table where indices are stored
	 */
	template <typename PFP>
	void initTriangles(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>* position, unsigned int thread = 0) ;
	template <typename PFP>
	void initTrianglesOptimized(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>* position, unsigned int thread = 0) ;

	/**
	 * creation of indices table of lines (optimized order)
	 * @param tableIndices the table where indices are stored
	 */
	template <typename PFP>
	void initLines(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread = 0) ;
	template <typename PFP>
	void initLinesOptimized(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread = 0) ;

	/**
	 * creation of indices table of points
	 * @param tableIndices the table where indices are stored
	 */
	template <typename PFP>
	void initPoints(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread = 0) ;

	/**
	 * creation of indices table of points
	 * @param tableIndices the table where indices are stored
	 */
	template <typename PFP>
	void initBoundaries(typename PFP::MAP& map, std::vector<GLuint>& tableIndices, unsigned int thread = 0) ;
	/**
	 * initialization of the VBO indices primitives
	 * computed by a traversal of the map
	 * @param prim primitive to draw: POINTS, LINES, TRIANGLES
	 */
	template <typename PFP>
	void initPrimitives(typename PFP::MAP& map, int prim, bool optimized = true, unsigned int thread = 0) ;

	template <typename PFP>
	void initPrimitives(typename PFP::MAP& map, int prim, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>* position, bool optimized = true, unsigned int thread = 0) ;

	/**
	 * add primitives to the VBO of indices
	 */
	template <typename PFP>
	void addPrimitives(typename PFP::MAP& map, int prim, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>* position, bool optimized = true, unsigned int thread = 0);

	/**
	 * initialization of the VBO indices primitives
	 * using the given table
	 * @param prim primitive to draw: POINTS, LINES, TRIANGLES
	 */
	void initPrimitives(int prim, std::vector<GLuint>& tableIndices) ;

	/**
	 * return if the given primitive connectivity VBO is up to date
	 * @param prim primitive to draw: POINT_INDICES, LINE_INDICES, TRIANGLE_INDICES
	 */
	bool isPrimitiveUpToDate(int prim) { return m_indexBufferUpToDate[prim]; }

	/**
	 * set the given primitive connectivity VBO dirty
	 * @param prim primitive to draw: POINT_INDICES, LINE_INDICES, TRIANGLE_INDICES
	 */
	void setPrimitiveDirty(int prim) { m_indexBufferUpToDate[prim] = false; }

	/**
	 * draw the VBO (function to call in the drawing callback)
	 */
	void draw(Utils::GLSLShader* sh, int prim) ;

	unsigned int drawSub(Utils::GLSLShader* sh, int prim, unsigned int nb_elm);
} ;

} // namespace GL2

} // namespace Render

} // namespace Algo

} // namespace CGoGN

#include "Algo/Render/GL2/mapRender.hpp"

#endif
