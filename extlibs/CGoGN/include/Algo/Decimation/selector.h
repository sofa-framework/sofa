/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2013, IGG Team, ICube, University of Strasbourg           *
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

#ifndef __SELECTOR_H__
#define __SELECTOR_H__

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Decimation
{

enum SelectorType
{
	// EDGE selectors
	// Topo criteria
	S_MapOrder = 0, /**< Prioritizes edges in topological order. Requires a geometry approximator. */
	S_Random = 1, /**< Prioritizes edges in random order. Requires a geometry approximator. */
	// Geom criteria
	S_EdgeLength = 2, /**< Prioritizes edges in increasing order of their lengths. Requires a geometry approximator. */
	S_QEM = 3, /**< Prioritizes by their quadric error metric (QEM) error [GH97]. Requires a geometry approximator. */
	S_QEMml = 4, /**< Prioritizes edges by their "memoryless" quadric error metric (QEM) error [Hop99]. Requires a geometry approximator. */
	S_MinDetail = 5, /**< Requires a geometry approximator. */
	S_Curvature = 6, /**< Requires a geometry approximator. */
	S_NormalArea = 7, /**< EXPERIMENTAL Prioritizes edges ranked by their normal times area measure [Sauvage]. Requires a geometry approximator. */
	S_CurvatureTensor = 8, /**< Requires a geometry approximator. */
	// Geom + color criteria
	S_ColorNaive = 9, /**< Prioritizes edges by a sum of the QEM measure on geometry and the L2 distance between RGB colors of both adjacent vertices.  Requires a geometry approximator. Requires a color approximator. */
	S_QEMextColor = 10, /**< Prioritizes edges by the quadric error metric (QEM) error extended in R^6 (x,y,z,R,G,B) [GH98].  Requires a geometry approximator. Requires a color approximator. */
	S_GeomColOptGrad = 11, /**< EXPERIMENTAL. Requires a geometry approximator. Requires a color approximator. */

	// HALF-EDGE selectors
	// Geom criteria
	S_hQEMml = 12, /**< Prioritizes half-edges according to the quadric error metric (QEM) of the considered collapse [Hop99]. Requires a geometry approximator. */
	// Geom + color criteria
	S_hQEMextColor = 13, /**< Prioritizes half-edges by the quadric error metric (QEM) error extended in R^6 (x,y,z,R,G,B) [GH98]. Requires a geometry approximator. Requires a color approximator. */
	S_hColorGradient = 14, /**< EXPERIMENTAL Prioritizes half-edges according to the sum of the quadric error metric (QEM) for geometry and the gradient color deviation metric of [Vanhoey,Sauvage]. Requires a geometry approximator. Requires a color approximator. */
	// Geom + color + normal criteria
	S_hQEMextColorNormal = 15, /**< Prioritizes half-edges by the quadric error metric (QEM) error extended in R^9 (x,y,z,R,G,B,nx,ny,nz) [GH98]. Requires a geometry approximator. Requires a color approximator. Requires a normal approximator */

	S_OTHER /**< Can be used for extensions. */
} ;

template <typename PFP> class ApproximatorGen ;
template <typename PFP, typename T, unsigned int ORBIT> class Approximator ;

/*!
 * \class Selector
 * \brief Generic class for selectors
 */
template <typename PFP>
class Selector
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	MAP& m_map ;
	VertexAttribute<VEC3, MAP>& m_position ;
	std::vector<ApproximatorGen<PFP>*>& m_approximators ;

public:
	Selector(MAP& m, VertexAttribute<VEC3, MAP>& pos, std::vector<ApproximatorGen<PFP>*>& approx) :
		m_map(m), m_position(pos), m_approximators(approx)
	{}
	virtual ~Selector()
	{}
	virtual SelectorType getType() = 0 ;
	virtual bool init() = 0 ;
	virtual bool nextEdge(Dart& d) const = 0 ;
	virtual void updateBeforeCollapse(Dart d) = 0 ;
	virtual void updateAfterCollapse(Dart d2, Dart dd2) = 0 ;
	virtual void updateWithoutCollapse() = 0;

	virtual void getEdgeErrors(EdgeAttribute<REAL, MAP>* /*errors*/) const
	{
		std::cout << "WARNING:: getEdgeErrors was not overridden" << std::endl ;
	}
} ;

} // namespace Decimation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif
