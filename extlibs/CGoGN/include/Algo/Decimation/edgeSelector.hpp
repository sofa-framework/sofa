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
#include <Algo/Topo/embedding.h>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Decimation
{

/************************************************************************************
 *                                  MAP ORDER                                       *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_MapOrder<PFP>::init()
{
	cur = this->m_map.begin() ;
	return true ;
}

template <typename PFP>
bool EdgeSelector_MapOrder<PFP>::nextEdge(Dart& d) const
{
	MAP& m = this->m_map ;
	if(cur == m.end())
		return false ;
	d = cur ;
	return true ;
}

template <typename PFP>
void EdgeSelector_MapOrder<PFP>::updateAfterCollapse(Dart /*d2*/, Dart /*dd2*/)
{
	MAP& m = this->m_map ;
	cur = m.begin() ;
	while( !m.edgeCanCollapse(cur))
	{
		m.next(cur) ;
		if(cur == m.end())
			break ;
	}
}

/************************************************************************************
 *                                    RANDOM                                        *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_Random<PFP>::init()
{
	MAP& m = this->m_map ;

	darts.reserve(m.getNbDarts()) ;
	darts.clear() ;

	for(Dart d = m.begin(); d != m.end(); m.next(d))
		darts.push_back(d) ;

	srand(time(NULL)) ;
	int remains = darts.size() ;
	for(unsigned int i = 0; i < darts.size()-1; ++i) // generate the random permutation
	{
		int r = (rand() % remains) + i ;
		// swap ith and rth elements
		Dart tmp = darts[i] ;
		darts[i] = darts[r] ;
		darts[r] = tmp ;
		--remains ;
	}
	cur = 0 ;
	allSkipped = true ;

	return true ;
}

template <typename PFP>
bool EdgeSelector_Random<PFP>::nextEdge(Dart& d) const
{
	if(cur == darts.size() && allSkipped)
		return false ;
	d = darts[cur] ;
	return true ;
}

template <typename PFP>
void EdgeSelector_Random<PFP>::updateAfterCollapse(Dart /*d2*/, Dart /*dd2*/)
{
	MAP& m = this->m_map ;
	allSkipped = false ;
	do
	{
		++cur ;
		if(cur == darts.size())
		{
			cur = 0 ;
			allSkipped = true ;
		}
	} while( !m.edgeCanCollapse(darts[cur])) ;
}

template <typename PFP>
void EdgeSelector_Random<PFP>::updateWithoutCollapse()
{
	MAP& m = this->m_map ;
	allSkipped = false ;
	do
	{
		++cur ;
		if(cur == darts.size())
		{
			cur = 0 ;
			allSkipped = true ;
		}
	} while( !m.edgeCanCollapse(darts[cur])) ;
}

/************************************************************************************
 *                                 EDGE LENGTH                                      *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_Length<PFP>::init()
{
	MAP& m = this->m_map ;

	edges.clear() ;

	CellMarker<MAP, EDGE> eMark(m) ;
	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		if(!eMark.isMarked(d))
		{
			initEdgeInfo(d) ;
			eMark.mark(d) ;
		}
	}

	cur = edges.begin() ; // init the current edge to the first one

	return true ;
}

template <typename PFP>
bool EdgeSelector_Length<PFP>::nextEdge(Dart& d) const
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_Length<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo *edgeE = &(edgeInfo[d]) ;
	if(edgeE->valid)
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)					// remove all
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi_1(d)]) ;	// the concerned edges
	if(edgeE->valid)
		edges.erase(edgeE->it) ;
									// from the multimap
	Dart dd = m.phi2(d) ;
	if(dd != d)
	{
		edgeE = &(edgeInfo[m.phi1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;

		edgeE = &(edgeInfo[m.phi_1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;
	}
}

template <typename PFP>
void EdgeSelector_Length<PFP>::updateAfterCollapse(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;

	Dart vit = d2 ;
	do
	{
		updateEdgeInfo(m.phi1(vit), false) ;		// must recompute some edge infos in the
		if(vit == d2 || vit == dd2)					// neighborhood of the collapsed edge
		{
			initEdgeInfo(vit) ;						// various optimizations are applied here by
													// treating differently :
			Dart vit2 = m.phi12(m.phi1(vit)) ;		// - edges for which the criteria must be recomputed
			Dart stop = m.phi2(vit) ;				// - edges that must be re-embedded
			do										// - edges for which only the collapsibility must be re-tested
			{
				updateEdgeInfo(vit2, false) ;
				updateEdgeInfo(m.phi1(vit2), false) ;
				vit2 = m.phi12(vit2) ;
			} while(vit2 != stop) ;
		}
		else
			updateEdgeInfo(vit, true) ;

		vit = m.phi2_1(vit) ;
	} while(vit != d2) ;

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_Length<PFP>::updateWithoutCollapse()
{
	EdgeInfo& einfo = edgeInfo[(*cur).second] ;
	einfo.valid = false ;
	edges.erase(einfo.it) ;

	//edges.erase(cur) ;
	cur = edges.begin();
}

template <typename PFP>
void EdgeSelector_Length<PFP>::initEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;
	EdgeInfo einfo ;
	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
	edgeInfo[d] = einfo ;
}

template <typename PFP>
void EdgeSelector_Length<PFP>::updateEdgeInfo(Dart d, bool recompute)
{
	MAP& m = this->m_map ;
	EdgeInfo& einfo = edgeInfo[d] ;
	if(recompute)
	{
		if(einfo.valid)
			edges.erase(einfo.it) ;			// remove the edge from the multimap
		if(m.edgeCanCollapse(d))
			computeEdgeInfo(d, einfo) ;
		else
			einfo.valid = false ;
	}
	else
	{
		if(m.edgeCanCollapse(d))
		{									// if the edge can be collapsed now
			if(!einfo.valid)				// but it was not before
				computeEdgeInfo(d, einfo) ;
		}
		else
		{									// if the edge cannot be collapsed now
			if(einfo.valid)					// and it was before
			{
				edges.erase(einfo.it) ;
				einfo.valid = false ;
			}
		}
	}
}

template <typename PFP>
void EdgeSelector_Length<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	VEC3 vec = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->m_map, d, this->m_position) ;
	einfo.it = edges.insert(std::make_pair(vec.norm2(), d)) ;
	einfo.valid = true ;
}

/************************************************************************************
 *                            QUADRIC ERROR METRIC                                  *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_QEM<PFP>::init()
{
	MAP& m = this->m_map ;

	bool ok = false ;
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin();
		it != this->m_approximators.end() && !ok;
		++it)
	{
		if((*it)->getApproximatedAttributeName() == "position")
		{
			assert((*it)->getType() != A_hQEM || !"Approximator(hQEM) and selector (EdgeSelector_QEM) are not compatible") ;
			assert((*it)->getType() != A_hHalfCollapse || !"Approximator(hHalfCollapse) and selector (EdgeSelector_QEM) are not compatible") ;
			m_positionApproximator = reinterpret_cast<Approximator<PFP, VEC3,EDGE>* >(*it) ;
			ok = true ;
		}
	}

	if(!ok)
		return false ;

	edges.clear() ;

	CellMarker<MAP, VERTEX> vMark(m) ;
	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		if(!vMark.isMarked(d))
		{
			Utils::Quadric<REAL> q ;	// create one quadric
			quadric[d] = q ;	// per vertex
			vMark.mark(d) ;
		}
	}

	DartMarker<MAP> mark(m) ;
	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		if(!mark.isMarked(d))
		{
			Dart d1 = m.phi1(d) ;				// for each triangle,
			Dart d_1 = m.phi_1(d) ;				// initialize the quadric of the triangle
			Utils::Quadric<REAL> q(this->m_position[d], this->m_position[d1], this->m_position[d_1]) ;
			quadric[d] += q ;					// and add the contribution of
			quadric[d1] += q ;					// this quadric to the ones
			quadric[d_1] += q ;					// of the 3 incident vertices
			mark.template markOrbit<FACE>(d) ;
		}
	}

	CellMarker<MAP, EDGE> eMark(m) ;
	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		if(!eMark.isMarked(d))
		{
			initEdgeInfo(d) ;	// init the edges with their optimal position
			eMark.mark(d) ;		// and insert them in the multimap according to their error
		}
	}

	cur = edges.begin() ; // init the current edge to the first one

	return true ;
}

template <typename PFP>
bool EdgeSelector_QEM<PFP>::nextEdge(Dart& d) const
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_QEM<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo *edgeE = &(edgeInfo[d]) ;
	if(edgeE->valid)
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)					// remove all
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi_1(d)]) ;	// the concerned edges
	if(edgeE->valid)
		edges.erase(edgeE->it) ;
									// from the multimap
	Dart dd = m.phi2(d) ;
	if(dd != d)
	{
		edgeE = &(edgeInfo[m.phi1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;

		edgeE = &(edgeInfo[m.phi_1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;
	}

	tmpQ.zero() ;			// compute quadric for the new
	tmpQ += quadric[d] ;	// vertex as the sum of those
	tmpQ += quadric[dd] ;	// of the contracted vertices
}

template <typename PFP>
void EdgeSelector_QEM<PFP>::updateAfterCollapse(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;

	quadric[d2] = tmpQ ;

	Dart vit = d2 ;
	do
	{
		updateEdgeInfo(m.phi1(vit), false) ;			// must recompute some edge infos in the
		if(vit == d2 || vit == dd2)						// neighborhood of the collapsed edge
		{
			initEdgeInfo(vit) ;							// various optimizations are applied here by
														// treating differently :
			Dart vit2 = m.phi12(m.phi1(vit)) ;		// - edges for which the criteria must be recomputed
			Dart stop = m.phi2(vit) ;					// - edges that must be re-embedded
			do											// - edges for which only the collapsibility must be re-tested
			{
				updateEdgeInfo(vit2, false) ;
				updateEdgeInfo(m.phi1(vit2), false) ;
				vit2 = m.phi12(vit2) ;
			} while(vit2 != stop) ;
		}
		else
			updateEdgeInfo(vit, true) ;

		vit = m.phi2_1(vit) ;
	} while(vit != d2) ;

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_QEM<PFP>::updateWithoutCollapse()
{
	EdgeInfo& einfo = edgeInfo[(*cur).second] ;
	einfo.valid = false ;
	edges.erase(einfo.it) ;

	//edges.erase(cur) ;
	cur = edges.begin();
}

template <typename PFP>
void EdgeSelector_QEM<PFP>::initEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo einfo ;
	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
	edgeInfo[d] = einfo ;
}

template <typename PFP>
void EdgeSelector_QEM<PFP>::updateEdgeInfo(Dart d, bool recompute)
{
	MAP& m = this->m_map ;

	EdgeInfo& einfo = edgeInfo[d] ;
	if(recompute)
	{
		if(einfo.valid)
			edges.erase(einfo.it) ;		// remove the edge from the multimap
		if(m.edgeCanCollapse(d))
			computeEdgeInfo(d, einfo) ;
		else
			einfo.valid = false ;
	}
	else
	{
		if(m.edgeCanCollapse(d))
		{								 	// if the edge can be collapsed now
			if(!einfo.valid)				// but it was not before
				computeEdgeInfo(d, einfo) ;
		}
		else
		{								 // if the edge cannot be collapsed now
			if(einfo.valid)				 // and it was before
			{
				edges.erase(einfo.it) ;
				einfo.valid = false ;
			}
		}
	}
}

template <typename PFP>
void EdgeSelector_QEM<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	MAP& m = this->m_map ;

	Dart dd = m.phi2(d) ;

	Utils::Quadric<REAL> quad ;
	quad += quadric[d] ;	// compute the sum of the
	quad += quadric[dd] ;	// two vertices quadrics

	m_positionApproximator->approximate(d) ;

	REAL err = quad(m_positionApproximator->getApprox(d)) ;

	einfo.it = edges.insert(std::make_pair(err, d)) ;
	einfo.valid = true ;
}

/************************************************************************************
 *                            QUADRIC ERROR METRIC (Memoryless version)             *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_QEMml<PFP>::init()
{
	MAP& m = this->m_map ;

	bool ok = false ;
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin();
		it != this->m_approximators.end() && !ok;
		++it)
	{
		if((*it)->getApproximatedAttributeName() == "position")
		{
			assert((*it)->getType() != A_hQEM || !"Approximator(hQEM) and selector (EdgeSelector_QEMml) are not compatible") ;
			assert((*it)->getType() != A_hHalfCollapse || !"Approximator(hHalfCollapse) and selector (EdgeSelector_QEMml) are not compatible") ;
			m_positionApproximator = reinterpret_cast<Approximator<PFP, VEC3,EDGE>* >(*it) ;
			ok = true ;
		}
	}

	if(!ok)
		return false ;

	edges.clear() ;

	CellMarker<MAP, VERTEX> vMark(m) ;
	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		if(!vMark.isMarked(d))
		{
			Utils::Quadric<REAL> q ;	// create one quadric
			quadric[d] = q ;	// per vertex
			vMark.mark(d) ;
		}
	}

	DartMarker<MAP> mark(m) ;
	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		if(!mark.isMarked(d))
		{
			Dart d1 = m.phi1(d) ;				// for each triangle,
			Dart d_1 = m.phi_1(d) ;				// initialize the quadric of the triangle
			Utils::Quadric<REAL> q(this->m_position[d], this->m_position[d1], this->m_position[d_1]) ;
			quadric[d] += q ;					// and add the contribution of
			quadric[d1] += q ;					// this quadric to the ones
			quadric[d_1] += q ;					// of the 3 incident vertices
			mark.template markOrbit<FACE>(d) ;
		}
	}

	CellMarker<MAP, EDGE> eMark(m) ;
	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		if(!eMark.isMarked(d))
		{
			initEdgeInfo(d) ;	// init the edges with their optimal position
			eMark.mark(d) ;		// and insert them in the multimap according to their error
		}
	}

	cur = edges.begin() ; // init the current edge to the first one

	return true ;
}

template <typename PFP>
bool EdgeSelector_QEMml<PFP>::nextEdge(Dart& d) const
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_QEMml<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo *edgeE = &(edgeInfo[d]) ;
	if(edgeE->valid)
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)					// remove all
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi_1(d)]) ;	// the concerned edges
	if(edgeE->valid)
		edges.erase(edgeE->it) ;
									// from the multimap
	Dart dd = m.phi2(d) ;
	if(dd != d)
	{
		edgeE = &(edgeInfo[m.phi1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;

		edgeE = &(edgeInfo[m.phi_1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;
	}
}

/**
 * Update quadric of a vertex
 * Discards quadrics of d and assigns freshly calculated
 * quadrics depending on the actual planes surrounding d
 * @param dart d
 */
template <typename PFP>
void EdgeSelector_QEMml<PFP>::recomputeQuadric(const Dart d, const bool recomputeNeighbors)
{
	Dart dFront,dBack ;
	Dart dInit = d ;

	// Init Front
	dFront = dInit ;

	quadric[d].zero() ;

   	do {
   		// Make step
   		dBack = this->m_map.phi2(dFront) ;
       	dFront = this->m_map.phi2_1(dFront) ;

       	if (dBack != dFront) { // if dFront is no border
           	quadric[d] += Utils::Quadric<REAL>(this->m_position[d],this->m_position[this->m_map.phi2(dFront)],this->m_position[dBack]) ;
       	}
       	if (recomputeNeighbors)
       		recomputeQuadric(dBack, false) ;

    } while(dFront != dInit) ;
}

template <typename PFP>
void EdgeSelector_QEMml<PFP>::updateAfterCollapse(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;

	// for local vertex and neighbors
	recomputeQuadric(d2, true) ;

	Dart vit = d2 ;
	do
	{
		updateEdgeInfo(m.phi1(vit), true) ;			// must recompute some edge infos in the
		if(vit == d2 || vit == dd2)					// neighborhood of the collapsed edge
			initEdgeInfo(vit) ;						// various optimizations are applied here by
		else										// treating differently :
			updateEdgeInfo(vit, true) ;

		Dart vit2 = m.phi12(m.phi1(vit)) ;		// - edges for which the criteria must be recomputed
		Dart stop = m.phi2(vit) ;					// - edges that must be re-embedded
		do											// - edges for which only the collapsibility must be re-tested
		{
			updateEdgeInfo(vit2, true) ;
			updateEdgeInfo(m.phi1(vit2), false) ;
			vit2 = m.phi12(vit2) ;
		} while(vit2 != stop) ;

		vit = m.phi2_1(vit) ;
	} while(vit != d2) ;

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_QEMml<PFP>::updateWithoutCollapse()
{
	EdgeInfo& einfo = edgeInfo[(*cur).second] ;
	einfo.valid = false ;
	edges.erase(einfo.it) ;

	//edges.erase(cur) ;
	cur = edges.begin();
}

template <typename PFP>
void EdgeSelector_QEMml<PFP>::initEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo einfo ;
	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
	edgeInfo[d] = einfo ;
}

template <typename PFP>
void EdgeSelector_QEMml<PFP>::updateEdgeInfo(Dart d, bool recompute)
{
	MAP& m = this->m_map ;

	EdgeInfo& einfo = edgeInfo[d] ;
	if(recompute)
	{
		if(einfo.valid)
			edges.erase(einfo.it) ;		// remove the edge from the multimap
		if(m.edgeCanCollapse(d))
			computeEdgeInfo(d, einfo) ;
		else
			einfo.valid = false ;
	}
	else
	{
		if(m.edgeCanCollapse(d))
		{								 	// if the edge can be collapsed now
			if(!einfo.valid)				// but it was not before
				computeEdgeInfo(d, einfo) ;
		}
		else
		{								 // if the edge cannot be collapsed now
			if(einfo.valid)				 // and it was before
			{
				edges.erase(einfo.it) ;
				einfo.valid = false ;
			}
		}
	}
}

template <typename PFP>
void EdgeSelector_QEMml<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	MAP& m = this->m_map ;

	Dart dd = m.phi2(d) ;

	Utils::Quadric<REAL> quad ;
	quad += quadric[d] ;	// compute the sum of the
	quad += quadric[dd] ;	// two vertices quadrics

	m_positionApproximator->approximate(d) ;

	REAL err = quad(m_positionApproximator->getApprox(d)) ;
	einfo.it = edges.insert(std::make_pair(err, d)) ;
	einfo.valid = true ;
}

/************************************************************************************
 *                            Metric based on Face Normal and Area deviation        *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_NormalArea<PFP>::init()
{
	MAP& m = this->m_map ;

	bool ok = false ;
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin();
		it != this->m_approximators.end() && !ok;
		++it)
	{
		if((*it)->getApproximatedAttributeName() == "position")
		{
//			assert((*it)->getType() == A_MidEdge || (*it)->getType() == A_NormalArea || !"Only MidEdge and NormalArea Approximator are valid") ;
			m_positionApproximator = reinterpret_cast<Approximator<PFP, VEC3,EDGE>* >(*it) ;
			ok = true ;
		}
	}

	if(!ok)
		return false ;

	edges.clear() ;

	TraversorE<MAP> travE(m);
	for(Dart dit = travE.begin() ; dit != travE.end() ; dit = travE.next())
	{
		computeEdgeMatrix(dit);
//		const VEC3 e = Algo::Geometry::vectorOutOfDart<PFP>(m, dit, this->m_position) ;
//		edgeMatrix[dit].identity();
//		edgeMatrix[dit] *= e.norm2();
//		edgeMatrix[dit] -= Geom::transposed_vectors_mult(e,e) ;
	}

	for(Dart dit = travE.begin() ; dit != travE.end() ; dit = travE.next())
	{
		initEdgeInfo(dit) ;	// init "edgeInfo" and "edges"
	}

	cur = edges.begin() ; // init the current edge to the first one

	return true ;
}

template <typename PFP>
bool EdgeSelector_NormalArea<PFP>::nextEdge(Dart& d) const
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_NormalArea<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	assert(!m.isBoundaryEdge(d));

	EdgeInfo* edgeE = &(edgeInfo[d]) ;
	if(edgeE->valid)
	{
		edges.erase(edgeE->it) ;
		edgeE->valid = false;
	}

	edgeE = &(edgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)					// remove all
	{
		edges.erase(edgeE->it) ;
		edgeE->valid = false;
	}

	edgeE = &(edgeInfo[m.phi_1(d)]) ;	// the concerned edges
	if(edgeE->valid)
	{
		edges.erase(edgeE->it) ;
		edgeE->valid = false;
	}
									// from the multimap
	Dart dd = m.phi2(d) ;
	edgeE = &(edgeInfo[m.phi1(dd)]) ;
	if(edgeE->valid)
	{
		edges.erase(edgeE->it) ;
		edgeE->valid = false;
	}

	edgeE = &(edgeInfo[m.phi_1(dd)]) ;
	if(edgeE->valid)
	{
		edges.erase(edgeE->it) ;
		edgeE->valid = false;
	}
}

template <typename PFP>
void EdgeSelector_NormalArea<PFP>::updateAfterCollapse(Dart d2, Dart /*dd2*/)
{
	MAP& m = this->m_map ;

	// update the edge matrices
	Traversor2VE<MAP> te (m,d2);
	for(Dart dit = te.begin() ; dit != te.end() ; dit = te.next())
	{
		computeEdgeMatrix(dit);
	}

	// update the multimap

	Traversor2VVaE<MAP> tv (m,d2);
	CellMarker<MAP, EDGE> eMark (m);

	for(Dart dit = tv.begin() ; dit != tv.end() ; dit = tv.next())
	{
		Traversor2VE<MAP> te2 (m,dit);
		for(Dart dit2 = te2.begin() ; dit2 != te2.end() ; dit2 = te2.next())
		{
			if (!eMark.isMarked(dit2))
			{
				updateEdgeInfo(dit2) ;
				eMark.mark(dit2);
			}
		}
	}

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_NormalArea<PFP>::initEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo einfo ;
	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
	edgeInfo[d] = einfo ;
}

template <typename PFP>
void EdgeSelector_NormalArea<PFP>::updateEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo& einfo = edgeInfo[d] ;

	if(einfo.valid)
		edges.erase(einfo.it) ;		// remove the edge from the multimap

	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
}

template <typename PFP>
void EdgeSelector_NormalArea<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	MAP& m = this->m_map ;

	Dart dd = m.phi2(d);
	Geom::Matrix33f M1; // init zero included
	Geom::Matrix33f M2; // init zero included

	assert(! m.isBoundaryEdge(d));

	Traversor2VF<MAP> td (m,d);
	Dart it = td.begin();
	it = td.next();
	Dart it2 = td.next();
	while( it2 != td.end())
	{
		M1 += edgeMatrix[m.phi1(it)];
		it = it2;
		it2 = td.next();
	}

	Traversor2VF<MAP> tdd (m,dd);
	it = tdd.begin();
	it = tdd.next();
	it2 = tdd.next();
	while( it2 != tdd.end())
	{
		M2 += edgeMatrix[m.phi1(it)];
		it = it2;
		it2 = tdd.next();
	}

	m_positionApproximator->approximate(d) ;
	const VEC3& a = m_positionApproximator->getApprox(d) ;

	const VEC3 av1 = a - this->m_position[d] ;
	const VEC3 av2 = a - this->m_position[dd] ;

	REAL err = av1 * (M1 * av1) + av2 * (M2 * av2);

/*
	// TODO : test to normalize by area
	// TODO : not border-safe
 	REAL area = 0.0;
	Traversor2EEaV<MAP> ta (m,d);
	for (Dart dita = ta.begin(); dita != ta.end(); dita=ta.next())
	{
		area += Algo::Geometry::triangleArea<PFP>(m,dita,this->m_position);
	}
	err /= area ; // résultats sensiblment identiques à ceux sans pris en compte de l'aire.
//	err /= area*area ; // ca favorise la contraction des gros triangles : maillages très in-homogènes et qualité géométrique mauvaise
*/
	einfo.it = edges.insert(std::make_pair(err, d)) ;
	einfo.valid = true ;
}

template <typename PFP>
void EdgeSelector_NormalArea<PFP>::computeEdgeMatrix(Dart d)
{
	const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->m_map, d, this->m_position) ;
	edgeMatrix[d].identity();
	edgeMatrix[d] *= e.norm2();
	edgeMatrix[d] -= Geom::transposed_vectors_mult(e,e) ;
	// TODO : test : try to normalize by area
//	edgeMatrix[d] /= e.norm2(); // pas d'amélioration significative par rapport à la version sans normalisation
//	edgeMatrix[d] /= e.norm2() * e.norm2(); // étonnament bon : sur certains maillages équivalant à la QEMml
}

/************************************************************************************
 *                                   CURVATURE                                      *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_Curvature<PFP>::init()
{
	MAP& m = this->m_map ;

	bool ok = false ;
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin();
		it != this->m_approximators.end() && !ok;
		++it)
	{
		if((*it)->getApproximatedAttributeName() == "position")
		{
			assert((*it)->getType() != A_hQEM || !"Approximator(hQEM) and selector (EdgeSelector_Curvature) are not compatible") ;
			assert((*it)->getType() != A_hHalfCollapse || !"Approximator(hHalfCollapse) and selector (EdgeSelector_Curvature) are not compatible") ;
			m_positionApproximator = reinterpret_cast<Approximator<PFP, VEC3,EDGE>* >(*it) ;
			ok = true ;
		}
	}

	if(!ok)
		return false ;

	edges.clear() ;

	CellMarker<MAP, VERTEX> eMark(m) ;
	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		if(!eMark.isMarked(d))
		{
			initEdgeInfo(d) ;	// init the edges with their optimal position
			eMark.mark(d) ;		// and insert them in the multimap according to their error
		}
	}

	cur = edges.begin() ; // init the current edge to the first one

	return true ;
}

template <typename PFP>
bool EdgeSelector_Curvature<PFP>::nextEdge(Dart& d) const
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_Curvature<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo *edgeE = &(edgeInfo[d]) ;
	if(edgeE->valid)
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)					// remove all
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi_1(d)]) ;	// the concerned edges
	if(edgeE->valid)
		edges.erase(edgeE->it) ;
									// from the multimap
	Dart dd = m.phi2(d) ;
	if(dd != d)
	{
		edgeE = &(edgeInfo[m.phi1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;

		edgeE = &(edgeInfo[m.phi_1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;
	}
}

template <typename PFP>
void EdgeSelector_Curvature<PFP>::updateAfterCollapse(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;

	normal[d2] = Algo::Surface::Geometry::vertexNormal<PFP>(m, d2, this->m_position) ;
	Algo::Surface::Geometry::computeCurvatureVertex_NormalCycles<PFP>(m, d2, radius, this->m_position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal) ;

	Dart vit = d2 ;
	do
	{
		Dart nVert = m.phi1(vit) ;
		normal[nVert] = Algo::Surface::Geometry::vertexNormal<PFP>(m, nVert, this->m_position) ;
		Algo::Surface::Geometry::computeCurvatureVertex_NormalCycles<PFP>(m, nVert, radius, this->m_position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal) ;

		updateEdgeInfo(m.phi1(vit), false) ;			// must recompute some edge infos in the
		if(vit == d2 || vit == dd2)						// neighborhood of the collapsed edge
		{
			initEdgeInfo(vit) ;							// various optimizations are applied here by
														// treating differently :
			Dart vit2 = m.phi12(m.phi1(vit)) ;		// - edges for which the criteria must be recomputed
			Dart stop = m.phi2(vit) ;					// - edges that must be re-embedded
			do											// - edges for which only the collapsibility must be re-tested
			{
				updateEdgeInfo(vit2, false) ;
				updateEdgeInfo(m.phi1(vit2), false) ;
				vit2 = m.phi12(vit2) ;
			} while(vit2 != stop) ;
		}
		else
			updateEdgeInfo(vit, true) ;

		vit = m.phi2_1(vit) ;
	} while(vit != d2) ;

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_Curvature<PFP>::updateWithoutCollapse()
{
	EdgeInfo& einfo = edgeInfo[(*cur).second] ;
	einfo.valid = false ;
	edges.erase(einfo.it) ;

	//edges.erase(cur) ;
	cur = edges.begin();
}

template <typename PFP>
void EdgeSelector_Curvature<PFP>::initEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo einfo ;
	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
	edgeInfo[d] = einfo ;
}

template <typename PFP>
void EdgeSelector_Curvature<PFP>::updateEdgeInfo(Dart d, bool recompute)
{
	MAP& m = this->m_map ;

	EdgeInfo& einfo = edgeInfo[d] ;
	if(recompute)
	{
		if(einfo.valid)
			edges.erase(einfo.it) ;			// remove the edge from the multimap
		if(m.edgeCanCollapse(d))
			computeEdgeInfo(d, einfo) ;
		else
			einfo.valid = false ;
	}
	else
	{
		if(m.edgeCanCollapse(d))
		{									// if the edge can be collapsed now
			if(!einfo.valid)				// but it was not before
				computeEdgeInfo(d, einfo) ;
		}
		else
		{									// if the edge cannot be collapsed now
			if(einfo.valid)					// and it was before
			{
				edges.erase(einfo.it) ;
				einfo.valid = false ;
			}
		}
	}
}

template <typename PFP>
void EdgeSelector_Curvature<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	MAP& m = this->m_map ;

	Dart dd = m.phi2(d) ;

	unsigned int v1 = m.template getEmbedding<VERTEX>(d) ;
	unsigned int v2 = m.template getEmbedding<VERTEX>(dd) ;

	m_positionApproximator->approximate(d) ;

	// temporary edge collapse
	Dart d2 = m.phi2(m.phi_1(d)) ;
	Dart dd2 = m.phi2(m.phi_1(dd)) ;
	m.extractTrianglePair(d) ;
//	unsigned int newV = m.template setOrbitEmbeddingOnNewCell<VERTEX>(d2) ;
	const unsigned int newV = Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(m,d2);
	this->m_position[newV] = m_positionApproximator->getApprox(d) ;

	// compute things on the coarse version of the mesh
	normal[newV] = Algo::Surface::Geometry::vertexNormal<PFP>(m, d2, this->m_position) ;
	Algo::Surface::Geometry::computeCurvatureVertex_NormalCycles<PFP>(m, d2, radius, this->m_position, normal, edgeangle, kmax, kmin, Kmax, Kmin, Knormal) ;

//	VEC3 norm = normal[newV] ;
	REAL mCurv = (kmax[newV] + kmin[newV]) / REAL(2) ;
//	VEC3 cDir1 = Kmax[newV] ;

	// vertex split to reset the initial connectivity and embeddings
	m.insertTrianglePair(d, d2, dd2) ;
//	m.template setOrbitEmbedding<VERTEX>(d, v1) ;
//	m.template setOrbitEmbedding<VERTEX>(dd, v2) ;
	Algo::Topo::setOrbitEmbedding<VERTEX>(m,d,v1);
	Algo::Topo::setOrbitEmbedding<VERTEX>(m,dd,v2);


	REAL err = 0 ;

//	REAL norm_deviation_1 = REAL(1) / fabs(norm * normal[v1]) ;
//	REAL norm_deviation_2 = REAL(1) / fabs(norm * normal[v2]) ;
//	err += norm_deviation_1 + norm_deviation_2 ;

	REAL mCurv_deviation_1 = fabs(mCurv - (kmax[v1] + kmin[v1] / REAL(2))) ;
	REAL mCurv_deviation_2 = fabs(mCurv - (kmax[v2] + kmin[v2] / REAL(2))) ;
	err += mCurv_deviation_1 + mCurv_deviation_2 ;

//	REAL cDir1_deviation_1 = REAL(1) / fabs(cDir1 * Kmax[v1]) ;
//	REAL cDir1_deviation_2 = REAL(1) / fabs(cDir1 * Kmax[v2]) ;
//	err += cDir1_deviation_1 + cDir1_deviation_2 ;

	einfo.it = edges.insert(std::make_pair(err, d)) ;
	einfo.valid = true ;
}

/************************************************************************************
 *                            CURVATURE TENSOR                                      *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_CurvatureTensor<PFP>::init()
{
	MAP& m = this->m_map ;

	bool ok = false ;
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin();
		it != this->m_approximators.end() && !ok;
		++it)
	{
		if((*it)->getApproximatedAttributeName() == "position")
		{
			assert((*it)->getType() == A_MidEdge || !"Only MidEdge Approximator is valid") ;
			m_positionApproximator = reinterpret_cast<Approximator<PFP, VEC3,EDGE>* >(*it) ;
			ok = true ;
		}
	}

	if(!ok)
		return false ;

	edges.clear() ;

	TraversorE<MAP> travE(m);
//	for(Dart dit = travE.begin() ; dit != travE.end() ; dit = travE.next())
//	{
//		computeEdgeMatrix(dit);
//	}

	for(Dart dit = travE.begin() ; dit != travE.end() ; dit = travE.next())
	{
		initEdgeInfo(dit) ;	// init "edgeInfo" and "edges"
	}

	cur = edges.begin() ; // init the current edge to the first one

	return true ;
}

template <typename PFP>
bool EdgeSelector_CurvatureTensor<PFP>::nextEdge(Dart& d) const
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_CurvatureTensor<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	assert(!m.isBoundaryEdge(d));

	EdgeInfo* edgeE = &(edgeInfo[d]) ;
	if(edgeE->valid)
	{
		edges.erase(edgeE->it) ;
		edgeE->valid = false;
	}

	edgeE = &(edgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)					// remove all
	{
		edges.erase(edgeE->it) ;
		edgeE->valid = false;
	}

	edgeE = &(edgeInfo[m.phi_1(d)]) ;	// the concerned edges
	if(edgeE->valid)
	{
		edges.erase(edgeE->it) ;
		edgeE->valid = false;
	}

									// from the multimap
	Dart dd = m.phi2(d) ;
	edgeE = &(edgeInfo[m.phi1(dd)]) ;
	if(edgeE->valid)
	{
		edges.erase(edgeE->it) ;
		edgeE->valid = false;
	}

	edgeE = &(edgeInfo[m.phi_1(dd)]) ;
	if(edgeE->valid)
	{
		edges.erase(edgeE->it) ;
		edgeE->valid = false;
	}
}

template <typename PFP>
void EdgeSelector_CurvatureTensor<PFP>::updateAfterCollapse(Dart d2, Dart /*dd2*/)
{
	MAP& m = this->m_map ;

	CellMarkerStore<MAP, EDGE> eMark (m);

	// update edge angles
	Traversor2VF<MAP> tf (m,d2);
	for(Dart dit = tf.begin(); dit != tf.end(); dit = tf.next())
	{
		Traversor2FE<MAP> te (m,dit);
		for(Dart dit2 = te.begin(); dit2 != te.end(); dit2=te.next())
		{
			if (!eMark.isMarked(dit2))
			{
				edgeangle[dit2] = Algo::Surface::Geometry::computeAngleBetweenNormalsOnEdge<PFP>(m, dit2, this->m_position) ;
				eMark.mark(dit2);
			}
		}
	}

	// update the multimap

	Traversor2VVaE<MAP> tv (m,d2);
	eMark.unmarkAll();

	for(Dart dit = tv.begin() ; dit != tv.end() ; dit = tv.next())
	{
		Traversor2VE<MAP> te2 (m,dit);
		for(Dart dit2 = te2.begin() ; dit2 != te2.end() ; dit2 = te2.next())
		{
			if (!eMark.isMarked(dit2))
			{
				updateEdgeInfo(dit2) ;
				eMark.mark(dit2);
			}
		}
	}

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_CurvatureTensor<PFP>::initEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo einfo ;
	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
	edgeInfo[d] = einfo ;
}

template <typename PFP>
void EdgeSelector_CurvatureTensor<PFP>::updateEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo& einfo = edgeInfo[d] ;

	if(einfo.valid)
		edges.erase(einfo.it) ;		// remove the edge from the multimap

	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
}

template <typename PFP>
void EdgeSelector_CurvatureTensor<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	typedef Geom::Matrix<3,3,REAL> MATRIX;
//	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3,Eigen::RowMajor> E_MATRIX;

	MAP& m = this->m_map ;

	Dart dd = m.phi2(d) ;

	unsigned int v1 = m.template getEmbedding<VERTEX>(d) ;
	unsigned int v2 = m.template getEmbedding<VERTEX>(dd) ;

	m_positionApproximator->approximate(d) ;

	// compute tensor before collapse
	MATRIX tens1;
	Algo::Surface::Selection::Collector_OneRing_AroundEdge<PFP> col1 (m);
	col1.collectAll(d);
	col1.computeNormalCyclesTensor(this->m_position,edgeangle,tens1); // edgeangle is up to date here
	tens1 *= col1.computeArea(this->m_position); // mean tensor * area = integral of the tensor
	Algo::Surface::Geometry::normalCycles_SortTensor<PFP>(tens1);

	// temporary edge collapse
	Dart d2 = m.phi2(m.phi_1(d)) ;
	Dart dd2 = m.phi2(m.phi_1(dd)) ;
	m.extractTrianglePair(d) ;
//	const unsigned int newV = m.template setOrbitEmbeddingOnNewCell<VERTEX>(d2) ;
	const unsigned int newV = Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(m,d2);
	this->m_position[newV] = m_positionApproximator->getApprox(d) ;

	// compute tensor after collapse
	MATRIX tens2;
	Algo::Surface::Selection::Collector_OneRing<PFP> col2 (m);
	col2.collectAll(d);
	col2.computeNormalCyclesTensor(this->m_position,tens2); // edgeangle is not up to date here
	tens2 *= col2.computeArea(this->m_position); // mean tensor * area = integral of the tensor
	Algo::Surface::Geometry::normalCycles_SortTensor<PFP>(tens2);

	// vertex split to reset the initial connectivity and embeddings
	m.insertTrianglePair(d, d2, dd2) ;
//	m.template setOrbitEmbedding<VERTEX>(d, v1) ;
//	m.template setOrbitEmbedding<VERTEX>(dd, v2) ;
	Algo::Topo::setOrbitEmbedding<VERTEX>(m,d,v1);
	Algo::Topo::setOrbitEmbedding<VERTEX>(m,dd,v2);

	// compute err from the tensors
	tens1 -= tens2;

	Eigen::SelfAdjointEigenSolver<E_MATRIX> solver (Utils::convertRef<E_MATRIX>(tens1),Eigen::EigenvaluesOnly);
	const VEC3& e_val = Utils::convertRef<VEC3>(solver.eigenvalues());

	REAL err = std::max(std::max(fabs(e_val[0]), fabs(e_val[1])) , fabs(e_val[2])) ;

//	if (v1 % 5000 == 0) CGoGNout << e_val << CGoGNendl << err << CGoGNendl ;

	// update the priority queue and edgeinfo
	einfo.it = edges.insert(std::make_pair(err, d)) ;
	einfo.valid = true ;
}

/************************************************************************************
 *                                  MIN DETAIL                                      *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_MinDetail<PFP>::init()
{
	MAP& m = this->m_map ;

	bool ok = false ;
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin();
		it != this->m_approximators.end() && !ok;
		++it)
	{
		if((*it)->getApproximatedAttributeName() == "position" && (*it)->getPredictor())
		{
			assert((*it)->getType() != A_hQEM || !"Approximator(hQEM) and selector (EdgeSelector_MinDetail) are not compatible") ;
			assert((*it)->getType() != A_hHalfCollapse || !"Approximator(hHalfCollapse) and selector (EdgeSelector_MinDetail) are not compatible") ;
			m_positionApproximator = reinterpret_cast<Approximator<PFP, VEC3,EDGE>* >(*it) ;
			ok = true ;
		}
	}

	if(!ok)
		return false ;

	edges.clear() ;

	TraversorE<MAP> travE(m);
	for(Dart dit = travE.begin() ; dit != travE.end() ; dit = travE.next())
	{
		initEdgeInfo(dit);
	}

	cur = edges.begin() ; // init the current edge to the first one

	return true ;
}

template <typename PFP>
bool EdgeSelector_MinDetail<PFP>::nextEdge(Dart& d) const
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_MinDetail<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo *edgeE = &(edgeInfo[d]) ;
	if(edgeE->valid)
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)					// remove all
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi_1(d)]) ;	// the concerned edges
	if(edgeE->valid)
		edges.erase(edgeE->it) ;
									// from the multimap
	Dart dd = m.phi2(d) ;
	if(dd != d)
	{
		edgeE = &(edgeInfo[m.phi1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;

		edgeE = &(edgeInfo[m.phi_1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;
	}
}

template <typename PFP>
void EdgeSelector_MinDetail<PFP>::updateAfterCollapse(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;

	Dart vit = d2 ;
	do
	{
		updateEdgeInfo(m.phi1(vit), false) ;		// must recompute some edge infos in the
		if(vit == d2 || vit == dd2)					// neighborhood of the collapsed edge
		{
			initEdgeInfo(vit) ;						// various optimizations are applied here by
													// treating differently :
			Dart vit2 = m.phi12(m.phi1(vit)) ;		// - edges for which the criteria must be recomputed
			Dart stop = m.phi2(vit) ;				// - edges that must be re-embedded
			do										// - edges for which only the collapsibility must be re-tested
			{
				updateEdgeInfo(vit2, false) ;
				updateEdgeInfo(m.phi1(vit2), false) ;
				vit2 = m.phi12(vit2) ;
			} while(vit2 != stop) ;
		}
		else
			updateEdgeInfo(vit, true) ;

		vit = m.phi2_1(vit) ;
	} while(vit != d2) ;

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_MinDetail<PFP>::updateWithoutCollapse()
{
	EdgeInfo& einfo = edgeInfo[(*cur).second] ;
	einfo.valid = false ;
	edges.erase(einfo.it) ;

	//edges.erase(cur) ;
	cur = edges.begin();
}

template <typename PFP>
void EdgeSelector_MinDetail<PFP>::initEdgeInfo(Dart d)
{
	typename PFP::MAP& m = this->m_map ;
	EdgeInfo einfo ;
	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
	edgeInfo[d] = einfo ;
}

template <typename PFP>
void EdgeSelector_MinDetail<PFP>::updateEdgeInfo(Dart d, bool recompute)
{
	typename PFP::MAP& m = this->m_map ;
	EdgeInfo& einfo = edgeInfo[d] ;
	if(recompute)
	{
		if(einfo.valid)
			edges.erase(einfo.it) ;			// remove the edge from the multimap
		if(m.edgeCanCollapse(d))
			computeEdgeInfo(d, einfo) ;
		else
			einfo.valid = false ;
	}
	else
	{
		if(m.edgeCanCollapse(d))
		{									// if the edge can be collapsed now
			if(!einfo.valid)				// but it was not before
				computeEdgeInfo(d, einfo) ;
		}
		else
		{									// if the edge cannot be collapsed now
			if(einfo.valid)					// and it was before
			{
				edges.erase(einfo.it) ;
				einfo.valid = false ;
			}
		}
	}
}

template <typename PFP>
void EdgeSelector_MinDetail<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	// Dart dd = this->m_map.phi2(d) ;
	REAL err = REAL(0) ;

//	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin();
//		it != this->m_approximators.end();
//		++it)
//	{
//		if((*it)->getPredictor())
//		{
//			(*it)->approximate(d) ;
//			err += (*it)->detailMagnitude(d) ;
//		}
//	}

	m_positionApproximator->approximate(d) ;
	err = m_positionApproximator->getDetail(d).norm2() ;

	einfo.it = edges.insert(std::make_pair(err, d)) ;
	einfo.valid = true ;
}

/************************************************************************************
 *                         EDGESELECTOR COLOR PER VERTEX                            *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_ColorNaive<PFP>::init()
{
	MAP& m = this->m_map ;

	// Verify availability of required approximators
	unsigned int ok = 0 ;
	for (unsigned int approxindex = 0 ; approxindex < this->m_approximators.size() ; ++approxindex)
	{
		assert(this->m_approximators[approxindex]->getType() != A_hQEM || !"Approximator(hQEM) and selector (ColorNaive) are not compatible") ;
		assert(this->m_approximators[approxindex]->getType() != A_hHalfCollapse || !"Approximator(hHalfCollapse) and selector (ColorNaive) are not compatible") ;

		bool saved = false ;
		for (unsigned int attrindex = 0 ; attrindex < this->m_approximators[approxindex]->getNbApproximated() ; ++ attrindex)
		{
			// constraint : 2 approximators in specific order
			if(ok == 0 && this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "position")
			{
				++ok ;
				m_approxindex_pos = approxindex ;
				m_attrindex_pos = attrindex ;
				m_pos = this->m_position ;
				if (!saved)
				{
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3,EDGE>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
			else if(ok == 1 && this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "color")
			{
				++ok ;
				m_approxindex_color = approxindex ;
				m_attrindex_color = attrindex ;
				m_color = m.template getAttribute<typename PFP::VEC3, VERTEX, MAP>("color") ;
				assert(m_color.isValid() || !"EdgeSelector_ColorNaive: color attribute is not valid") ;
				if (!saved)
				{
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3,EDGE>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
		}
	}

	if(ok != 2)
		return false ;

	TraversorV<MAP> travV(m);
	for(Dart dit = travV.begin() ; dit != travV.end() ; dit = travV.next())
	{
		Utils::Quadric<REAL> q ;		// create one quadric
		m_quadric[dit] = q ;		// per vertex
	}

	// Compute quadric per vertex
	TraversorF<MAP> travF(m) ;
	for(Dart dit = travF.begin() ; dit != travF.end() ; dit = travF.next()) // init QEM quadrics
	{
		Dart d1 = m.phi1(dit) ;					// for each triangle,
		Dart d_1 = m.phi_1(dit) ;					// initialize the quadric of the triangle
		Utils::Quadric<REAL> q(this->m_position[dit], this->m_position[d1], this->m_position[d_1]) ;
		m_quadric[dit] += q ;						// and add the contribution of
		m_quadric[d1] += q ;						// this quadric to the ones
		m_quadric[d_1] += q ;						// of the 3 incident vertices
	}

	TraversorE<MAP> travE(m);
	for(Dart dit = travE.begin() ; dit != travE.end() ; dit = travE.next())
	{
		initEdgeInfo(dit) ; // init the edges with their optimal position
							// and insert them in the multimap according to their error
	}

	cur = edges.begin() ; // init the current edge to the first one

	return true ;
}

template <typename PFP>
bool EdgeSelector_ColorNaive<PFP>::nextEdge(Dart& d) const
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_ColorNaive<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo *edgeE = &(edgeInfo[d]) ;
	if(edgeE->valid)
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)						// remove all
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi_1(d)]) ;	// the edges that will disappear
	if(edgeE->valid)
		edges.erase(edgeE->it) ;
										// from the multimap
	Dart dd = m.phi2(d) ;
	if(dd != d)
	{
		edgeE = &(edgeInfo[m.phi1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;

		edgeE = &(edgeInfo[m.phi_1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;
	}
}

template <typename PFP>
void EdgeSelector_ColorNaive<PFP>::recomputeQuadric(const Dart d, const bool recomputeNeighbors)
{
	Dart dFront,dBack ;
	Dart dInit = d ;

	// Init Front
	dFront = dInit ;

	m_quadric[d].zero() ;

   	do
   	{
   		// Make step
   		dBack = this->m_map.phi2(dFront) ;
       	dFront = this->m_map.phi2_1(dFront) ;

       	if (dBack != dFront)
       	{ // if dFront is no border
       		m_quadric[d] += Utils::Quadric<REAL>(this->m_position[d],this->m_position[this->m_map.phi2(dFront)],this->m_position[dBack]) ;
       	}
       	if (recomputeNeighbors)
       		recomputeQuadric(dBack, false) ;

    } while(dFront != dInit) ;
}

template <typename PFP>
void EdgeSelector_ColorNaive<PFP>::updateAfterCollapse(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;

	recomputeQuadric(d2, true) ;

	Dart vit = d2 ;
	do
	{
		updateEdgeInfo(m.phi1(vit), true) ;			// must recompute some edge infos in the
		if(vit == d2 || vit == dd2)					// neighborhood of the collapsed edge
			initEdgeInfo(vit) ;						// various optimizations are applied here by
		else										// treating differently :
			updateEdgeInfo(vit, true) ;

		Dart vit2 = m.phi12(m.phi1(vit)) ;		// - edges for which the criteria must be recomputed
		Dart stop = m.phi2(vit) ;					// - edges that must be re-embedded
		do											// - edges for which only the collapsibility must be re-tested
		{
			updateEdgeInfo(vit2, true) ;
			updateEdgeInfo(m.phi1(vit2), false) ;
			vit2 = m.phi12(vit2) ;
		} while(vit2 != stop) ;

		vit = m.phi2_1(vit) ;
	} while(vit != d2) ;

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_ColorNaive<PFP>::initEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo einfo ;
	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
	edgeInfo[d] = einfo ;
}

template <typename PFP>
void EdgeSelector_ColorNaive<PFP>::updateEdgeInfo(Dart d, bool recompute)
{
	MAP& m = this->m_map ;

	EdgeInfo& einfo = edgeInfo[d] ;
	if(recompute)
	{
		if(einfo.valid)
			edges.erase(einfo.it) ;		// remove the edge from the multimap
		if(m.edgeCanCollapse(d))
			computeEdgeInfo(d, einfo) ;
		else
			einfo.valid = false ;
	}
	else
	{
		if(m.edgeCanCollapse(d))
		{								 	// if the edge can be collapsed now
			if(!einfo.valid)				// but it was not before
				computeEdgeInfo(d, einfo) ;
		}
		else
		{								 // if the edge cannot be collapsed now
			if(einfo.valid)				 // and it was before
			{
				edges.erase(einfo.it) ;
				einfo.valid = false ;
			}
		}
	}
}

template <typename PFP>
void EdgeSelector_ColorNaive<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	MAP& m = this->m_map ;

	Dart dd = m.phi1(d) ;

	// New position
	Utils::Quadric<REAL> quad ;
	quad += m_quadric[d] ;	// compute the sum of the
	quad += m_quadric[dd] ;	// two vertices quadrics

	// compute all approximated attributes
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin() ;
			it != this->m_approximators.end() ;
			++it)
	{
		(*it)->approximate(d) ;
	}

	// get pos
	const VEC3& newPos = this->m_approx[m_approxindex_pos]->getApprox(d,m_attrindex_pos) ; // get newPos
	// get col
	const VEC3& newCol = this->m_approx[m_approxindex_color]->getApprox(d,m_attrindex_color) ; // get newPos

	// compute error
	VEC3 colDiff1 = newCol ;
	VEC3 colDiff2 = newCol ;
	colDiff1 -= m_color[d] ;
	colDiff2 -= m_color[dd] ;
	const VEC3& colDiff = colDiff1 + colDiff2 ;

	// sum of QEM metric and squared difference between new color and old colors
	REAL err = quad(newPos) + colDiff.norm() ;

	einfo.it = edges.insert(std::make_pair(err, d)) ;
	einfo.valid = true ;
}

/************************************************************************************
 *                         EDGESELECTOR GEOM+COL OPT GRADIENT                       *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_GeomColOptGradient<PFP>::init()
{
	MAP& m = this->m_map ;

	// Verify availability of required approximators
	unsigned int ok = 0 ;
	for (unsigned int approxindex = 0 ; approxindex < this->m_approximators.size() ; ++approxindex)
	{
		bool saved = false ;
		for (unsigned int attrindex = 0 ; attrindex < this->m_approximators[approxindex]->getNbApproximated() ; ++ attrindex)
		{
			// constraint : 2 approximators in specific order
			if(ok == 0 && this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "position")
			{
				++ok ;
				m_approxindex_pos = approxindex ;
				m_attrindex_pos = attrindex ;
				m_pos = this->m_position ;
				if (!saved)
				{
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, EDGE>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
			else if(ok == 1 && this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "color")
			{
				++ok ;
				m_approxindex_color = approxindex ;
				m_attrindex_color = attrindex ;
				m_color = m.template getAttribute<VEC3, VERTEX, MAP>("color") ;
				assert(m_color.isValid() || !"EdgeSelector_GeomColOptGradient: color attribute is not valid") ;
				if (!saved)
				{
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, EDGE>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
		}
	}

	if(ok != 2)
		return false ;

	TraversorV<MAP> travV(m);
	for(Dart dit = travV.begin() ; dit != travV.end() ; dit = travV.next())
	{
		Utils::Quadric<REAL> q ;	// create one quadric
		m_quadric[dit] = q ;		// per vertex
	}

	// Compute quadric per vertex
	TraversorF<MAP> travF(m) ;
	for(Dart dit = travF.begin() ; dit != travF.end() ; dit = travF.next()) // init QEM quadrics
	{
		Dart d1 = m.phi1(dit) ;					// for each triangle,
		Dart d_1 = m.phi_1(dit) ;					// initialize the quadric of the triangle
		Utils::Quadric<REAL> q(this->m_position[dit], this->m_position[d1], this->m_position[d_1]) ;
		m_quadric[dit] += q ;						// and add the contribution of
		m_quadric[d1] += q ;						// this quadric to the ones
		m_quadric[d_1] += q ;						// of the 3 incident vertices
	}

	TraversorE<MAP> travE(m);
	for(Dart dit = travE.begin() ; dit != travE.end() ; dit = travE.next())
	{
		initEdgeInfo(dit) ; // init the edges with their optimal position
							// and insert them in the multimap according to their error
	}

	cur = edges.begin() ; // init the current edge to the first one

	return true ;
}

template <typename PFP>
bool EdgeSelector_GeomColOptGradient<PFP>::nextEdge(Dart& d) const
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_GeomColOptGradient<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	const Dart& v0 = d ;
	const Dart& v1 = m.phi2(d) ;

	// remove all the edges that will disappear from the multimap
	// namely : all edges adjacent to a vertex which is adjacent
	// to either v0 or v1

	// collect vertices (1-ring)
	std::vector<Dart> vertices ;

	CellMarker<MAP, VERTEX> cvm(m) ;
	Traversor2VVaE<MAP> tv0(m,v0) ;
	for (Dart v = tv0.begin() ; v != tv0.end() ; v = tv0.next())
	{
		if (!cvm.isMarked(v))
		{
			vertices.push_back(v) ;
			cvm.mark(v) ;
		}
	}
	Traversor2VVaE<MAP> tv1(m,v1) ;
	for (Dart v = tv1.begin() ; v != tv1.end() ; v = tv1.next())
	{
		if (!cvm.isMarked(v))
		{
			vertices.push_back(v) ;
			cvm.mark(v) ;
		}
	}

	// apply to all adjacent edges (2-ring w/o border)
	CellMarker<MAP, EDGE> cem(m) ;
	for (std::vector<Dart>::const_iterator it = vertices.begin() ; it != vertices.end() ; ++it)
	{
		const Dart& v = *it ;
		Traversor2VE<MAP> te(m,v) ;
		for (Dart e = te.begin() ; e != te.end() ; e = te.next())
		{
			if (!cem.isMarked(e))
			{
				if(edgeInfo[e].valid)
				{
					edges.erase(edgeInfo[e].it) ;
					edgeInfo[e].valid = false ;
				}

				cem.mark(e) ;
			}
		}
	}
}

template <typename PFP>
void EdgeSelector_GeomColOptGradient<PFP>::recomputeQuadric(const Dart d, const bool recomputeNeighbors)
{
	Dart dFront,dBack ;
	Dart dInit = d ;

	// Init Front
	dFront = dInit ;

	m_quadric[d].zero() ;

	do
	{
		// Make step
		dBack = this->m_map.phi2(dFront) ;
		dFront = this->m_map.phi2_1(dFront) ;

		if (dBack != dFront)
		{ // if dFront is no border
			m_quadric[d] += Utils::Quadric<REAL>(this->m_position[d],this->m_position[this->m_map.phi2(dFront)],this->m_position[dBack]) ;
		}
		if (recomputeNeighbors)
			recomputeQuadric(dBack, false) ;

	} while(dFront != dInit) ;
}

template <typename PFP>
void EdgeSelector_GeomColOptGradient<PFP>::updateAfterCollapse(Dart d2, Dart /*dd2*/)
{
	MAP& m = this->m_map ;

	// update quadrics
	recomputeQuadric(d2, true) ;

	// update the multimap
	Traversor2VVaE<MAP> tv (m,d2);
	CellMarker<MAP, EDGE> eMark (m);

	for(Dart dit = tv.begin() ; dit != tv.end() ; dit = tv.next())
	{
		Traversor2VE<MAP> te2 (m,dit);
		for(Dart dit2 = te2.begin() ; dit2 != te2.end() ; dit2 = te2.next())
		{
			if (!eMark.isMarked(dit2))
			{
				updateEdgeInfo(dit2) ;
				eMark.mark(dit2);
			}
		}
	}

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_GeomColOptGradient<PFP>::initEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo einfo ;
	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
	edgeInfo[d] = einfo ;
}

template <typename PFP>
void EdgeSelector_GeomColOptGradient<PFP>::updateEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo& einfo = edgeInfo[d] ;

	if(einfo.valid)
		edges.erase(einfo.it) ;		// remove the edge from the multimap

	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
}

template <typename PFP>
void EdgeSelector_GeomColOptGradient<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	MAP& m = this->m_map ;

	Dart dd = m.phi1(d) ;

	// New position
	Utils::Quadric<REAL> quad ;
	quad += m_quadric[d] ;	// compute the sum of the
	quad += m_quadric[dd] ;	// two vertices quadrics

	// compute all approximated attributes
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin() ;
			it != this->m_approximators.end() ;
			++it)
	{
		(*it)->approximate(d) ;
	}

	// get pos
	const VEC3& newPos = this->m_approx[m_approxindex_pos]->getApprox(d,m_attrindex_pos) ; // get newPos
	// get col
	const VEC3& newCol = this->m_approx[m_approxindex_color]->getApprox(d,m_attrindex_color) ; // get newPos

	// sum of QEM metric and color gradient metric
	const REAL t = 0.01 ;
	const REAL err = t*quad(newPos) + (1-t)*(computeEdgeGradientColorError(d,newPos,newCol) + computeEdgeGradientColorError(m.phi2(d),newPos,newCol)).norm() / sqrt(3) ;

	einfo.it = edges.insert(std::make_pair(err, d)) ;
	einfo.valid = true ;
}

template <typename PFP>
typename PFP::VEC3
EdgeSelector_GeomColOptGradient<PFP>::computeEdgeGradientColorError(const Dart& v0, const VEC3& P, const VEC3& newCol)
{
	MAP& m = this->m_map ;

	Traversor2VF<MAP> tf(m,v0) ; // all faces around vertex v0

	const VEC3& P0 = m_pos[v0] ;
	const VEC3& c0 = m_color[v0] ;
	const VEC3 d = P - P0 ; // displacement vector

	VEC3 count ;
	REAL areaSum = 0 ;
	for (Dart fi = tf.begin() ; fi != tf.end() ; fi = tf.next()) // foreach "blue" face
	{
		// get the data
		const Dart& vi = m.phi1(fi) ;
		const Dart& vj = m.phi_1(fi) ;
		const VEC3& Pi = this->m_position[vi] ;
		const VEC3& Pj = this->m_position[vj] ;
		const VEC3& ci = m_color[vi] ;
		const VEC3& cj = m_color[vj] ;

		// utils
		const VEC3 ei = P0 - Pj ;
		const VEC3 ej = Pi - P0 ;
		//const VEC3 e0 = Pj - Pi ;

		const REAL areaIJ0sq = (ei ^ ej).norm2() ;
		const REAL areaIJ0 = sqrt(areaIJ0sq)/2. ;
		areaSum += areaIJ0 ;

		// per-channel treatment
		for (unsigned int c = 0 ; c < 3 ;  ++c)
		{
			// color gradient for channel i
			VEC3 grad = (ei.norm2()*(ci[c]-newCol[c]) + (ei*ej)*(cj[c]-newCol[c]))*ej ;
			grad -= (ej.norm2()*(cj[c]-newCol[c]) + (ei*ej)*(ci[c]-newCol[c]))*ei ;
			const REAL denom = areaIJ0sq ;
			if (denom < 1e-9) // case flat triangles
				grad = VEC3(0,0,0) ;
			else
				grad /= denom ;

			// displacement error for channel i
			const REAL displacementE = areaIJ0 * pow(d*grad,2) ; // area x <disp,grad>

			count[c] += displacementE ;
		}
	}

	const VEC3 colDiff = newCol - c0 ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
		count[c] += pow(colDiff[c],2) * areaSum ;

	return count ;
}

/************************************************************************************
 *                         EDGESELECTOR QEMext for Color                            *
 ************************************************************************************/
template <typename PFP>
bool EdgeSelector_QEMextColor<PFP>::init()
{
	MAP& m = this->m_map ;

	// Verify availability of required approximators
	unsigned int ok = 0 ;
	for (unsigned int approxindex = 0 ; approxindex < this->m_approximators.size() ; ++approxindex)
	{
		assert(this->m_approximators[approxindex]->getType() != A_hQEM || !"Approximator(hQEM) and selector (EdgeSelector_QEMextColor) are not compatible") ;
		assert(this->m_approximators[approxindex]->getType() != A_hHalfCollapse || !"Approximator(hHalfCollapse) and selector (EdgeSelector_QEMextColor) are not compatible") ;

		bool saved = false ;
		for (unsigned int attrindex = 0 ; attrindex < this->m_approximators[approxindex]->getNbApproximated() ; ++ attrindex)
		{
			// constraint : 2 approximators in specific order
			if(ok == 0 && this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "position")
			{
				++ok ;
				m_approxindex_pos = approxindex ;
				m_attrindex_pos = attrindex ;
				m_pos = this->m_position ;
				if (!saved)
				{
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, EDGE>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
			else if(ok == 1 && this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "color")
			{
				++ok ;
				m_approxindex_color = approxindex ;
				m_attrindex_color = attrindex ;
				m_color = m.template getAttribute<VEC3, VERTEX, MAP>("color") ;
				assert(m_color.isValid() || !"EdgeSelector_QEMextColor: color attribute is not valid") ;
				if (!saved)
				{
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, EDGE>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
		}
	}

	if(ok != 2)
		return false ;

	TraversorV<MAP> travV(m);
	for(Dart dit = travV.begin() ; dit != travV.end() ; dit = travV.next())
	{
		Utils::QuadricNd<REAL,6> q ;		// create one quadric
		m_quadric[dit] = q ;		// per vertex
	}

	// Compute quadric per vertex
	TraversorF<MAP> travF(m) ;
	for(Dart dit = travF.begin() ; dit != travF.end() ; dit = travF.next()) // init QEM quadrics
	{
		Dart d1 = m.phi1(dit) ;					// for each triangle,
		Dart d_1 = m.phi_1(dit) ;				// initialize the quadric of the triangle

   		VEC6 p0, p1, p2 ;
   		for (unsigned int i = 0 ; i < 3 ; ++i)
   		{
   			p0[i] = this->m_position[dit][i] ;
   			p0[i+3] = this->m_color[dit][i] ;
   			p1[i] = this->m_position[d1][i] ;
   			p1[i+3] = this->m_color[d1][i] ;
   			p2[i] = this->m_position[d_1][i] ;
   			p2[i+3] = this->m_color[d_1][i] ;
   		}
		Utils::QuadricNd<REAL,6> q(p0,p1,p2) ;
		m_quadric[dit] += q ;						// and add the contribution of
		m_quadric[d1] += q ;						// this quadric to the ones
		m_quadric[d_1] += q ;						// of the 3 incident vertices
	}

	edges.clear() ;

	TraversorE<MAP> travE(m);
	for(Dart dit = travE.begin() ; dit != travE.end() ; dit = travE.next())
	{
		initEdgeInfo(dit) ; // init the edges with their optimal position
							// and insert them in the multimap according to their error
	}

	cur = edges.begin() ; // init the current edge to the first one

	return true ;
}

template <typename PFP>
bool EdgeSelector_QEMextColor<PFP>::nextEdge(Dart& d) const
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_QEMextColor<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo *edgeE = &(edgeInfo[d]) ;
	if(edgeE->valid)
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)					// remove all
		edges.erase(edgeE->it) ;

	edgeE = &(edgeInfo[m.phi_1(d)]) ;	// the edges that will disappear
	if(edgeE->valid)
		edges.erase(edgeE->it) ;
										// from the multimap
	Dart dd = m.phi2(d) ;
	if(dd != d)
	{
		edgeE = &(edgeInfo[m.phi1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;

		edgeE = &(edgeInfo[m.phi_1(dd)]) ;
		if(edgeE->valid)
			edges.erase(edgeE->it) ;
	}
}

template <typename PFP>
void EdgeSelector_QEMextColor<PFP>::recomputeQuadric(const Dart d, const bool recomputeNeighbors)
{
	Dart dFront,dBack ;
	Dart dInit = d ;

	// Init Front
	dFront = dInit ;

	m_quadric[d].zero() ;

   	do
   	{
   		// Make step
   		dBack = this->m_map.phi2(dFront) ;
       	dFront = this->m_map.phi2_1(dFront) ;

       	if (dBack != dFront)
       	{ // if dFront is no border
   			Dart d2 = this->m_map.phi2(dFront) ;

       		VEC6 p0, p1, p2 ;
       		for (unsigned int i = 0 ; i < 3 ; ++i)
       		{
       			p0[i] = this->m_position[d][i] ;
       			p0[i+3] = this->m_color[d][i] ;
       			p1[i] = this->m_position[d2][i] ;
       			p1[i+3] = this->m_color[d2][i] ;
       			p2[i] = this->m_position[dBack][i] ;
       			p2[i+3] = this->m_color[dBack][i] ;
       		}
       		m_quadric[d] += Utils::QuadricNd<REAL,6>(p0,p1,p2) ;
       	}
       	if (recomputeNeighbors)
       		recomputeQuadric(dBack, false) ;

    } while(dFront != dInit) ;
}

template <typename PFP>
void EdgeSelector_QEMextColor<PFP>::updateAfterCollapse(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;

	recomputeQuadric(d2, true) ;

	Dart vit = d2 ;
	do
	{
		updateEdgeInfo(m.phi1(vit), true) ;		// must recompute some edge infos in the
		if(vit == d2 || vit == dd2)				// neighborhood of the collapsed edge
			initEdgeInfo(vit) ;					// various optimizations are applied here by
		else									// treating differently :
			updateEdgeInfo(vit, true) ;

		Dart vit2 = m.phi12(m.phi1(vit)) ;		// - edges for which the criteria must be recomputed
		Dart stop = m.phi2(vit) ;				// - edges that must be re-embedded
		do										// - edges for which only the collapsibility must be re-tested
		{
			updateEdgeInfo(vit2, true) ;
			updateEdgeInfo(m.phi1(vit2), false) ;
			vit2 = m.phi12(vit2) ;
		} while(vit2 != stop) ;

		vit = m.phi2_1(vit) ;
	} while(vit != d2) ;

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_QEMextColor<PFP>::initEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;

	EdgeInfo einfo ;
	if(m.edgeCanCollapse(d))
		computeEdgeInfo(d, einfo) ;
	else
		einfo.valid = false ;
	edgeInfo[d] = einfo ;
}

template <typename PFP>
void EdgeSelector_QEMextColor<PFP>::updateEdgeInfo(Dart d, bool recompute)
{
	MAP& m = this->m_map ;

	EdgeInfo& einfo = edgeInfo[d] ;
	if(recompute)
	{
		if(einfo.valid)
			edges.erase(einfo.it) ;		// remove the edge from the multimap
		if(m.edgeCanCollapse(d))
			computeEdgeInfo(d, einfo) ;
		else
			einfo.valid = false ;
	}
	else
	{
		if(m.edgeCanCollapse(d))
		{								 	// if the edge can be collapsed now
			if(!einfo.valid)				// but it was not before
				computeEdgeInfo(d, einfo) ;
		}
		else
		{								 // if the edge cannot be collapsed now
			if(einfo.valid)				 // and it was before
			{
				edges.erase(einfo.it) ;
				einfo.valid = false ;
			}
		}
	}
}

template <typename PFP>
void EdgeSelector_QEMextColor<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	MAP& m = this->m_map ;

	Dart dd = m.phi1(d) ;

	// New position
	Utils::QuadricNd<REAL,6> quad ;
	quad += m_quadric[d] ;	// compute the sum of the
	quad += m_quadric[dd] ;	// two vertices quadrics

	// compute all approximated attributes
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin() ;
			it != this->m_approximators.end() ;
			++it)
	{
		(*it)->approximate(d) ;
	}

	// get pos
	const VEC3& newPos = this->m_approx[m_approxindex_pos]->getApprox(d, m_attrindex_pos) ; // get newPos
	// get col
	const VEC3& newCol = this->m_approx[m_approxindex_color]->getApprox(d, m_attrindex_color) ; // get newCol

	// compute error
	VEC6 newEmb ;
	for (unsigned int i = 0 ; i < 3 ; ++i)
	{
		newEmb[i] = newPos[i] ;
		newEmb[i+3] = newCol[i] ;
	}

	const REAL& err = quad(newEmb) ;

	// Check if errated values appear
	if (err < -1e-6)
		einfo.valid = false ;
	else
	{
		einfo.it = edges.insert(std::make_pair(std::max(err,REAL(0)), d)) ;
		einfo.valid = true ;
	}
}

} // namespace Decimation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
