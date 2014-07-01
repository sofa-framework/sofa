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

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Decimation
{

/*****************************************************************************************************************
 *                                 HALF-EDGE MEMORYLESS QEM METRIC                                               *
 *****************************************************************************************************************/

template <typename PFP>
bool HalfEdgeSelector_QEMml<PFP>::init()
{
	MAP& m = this->m_map ;

	bool ok = false ;
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin();
		it != this->m_approximators.end() && !ok;
		++it)
	{
		if((*it)->getApproximatedAttributeName() == "position")
		{
			m_positionApproximator = reinterpret_cast<Approximator<PFP, VEC3,DART>* >(*it) ;
			ok = true ;
		}
	}

	if(!ok)
		return false ;

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

	// Init multimap for each Half-edge
	halfEdges.clear() ;

	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		initHalfEdgeInfo(d) ;	// init the edges with their optimal info
	}							// and insert them in the multimap according to their error

	cur = halfEdges.begin() ; 	// init the current edge to the first one

	return true ;
}

template <typename PFP>
bool HalfEdgeSelector_QEMml<PFP>::nextEdge(Dart& d) const
{
	if(cur == halfEdges.end() || halfEdges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void HalfEdgeSelector_QEMml<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	HalfEdgeInfo* edgeE = &(halfEdgeInfo[d]) ;
	if(edgeE->valid)
		halfEdges.erase(edgeE->it) ;

	edgeE = &(halfEdgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)						// remove all
		halfEdges.erase(edgeE->it) ;

	edgeE = &(halfEdgeInfo[m.phi_1(d)]) ;	// the halfedges that will disappear
	if(edgeE->valid)
		halfEdges.erase(edgeE->it) ;
										// from the multimap
	Dart dd = m.phi2(d) ;
	assert(dd != d) ;
	if(dd != d)
	{
		edgeE = &(halfEdgeInfo[dd]) ;
		if(edgeE->valid)
			halfEdges.erase(edgeE->it) ;

		edgeE = &(halfEdgeInfo[m.phi1(dd)]) ;
		if(edgeE->valid)
			halfEdges.erase(edgeE->it) ;

		edgeE = &(halfEdgeInfo[m.phi_1(dd)]) ;
		if(edgeE->valid)
			halfEdges.erase(edgeE->it) ;
	}
}

template <typename PFP>
void HalfEdgeSelector_QEMml<PFP>::recomputeQuadric(const Dart d, const bool recomputeNeighbors)
{
	Dart dFront,dBack ;
	Dart dInit = d ;

	// Init Front
	dFront = dInit ;

	quadric[d].zero() ;

   	do {
   		// Make step
   		dBack = this->m_map.phi1(dFront) ;
       	dFront = this->m_map.phi2_1(dFront) ;

       	if (this->m_map.phi2(dFront) != dFront) { // if dFront is no border
           	quadric[d] += Utils::Quadric<REAL>(this->m_position[d],this->m_position[dBack],this->m_position[this->m_map.phi1(dFront)]) ;
       	}

       	if (recomputeNeighbors)
       		recomputeQuadric(dBack, false) ;

    } while(dFront != dInit) ;
}

template <typename PFP>
void HalfEdgeSelector_QEMml<PFP>::updateAfterCollapse(Dart d2, Dart /*dd2*/)
{
	MAP& m = this->m_map ;

	recomputeQuadric(d2, true) ;

	Dart vit = d2 ;
	do
	{
		updateHalfEdgeInfo(vit, true) ;
		Dart d = m.phi2(vit) ;
		if (d != vit)
			updateHalfEdgeInfo(d, true) ;

		updateHalfEdgeInfo(m.phi1(vit), true) ;
		d = m.phi2(m.phi1(vit)) ;
		if (d != m.phi1(vit))
			updateHalfEdgeInfo(d, true) ;

		Dart stop = m.phi2(vit) ;
		assert (stop != vit) ;
		Dart vit2 = m.phi12(m.phi1(vit)) ;
		do {
			updateHalfEdgeInfo(vit2, true) ;
			d = m.phi2(vit2) ;
			if (d != vit2)
				updateHalfEdgeInfo(d, true) ;

			updateHalfEdgeInfo(m.phi1(vit2), false) ;
			d = m.phi2(m.phi1(vit2)) ;
			if (d != m.phi1(vit2))
				updateHalfEdgeInfo(d, false) ;

			vit2 = m.phi12(vit2) ;
		} while (stop != vit2) ;
		vit = m.phi2_1(vit) ;
	} while(vit != d2) ;

	cur = halfEdges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void HalfEdgeSelector_QEMml<PFP>::initHalfEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;
	HalfEdgeInfo heinfo ;
	if(m.edgeCanCollapse(d))
		computeHalfEdgeInfo(d, heinfo) ;
	else
		heinfo.valid = false ;

	halfEdgeInfo[d] = heinfo ;
}

template <typename PFP>
void HalfEdgeSelector_QEMml<PFP>::updateHalfEdgeInfo(Dart d, bool recompute)
{
	MAP& m = this->m_map ;
	HalfEdgeInfo& heinfo = halfEdgeInfo[d] ;
	if(recompute)
	{
		if(heinfo.valid)
			halfEdges.erase(heinfo.it) ;			// remove the edge from the multimap
		if(m.edgeCanCollapse(d))
			computeHalfEdgeInfo(d, heinfo) ;
		else
			heinfo.valid = false ;
	}
	else
	{
		if(m.edgeCanCollapse(d))
		{								 	// if the edge can be collapsed now
			if(!heinfo.valid)				 // but it was not before
				computeHalfEdgeInfo(d, heinfo) ;
		}
		else
		{								 // if the edge cannot be collapsed now
			if(heinfo.valid)				 // and it was before
			{
				halfEdges.erase(heinfo.it) ;
				heinfo.valid = false ;
			}
		}
	}
}

template <typename PFP>
void HalfEdgeSelector_QEMml<PFP>::computeHalfEdgeInfo(Dart d, HalfEdgeInfo& heinfo)
{
	MAP& m = this->m_map ;
	Dart dd = m.phi1(d) ;

	Utils::Quadric<REAL> quad ;
	quad += quadric[d] ;	// compute the sum of the
	quad += quadric[dd] ;	// two vertices quadrics

	m_positionApproximator->approximate(d) ;

	REAL err = quad(m_positionApproximator->getApprox(d)) ;
	heinfo.it = halfEdges.insert(std::make_pair(err, d)) ;
	heinfo.valid = true ;
}

/*****************************************************************************************************************
 *                                 HALF-EDGE QEMextColor METRIC                                                  *
 *****************************************************************************************************************/

template <typename PFP>
bool HalfEdgeSelector_QEMextColor<PFP>::init()
{
	MAP& m = this->m_map ;

	// Verify availability of required approximators
	unsigned int ok = 0 ;
	for (unsigned int approxindex = 0 ; approxindex < this->m_approximators.size() ; ++approxindex)
	{
		assert(this->m_approximators[approxindex]->getType() == A_hQEM
				|| this->m_approximators[approxindex]->getType() == A_hHalfCollapse
				|| !"Approximator for selector (HalfEdgeSelector_QEMextColor) must be of a half-edge approximator") ;

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
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, DART>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
			else if(ok == 1 && this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "color")
			{
				++ok ;
				m_approxindex_color = approxindex ;
				m_attrindex_color = attrindex ;
				m_color = m.template getAttribute<typename PFP::VEC3, VERTEX, MAP>("color") ;
				assert(m_color.isValid() || !"EdgeSelector_QEMextColor: color attribute is not valid") ;
				if (!saved)
				{
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, DART>* >(this->m_approximators[approxindex])) ;
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
		Dart d_1 = m.phi_1(dit) ;					// initialize the quadric of the triangle

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


	// Init multimap for each Half-edge
	halfEdges.clear() ;

	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		initHalfEdgeInfo(d) ;	// init the edges with their optimal info
	}							// and insert them in the multimap according to their error

	cur = halfEdges.begin() ; 	// init the current edge to the first one

	return true ;
}

template <typename PFP>
bool HalfEdgeSelector_QEMextColor<PFP>::nextEdge(Dart& d) const
{
	if(cur == halfEdges.end() || halfEdges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void HalfEdgeSelector_QEMextColor<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	HalfEdgeInfo* edgeE = &(halfEdgeInfo[d]) ;
	if(edgeE->valid)
		halfEdges.erase(edgeE->it) ;

	edgeE = &(halfEdgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)						// remove all
		halfEdges.erase(edgeE->it) ;

	edgeE = &(halfEdgeInfo[m.phi_1(d)]) ;	// the halfedges that will disappear
	if(edgeE->valid)
		halfEdges.erase(edgeE->it) ;
										// from the multimap
	Dart dd = m.phi2(d) ;
	assert(dd != d) ;
	if(dd != d)
	{
		edgeE = &(halfEdgeInfo[dd]) ;
		if(edgeE->valid)
			halfEdges.erase(edgeE->it) ;

		edgeE = &(halfEdgeInfo[m.phi1(dd)]) ;
		if(edgeE->valid)
			halfEdges.erase(edgeE->it) ;

		edgeE = &(halfEdgeInfo[m.phi_1(dd)]) ;
		if(edgeE->valid)
			halfEdges.erase(edgeE->it) ;
	}
}

template <typename PFP>
void HalfEdgeSelector_QEMextColor<PFP>::recomputeQuadric(const Dart d, const bool recomputeNeighbors) {
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
void HalfEdgeSelector_QEMextColor<PFP>::updateAfterCollapse(Dart d2, Dart /*dd2*/)
{
	MAP& m = this->m_map ;

	recomputeQuadric(d2, true) ;

	Dart vit = d2 ;
	do
	{
		updateHalfEdgeInfo(vit, true) ;
		Dart d = m.phi2(vit) ;
		if (d != vit)
			updateHalfEdgeInfo(d, true) ;

		updateHalfEdgeInfo(m.phi1(vit), true) ;
		d = m.phi2(m.phi1(vit)) ;
		if (d != m.phi1(vit))
			updateHalfEdgeInfo(d, true) ;

		Dart stop = m.phi2(vit) ;
		assert (stop != vit) ;
		Dart vit2 = m.phi12(m.phi1(vit)) ;
		do {
			updateHalfEdgeInfo(vit2, true) ;
			d = m.phi2(vit2) ;
			if (d != vit2)
				updateHalfEdgeInfo(d, true) ;

			updateHalfEdgeInfo(m.phi1(vit2), false) ;
			d = m.phi2(m.phi1(vit2)) ;
			if (d != m.phi1(vit2))
				updateHalfEdgeInfo(d, false) ;

			vit2 = m.phi12(vit2) ;
		} while (stop != vit2) ;
		vit = m.phi2_1(vit) ;
	} while(vit != d2) ;

	cur = halfEdges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void HalfEdgeSelector_QEMextColor<PFP>::initHalfEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;
	HalfEdgeInfo heinfo ;
	if(m.edgeCanCollapse(d))
		computeHalfEdgeInfo(d, heinfo) ;
	else
		heinfo.valid = false ;

	halfEdgeInfo[d] = heinfo ;
}

template <typename PFP>
void HalfEdgeSelector_QEMextColor<PFP>::updateHalfEdgeInfo(Dart d, bool recompute)
{
	MAP& m = this->m_map ;
	HalfEdgeInfo& heinfo = halfEdgeInfo[d] ;
	if(recompute)
	{
		if(heinfo.valid)
			halfEdges.erase(heinfo.it) ;			// remove the edge from the multimap
		if(m.edgeCanCollapse(d))
			computeHalfEdgeInfo(d, heinfo) ;
		else
			heinfo.valid = false ;
	}
	else
	{
		if(m.edgeCanCollapse(d))
		{								 	// if the edge can be collapsed now
			if(!heinfo.valid)				 // but it was not before
				computeHalfEdgeInfo(d, heinfo) ;
		}
		else
		{								 // if the edge cannot be collapsed now
			if(heinfo.valid)				 // and it was before
			{
				halfEdges.erase(heinfo.it) ;
				heinfo.valid = false ;
			}
		}
	}
}

template <typename PFP>
void HalfEdgeSelector_QEMextColor<PFP>::computeHalfEdgeInfo(Dart d, HalfEdgeInfo& heinfo)
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
	const VEC3& newPos = this->m_approx[m_approxindex_pos]->getApprox(d,m_attrindex_pos) ; // get newPos
	// get col
	const VEC3& newCol = this->m_approx[m_approxindex_color]->getApprox(d,m_attrindex_color) ; // get newCol

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
		heinfo.valid = false ;
	else
	{
		heinfo.it = this->halfEdges.insert(std::make_pair(std::max(err,REAL(0)), d)) ;
		heinfo.valid = true ;
	}
}

/*****************************************************************************************************************
 *                                 HALF-EDGE QEMextColorNormal METRIC                                                  *
 *****************************************************************************************************************/

template <typename PFP>
bool HalfEdgeSelector_QEMextColorNormal<PFP>::init()
{
	MAP& m = this->m_map ;

	// Verify availability of required approximators
	unsigned int ok = 0 ;
	for (unsigned int approxindex = 0 ; approxindex < this->m_approximators.size() ; ++approxindex)
	{
		assert(this->m_approximators[approxindex]->getType() == A_hQEM
				|| this->m_approximators[approxindex]->getType() == A_hHalfCollapse
				|| !"Approximator for selector (HalfEdgeSelector_QEMextColorNormal) must be of a half-edge approximator") ;

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
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, DART>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
			else if(ok == 1 && this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "color")
			{
				++ok ;
				m_approxindex_color = approxindex ;
				m_attrindex_color = attrindex ;
				m_color = m.template getAttribute<typename PFP::VEC3, VERTEX>("color") ;
				assert(m_color.isValid() || !"EdgeSelector_QEMextColor: color attribute is not valid") ;
				if (!saved)
				{
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, DART>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
			else if(ok == 2 && (this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "normal" || this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "frameN"))
			{
				++ok ;
				m_approxindex_normal = approxindex ;
				m_attrindex_normal = attrindex ;
				m_normal = m.template getAttribute<typename PFP::VEC3, VERTEX>("normal") ;
				assert(m_normal.isValid() || !"EdgeSelector_QEMextColorNormal: normal attribute is not valid") ;
				if (!saved)
				{
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, DART>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
		}
	}

	if(ok != 3)
		return false ;

	TraversorV<MAP> travV(m);
	for(Dart dit = travV.begin() ; dit != travV.end() ; dit = travV.next())
	{
		Utils::QuadricNd<REAL,9> q ;		// create one quadric
		m_quadric[dit] = q ;		// per vertex
	}

	// Compute quadric per vertex
	TraversorF<MAP> travF(m) ;
	for(Dart dit = travF.begin() ; dit != travF.end() ; dit = travF.next()) // init QEM quadrics
	{
		Dart d1 = m.phi1(dit) ;					// for each triangle,
		Dart d_1 = m.phi_1(dit) ;					// initialize the quadric of the triangle

		VEC9 p0, p1, p2 ;
		for (unsigned int i = 0 ; i < 3 ; ++i)
		{
			p0[i] = this->m_position[dit][i] ;
			p0[i+3] = this->m_color[dit][i] ;
			p0[i+6] = this->m_normal[dit][i] ;
			p1[i] = this->m_position[d1][i] ;
			p1[i+3] = this->m_color[d1][i] ;
			p1[i+6] = this->m_normal[d1][i] ;
			p2[i] = this->m_position[d_1][i] ;
			p2[i+3] = this->m_color[d_1][i] ;
			p2[i+6] = this->m_normal[d_1][i] ;

		}
		Utils::QuadricNd<REAL,9> q(p0,p1,p2) ;
		m_quadric[dit] += q ;						// and add the contribution of
		m_quadric[d1] += q ;						// this quadric to the ones
		m_quadric[d_1] += q ;						// of the 3 incident vertices
	}


	// Init multimap for each Half-edge
	halfEdges.clear() ;

	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		initHalfEdgeInfo(d) ;	// init the edges with their optimal info
	}							// and insert them in the multimap according to their error

	cur = halfEdges.begin() ; 	// init the current edge to the first one

	return true ;
}

template <typename PFP>
bool HalfEdgeSelector_QEMextColorNormal<PFP>::nextEdge(Dart& d) const
{
	if(cur == halfEdges.end() || halfEdges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void HalfEdgeSelector_QEMextColorNormal<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	HalfEdgeInfo* edgeE = &(halfEdgeInfo[d]) ;
	if(edgeE->valid)
		halfEdges.erase(edgeE->it) ;

	edgeE = &(halfEdgeInfo[m.phi1(d)]) ;
	if(edgeE->valid)						// remove all
		halfEdges.erase(edgeE->it) ;

	edgeE = &(halfEdgeInfo[m.phi_1(d)]) ;	// the halfedges that will disappear
	if(edgeE->valid)
		halfEdges.erase(edgeE->it) ;
										// from the multimap
	Dart dd = m.phi2(d) ;
	assert(dd != d) ;
	if(dd != d)
	{
		edgeE = &(halfEdgeInfo[dd]) ;
		if(edgeE->valid)
			halfEdges.erase(edgeE->it) ;

		edgeE = &(halfEdgeInfo[m.phi1(dd)]) ;
		if(edgeE->valid)
			halfEdges.erase(edgeE->it) ;

		edgeE = &(halfEdgeInfo[m.phi_1(dd)]) ;
		if(edgeE->valid)
			halfEdges.erase(edgeE->it) ;
	}
}

template <typename PFP>
void HalfEdgeSelector_QEMextColorNormal<PFP>::recomputeQuadric(const Dart d, const bool recomputeNeighbors) {
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

			VEC9 p0, p1, p2 ;
			for (unsigned int i = 0 ; i < 3 ; ++i)
			{

				p0[i] = this->m_position[d][i] ;
				p0[i+3] = this->m_color[d][i] ;
				p0[i+6] = this->m_normal[d][i] ;
				p1[i] = this->m_position[d2][i] ;
				p1[i+3] = this->m_color[d2][i] ;
				p1[i+6] = this->m_normal[d2][i] ;
				p2[i] = this->m_position[dBack][i] ;
				p2[i+3] = this->m_color[dBack][i] ;
				p2[i+6] = this->m_normal[dBack][i] ;
			}
			m_quadric[d] += Utils::QuadricNd<REAL,9>(p0,p1,p2) ;
		}
		if (recomputeNeighbors)
			recomputeQuadric(dBack, false) ;

	} while(dFront != dInit) ;
}

template <typename PFP>
void HalfEdgeSelector_QEMextColorNormal<PFP>::updateAfterCollapse(Dart d2, Dart /*dd2*/)
{
	MAP& m = this->m_map ;

	recomputeQuadric(d2, true) ;

	Dart vit = d2 ;
	do
	{
		updateHalfEdgeInfo(vit, true) ;
		Dart d = m.phi2(vit) ;
		if (d != vit)
			updateHalfEdgeInfo(d, true) ;

		updateHalfEdgeInfo(m.phi1(vit), true) ;
		d = m.phi2(m.phi1(vit)) ;
		if (d != m.phi1(vit))
			updateHalfEdgeInfo(d, true) ;

		Dart stop = m.phi2(vit) ;
		assert (stop != vit) ;
		Dart vit2 = m.phi12(m.phi1(vit)) ;
		do {
			updateHalfEdgeInfo(vit2, true) ;
			d = m.phi2(vit2) ;
			if (d != vit2)
				updateHalfEdgeInfo(d, true) ;

			updateHalfEdgeInfo(m.phi1(vit2), false) ;
			d = m.phi2(m.phi1(vit2)) ;
			if (d != m.phi1(vit2))
				updateHalfEdgeInfo(d, false) ;

			vit2 = m.phi12(vit2) ;
		} while (stop != vit2) ;
		vit = m.phi2_1(vit) ;
	} while(vit != d2) ;

	cur = halfEdges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void HalfEdgeSelector_QEMextColorNormal<PFP>::initHalfEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;
	HalfEdgeInfo heinfo ;
	if(m.edgeCanCollapse(d))
		computeHalfEdgeInfo(d, heinfo) ;
	else
		heinfo.valid = false ;

	halfEdgeInfo[d] = heinfo ;
}

template <typename PFP>
void HalfEdgeSelector_QEMextColorNormal<PFP>::updateHalfEdgeInfo(Dart d, bool recompute)
{
	MAP& m = this->m_map ;
	HalfEdgeInfo& heinfo = halfEdgeInfo[d] ;
	if(recompute)
	{
		if(heinfo.valid)
			halfEdges.erase(heinfo.it) ;			// remove the edge from the multimap
		if(m.edgeCanCollapse(d))
			computeHalfEdgeInfo(d, heinfo) ;
		else
			heinfo.valid = false ;
	}
	else
	{
		if(m.edgeCanCollapse(d))
		{								 	// if the edge can be collapsed now
			if(!heinfo.valid)				 // but it was not before
				computeHalfEdgeInfo(d, heinfo) ;
		}
		else
		{								 // if the edge cannot be collapsed now
			if(heinfo.valid)				 // and it was before
			{
				halfEdges.erase(heinfo.it) ;
				heinfo.valid = false ;
			}
		}
	}
}

template <typename PFP>
void HalfEdgeSelector_QEMextColorNormal<PFP>::computeHalfEdgeInfo(Dart d, HalfEdgeInfo& heinfo)
{
	MAP& m = this->m_map ;
	Dart dd = m.phi1(d) ;

	// New position
	Utils::QuadricNd<REAL,9> quad ;
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
	const VEC3& newCol = this->m_approx[m_approxindex_color]->getApprox(d,m_attrindex_color) ; // get newCol
	// get normal
	const VEC3& newN = this->m_approx[m_approxindex_normal]->getApprox(d,m_attrindex_normal) ; // get newN

	// compute error
	VEC9 newEmb ;
	for (unsigned int i = 0 ; i < 3 ; ++i)
	{
		newEmb[i] = newPos[i] ;
		newEmb[i+3] = newCol[i] ;
		newEmb[i+6] = newN[i] ;
	}

	const REAL& err = quad(newEmb) ;

	// Check if errated values appear
	if (err < -1e-6)
		heinfo.valid = false ;
	else
	{
		heinfo.it = this->halfEdges.insert(std::make_pair(std::max(err,REAL(0)), d)) ;
		heinfo.valid = true ;
	}
}


/*****************************************************************************************************************
 *                                 HALF-EDGE COLOR GRADIENT ERR                                                  *
 *****************************************************************************************************************/

template <typename PFP>
bool HalfEdgeSelector_ColorGradient<PFP>::init()
{
	MAP& m = this->m_map ;


	// Verify availability of required approximators
	unsigned int ok = 0 ;
	for (unsigned int approxindex = 0 ; approxindex < this->m_approximators.size() ; ++approxindex)
	{
		assert(this->m_approximators[approxindex]->getType() == A_hQEM
				|| this->m_approximators[approxindex]->getType() == A_hHalfCollapse
				|| !"Approximator for selector (HalfEdgeSelector_ColorExperimental) must be of a half-edge approximator") ;

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
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, DART>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
			else if(ok == 1 && this->m_approximators[approxindex]->getApproximatedAttributeName(attrindex) == "color")
			{
				++ok ;
				m_approxindex_color = approxindex ;
				m_attrindex_color = attrindex ;
				m_color = m.template getAttribute<typename PFP::VEC3, VERTEX, MAP>("color") ;
				assert(m_color.isValid() || !"EdgeSelector_QEMextColor: color attribute is not valid") ;
				if (!saved)
				{
					m_approx.push_back(reinterpret_cast<Approximator<PFP, VEC3, DART>* >(this->m_approximators[approxindex])) ;
					saved = true ;
				}
			}
		}
	}

	if(ok != 2)
		return false ;

	CellMarker<MAP, VERTEX> vMark(m) ;
	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		if(!vMark.isMarked(d))
		{
			Utils::Quadric<REAL> q ;	// create one quadric
			m_quadric[d] = q ;	// per vertex
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
			m_quadric[d] += q ;					// and add the contribution of
			m_quadric[d1] += q ;					// this quadric to the ones
			m_quadric[d_1] += q ;					// of the 3 incident vertices
			mark.template markOrbit<FACE>(d) ;
		}
	}

	// Init multimap for each Half-edge
	halfEdges.clear() ;

	for(Dart d = m.begin(); d != m.end(); m.next(d))
	{
		initHalfEdgeInfo(d) ;	// init the edges with their optimal info
	}							// and insert them in the multimap according to their error

	cur = halfEdges.begin() ; 	// init the current edge to the first one

	return true ;
}

template <typename PFP>
bool HalfEdgeSelector_ColorGradient<PFP>::nextEdge(Dart& d) const
{
	if(cur == halfEdges.end() || halfEdges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void HalfEdgeSelector_ColorGradient<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	const Dart& v0 = m.phi1(d) ;

	Traversor2VVaE<MAP> tv(m,v0) ;
	for (Dart v = tv.begin() ; v != tv.end() ; v = tv.next())
	{
		Traversor2VE<MAP> te(m,v) ;
		for (Dart he = te.begin() ; he != te.end() ; he = te.next())
		{
			HalfEdgeInfo* edgeE = &(halfEdgeInfo[he]) ;
			if(edgeE->valid)
			{
				edgeE->valid = false ;
				halfEdges.erase(edgeE->it) ;
			}
			Dart de = m.phi2(he) ;
			edgeE = &(halfEdgeInfo[de]) ;
			if(edgeE->valid)
			{
				edgeE->valid = false ;
				halfEdges.erase(edgeE->it) ;
			}
		}
	}

//	HalfEdgeInfo* edgeE = &(halfEdgeInfo[d]) ;
//	if(edgeE->valid)
//		halfEdges.erase(edgeE->it) ;
//
//	edgeE = &(halfEdgeInfo[m.phi1(d)]) ;
//	if(edgeE->valid)						// remove all
//		halfEdges.erase(edgeE->it) ;
//
//	edgeE = &(halfEdgeInfo[m.phi_1(d)]) ;	// the halfedges that will disappear
//	if(edgeE->valid)
//		halfEdges.erase(edgeE->it) ;
//										// from the multimap
//	Dart dd = m.phi2(d) ;
//	assert(dd != d) ;
//	if(dd != d)
//	{
//		edgeE = &(halfEdgeInfo[dd]) ;
//		if(edgeE->valid)
//			halfEdges.erase(edgeE->it) ;
//
//		edgeE = &(halfEdgeInfo[m.phi1(dd)]) ;
//		if(edgeE->valid)
//			halfEdges.erase(edgeE->it) ;
//
//		edgeE = &(halfEdgeInfo[m.phi_1(dd)]) ;
//		if(edgeE->valid)
//			halfEdges.erase(edgeE->it) ;
//	}
}

template <typename PFP>
void HalfEdgeSelector_ColorGradient<PFP>::recomputeQuadric(const Dart d)
{
	Dart dFront,dBack ;
	Dart dInit = d ;

	// Init Front
	dFront = dInit ;

	m_quadric[d].zero() ;

	do {
		// Make step
		dBack = this->m_map.phi1(dFront) ;
		dFront = this->m_map.phi2_1(dFront) ;

		if (this->m_map.phi2(dFront) != dFront) { // if dFront is no border
			m_quadric[d] += Utils::Quadric<REAL>(this->m_position[d],this->m_position[dBack],this->m_position[this->m_map.phi1(dFront)]) ;
		}
	} while(dFront != dInit) ;
}

template <typename PFP>
void HalfEdgeSelector_ColorGradient<PFP>::updateAfterCollapse(Dart d2, Dart /*dd2*/)
{
	MAP& m = this->m_map ;

	const Dart& v1 = d2 ;

	recomputeQuadric(v1) ;
	Traversor2VVaE<MAP> tv(m,v1) ;
	for (Dart v = tv.begin() ; v != tv.end() ; v = tv.next())
	{
		recomputeQuadric(v) ;
	}

	for (Dart v = tv.begin() ; v != tv.end() ; v = tv.next())
	{
		Traversor2VE<MAP> te(m,v) ;
		for (Dart e = te.begin() ; e != te.end() ; e = te.next())
		{
			updateHalfEdgeInfo(e) ;
			updateHalfEdgeInfo(m.phi2(e)) ;
		}
	}

	cur = halfEdges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void HalfEdgeSelector_ColorGradient<PFP>::initHalfEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;
	HalfEdgeInfo heinfo ;

	if(m.edgeCanCollapse(d))
		computeHalfEdgeInfo(d, heinfo) ;
	else
		heinfo.valid = false ;

	halfEdgeInfo[d] = heinfo ;
}

template <typename PFP>
void HalfEdgeSelector_ColorGradient<PFP>::updateHalfEdgeInfo(Dart d)
{
	MAP& m = this->m_map ;
	HalfEdgeInfo& heinfo = halfEdgeInfo[d] ;

	if(!heinfo.valid && m.edgeCanCollapse(d))
		computeHalfEdgeInfo(d, heinfo) ;
}

template <typename PFP>
void HalfEdgeSelector_ColorGradient<PFP>::computeHalfEdgeInfo(Dart d, HalfEdgeInfo& heinfo)
{
	MAP& m = this->m_map ;
	Dart dd = m.phi1(d) ;

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

	// Get all approximated attributes
	// New position
	const VEC3& newPos = this->m_approx[m_approxindex_pos]->getApprox(d,m_attrindex_pos) ; // get newPos
	// New normal
//	const VEC3& newColor = this->m_approx[m_approxindex_color]->getApprox(d,m_attrindex_color) ; // get new color

	const Dart& v0 = dd ;
	const Dart& v1 = d ;

	assert(newPos == m_pos[v1]) ;
	assert( this->m_approx[m_approxindex_color]->getApprox(d,m_attrindex_color) == m_color[v1]) ;

	// Compute errors
	// Position
	Utils::Quadric<REAL> quadGeom ;
	quadGeom += m_quadric[d] ;	// compute the sum of the
	quadGeom += m_quadric[dd] ;	// two vertices quadrics

	//std::cout << quadGeom(newPos) / (alpha/M_PI + quadHF(newHF)) << std::endl ;
	// sum of QEM metric and color gradient metric
	const REAL t = 0.01 ;
	const REAL& err =
			t*quadGeom(newPos) + // geom
			(1-t)*computeGradientColorError(v0,v1).norm()/sqrt(3) // color
			;

	/*std::cout << quadGeom(newPos) << std::endl ;
	std::cerr << computeGradientColorError(v0,v1).norm()/sqrt(3) << std::endl ;*/

	// Check if errated values appear
	if (err < -1e-6)
		heinfo.valid = false ;
	else
	{
		heinfo.it = this->halfEdges.insert(std::make_pair(std::max(err,REAL(0)), d)) ;
		heinfo.valid = true ;
	}
}

template <typename PFP>
typename PFP::VEC3
HalfEdgeSelector_ColorGradient<PFP>::computeGradientColorError(const Dart& v0, const Dart& v1) const
{
	MAP& m = this->m_map ;

	Traversor2VF<MAP> tf(m,v0) ; // all faces around vertex v0

	const VEC3& P0 = m_pos[v0] ;
	const VEC3& P1 = m_pos[v1] ;
	const VEC3& c0 = m_color[v0] ;
	const VEC3& c1 = m_color[v1] ;
	const VEC3 d = P1 - P0 ; // displacement vector

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
			VEC3 grad = (ei.norm2()*(ci[c]-c1[c]) + (ei*ej)*(cj[c]-c1[c]))*ej ;
			grad -= (ej.norm2()*(cj[c]-c1[c]) + (ei*ej)*(ci[c]-c1[c]))*ei ;
			const REAL denom = areaIJ0sq ;
			if (denom < 1e-9) // case flat triangles
				grad = VEC3(0,0,0) ;
			else
				grad /= denom ;

			// color change error for channel i
			count[c] += areaIJ0 * pow(d*grad,2) ;
		}
	}

	const VEC3 colDiff = c1 - c0 ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		count[c] += pow(colDiff[c],2) * areaSum ;
	}

	return count ;
}

} // namespace Decimation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
