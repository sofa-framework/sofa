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

#include "Algo/DecimationVolumes/geometryApproximator.h"
#include "Algo/Geometry/volume.h"
#include <time.h>

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Decimation
{
/************************************************************************************
 *                                  MAP ORDER                                       *
 ************************************************************************************/

template <typename PFP>
bool EdgeSelector_MapOrder<PFP>::init()
{
	MAP& m = this->m_map ;
	cur = m.begin() ;
	while(!m.edgeCanCollapse(cur))
	{
		m.next(cur) ;
		if(cur == m.end())
			return false;
	}
	return true ;
}

template <typename PFP>
bool EdgeSelector_MapOrder<PFP>::nextEdge(Dart& d)
{
	MAP& m = this->m_map ;
	if(cur == m.end())
		return false ;
	d = cur ;
	return true ;
}

template <typename PFP>
void EdgeSelector_MapOrder<PFP>::updateAfterCollapse(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;
	cur = m.begin() ;
	while(!this->m_select(cur) || !m.edgeCanCollapse(cur))
	{
		m.next(cur) ;
		if(cur == m.end())
			break ;
	}
}

/************************************************************************************
 *	                                  RANDOM                                    	*
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
bool EdgeSelector_Random<PFP>::nextEdge(Dart& d)
{
	if(cur == darts.size() && allSkipped)
		return false ;
	d = darts[cur] ;
	return true ;
}


template <typename PFP>
void EdgeSelector_Random<PFP>::updateAfterCollapse(Dart d2, Dart dd2)
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
	} while(!m.edgeCanCollapse(darts[cur])) ;
}

/************************************************************************************
 *                                 		SG98                                     	*
 ************************************************************************************/
template <typename PFP>
bool EdgeSelector_SG98<PFP>::init()
{
	MAP& m = this->m_map ;

	bool ok = false ;
	for(typename std::vector<ApproximatorGen<PFP>*>::iterator it = this->m_approximators.begin();
		it != this->m_approximators.end() && !ok;
		++it)
	{
		if((*it)->getApproximatedAttributeName() == "position")
		{
			m_positionApproximator = reinterpret_cast<Approximator<PFP, VEC3>* >(*it) ;
			ok = true ;
		}
	}

	if(!ok)
		return false ;

	edges.clear() ;

	TraversorE<typename PFP::MAP> tE(m);
	for(Dart dit = tE.begin() ; dit != tE.end() ; dit = tE.next())
	{
		initEdgeInfo(dit);
	}

	cur = edges.begin();

	return true;
}

template <typename PFP>
bool EdgeSelector_SG98<PFP>::nextEdge(Dart& d)
{
	if(cur == edges.end() || edges.empty())
		return false ;
	d = (*cur).second ;
	return true ;
}

template <typename PFP>
void EdgeSelector_SG98<PFP>::updateBeforeCollapse(Dart d)
{
	MAP& m = this->m_map ;

	// remove all
	// the concerned edges
	// from the multimap

}

template <typename PFP>
void EdgeSelector_SG98<PFP>::updateAfterCollapse(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;


	// must recompute some edge infos in the

	// various optimizations are applied here by
	// treating differently :
	// - edges for which the criteria must be recomputed
	// - edges that must be re-embedded
	// - edges for which only the collapsibility must be re-tested

	cur = edges.begin() ; // set the current edge to the first one
}

template <typename PFP>
void EdgeSelector_SG98<PFP>::initEdgeInfo(Dart d)
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
void EdgeSelector_SG98<PFP>::updateEdgeInfo(Dart d, bool recompute)
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
void EdgeSelector_SG98<PFP>::computeEdgeInfo(Dart d, EdgeInfo& einfo)
{
	MAP& m = this->m_map ;
	Dart dd = m.phi2(d) ;

	typename PFP::REAL vol_icells = 0.0;
	typename PFP::REAL vol_ncells = 0.0;

	m_positionApproximator->approximate(d) ;
	typename PFP::VEC3 approx = m_positionApproximator->getApprox(d)) ;

	DartMarkerStore mv(m);

	Traversor3EW<TOPO_MAP> t3EW(m,d);
	for(Dart dit = t3EW.begin() ; dit != t3EW.end() ; dit = t3EW.next())
	{
		vol_icells += Algo::Geometry::tetrahedronSignedVolume<PFP>(m,dit,this->pos);

		mv.markOrbit<VOLUME>(dit);
	}

	Traversor3WWaV<TOPO_MAP> t3VVaE_v1(m,d);
	for(Dart dit = t3VVaE_v1.begin() ; dit != t3VVaE_v1.end() ; dit = t3VVaE_v1.next())
	{
		if(!mv.isMarked(dit))
		{
			typename PFP::VEC3 p2 = position[map.phi1(dit)] ;
			typename PFP::VEC3 p3 = position[map.phi_1(dit)] ;
			typename PFP::VEC3 p4 = position[map.phi_1(map.phi2(dit))] ;

			typename PFP::REAL voli =  Geom::tetraSignedVolume(approx, p2, p3, p4) ;

			vol_ncells += Algo::Geometry::tetrahedronSignedVolume<PFP>(m,dit,this->pos) - voli ;

			mv.mark(dit);
		}
	}

	Traversor3WWaV<TOPO_MAP> t3VVaE_v2(m,phi2(d));
	for(Dart dit = t3VVaE_v2.begin() ; dit != t3VVaE_v2.end() ; dit = t3VVaE_v2.next())
	{
		if(!mv.isMarked(dit))
		{
			typename PFP::VEC3 p2 = position[map.phi1(dit)] ;
			typename PFP::VEC3 p3 = position[map.phi_1(dit)] ;
			typename PFP::VEC3 p4 = position[map.phi_1(map.phi2(dit))] ;

			typename PFP::REAL voli =  Geom::tetraSignedVolume(approx, p2, p3, p4) ;

			vol_ncells += Algo::Geometry::tetrahedronSignedVolume<PFP>(m,dit,this->pos) - voli ;

			mv.mark(dit);
		}
	}



//	Quadric<REAL> quad ;
//	quad += quadric[d] ;	// compute the sum of the
//	quad += quadric[dd] ;	// two vertices quadrics
//
//	m_positionApproximator->approximate(d) ;
//
//	REAL err = quad(m_positionApproximator->getApprox(d)) ;
//
//	einfo.it = edges.insert(std::make_pair(err, d)) ;
//	einfo.valid = true ;
}

} //end namespace Decimation
} //namespace Volume
} //end namespace Algo
} //end namespace CGoGN
