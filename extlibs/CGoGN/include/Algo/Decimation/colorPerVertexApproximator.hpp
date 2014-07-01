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

/************************************************************************************
 *                      NAIVE COLOR METRIC                                          *
 ************************************************************************************/

template <typename PFP>
void Approximator_ColorNaive<PFP>::approximate(Dart d)
{
	Dart dd = this->m_map.phi1(d) ;

	const VEC3& p1 = m_position.operator[](d) ;
	const VEC3& p2 = m_position.operator[](dd) ;
	const VEC3& p = m_approxposition.operator[](d) ;

	const VEC3& edge = p2 - p1 ;
	const REAL& ratio = std::max(std::min(((p - p1) * edge) / edge.norm2(),REAL(1)),REAL(0)) ;

	this->m_approx[0][d] = m_color->operator[](d)*ratio + m_color->operator[](dd)*(1-ratio) ;
}

/************************************************************************************
 *                            EXTENDED QUADRIC ERROR METRIC                         *
 ************************************************************************************/
template <typename PFP>
bool Approximator_ColorQEMext<PFP>::init()
{
	m_quadric = this->m_map.template getAttribute<Utils::QuadricNd<REAL,6>, VERTEX, MAP>("QEMext-quadric") ;
	// Does not require to be valid (if it is not, altenatives will be used).

	if(this->m_predictor)
	{
		return false ;
	}

	return m_position->isValid() && m_color->isValid() ;
}

template <typename PFP>
void Approximator_ColorQEMext<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	// get some darts
	Dart dd = m.phi2(d) ;

	Utils::QuadricNd<REAL,6> q1, q2 ;
	if(!m_quadric.isValid()) // if the selector is not QEM, compute local error quadrics
	{
		// compute the error quadric associated to v1
		Dart it = d ;
		do
		{
			VEC6 p0,p1,p2 ;
			for (unsigned int i = 0 ; i < 3 ; ++i)
			{
				p0[i] = this->m_attrV[0]->operator[](it)[i] ;
				p0[i+3] = this->m_attrV[1]->operator[](it)[i] ;
				p1[i] = this->m_attrV[0]->operator[](m.phi1(it))[i] ;
				p1[i+3] = this->m_attrV[1]->operator[](m.phi1(it))[i] ;
				p2[i] = this->m_attrV[0]->operator[](m.phi_1(it))[i] ;
				p2[i+3] = this->m_attrV[1]->operator[](m.phi_1(it))[i] ;
			}

			Utils::QuadricNd<REAL,6> q(p0,p1,p2) ;
			q1 += q ;
			it = m.phi2_1(it) ;
		} while(it != d) ;

		// compute the error quadric associated to v2
		it = dd ;
		do
		{
			VEC6 p0,p1,p2 ;
			for (unsigned int i = 0 ; i < 3 ; ++i)
			{
				p0[i] = this->m_attrV[0]->operator[](it)[i] ;
				p0[i+3] = this->m_attrV[1]->operator[](it)[i] ;
				p1[i] = this->m_attrV[0]->operator[](m.phi1(it))[i] ;
				p1[i+3] = this->m_attrV[1]->operator[](m.phi1(it))[i] ;
				p2[i] = this->m_attrV[0]->operator[](m.phi_1(it))[i] ;
				p2[i+3] = this->m_attrV[1]->operator[](m.phi_1(it))[i] ;
			}

			Utils::QuadricNd<REAL,6> q(p0,p1,p2) ;
			q2 += q ;
			it = m.phi2_1(it) ;
		} while(it != dd) ;
	}
	else // if the selector is QEM, use the error quadrics computed by the selector
	{
		q1 = m_quadric[d] ;
		q2 = m_quadric[dd] ;
	}

	Utils::QuadricNd<REAL,6> quad ;
	quad += q1 ;	// compute the sum of the
	quad += q2 ;	// two vertices quadrics

	VEC6 res ;
	bool opt = quad.findOptimizedVec(res) ;	// try to compute an optimized position for the contraction of this edge
	if(!opt)
	{
		VEC6 p1, p2 ;
		for (unsigned int i = 0 ; i < 3; ++i)
		{
			p1[i] = this->m_attrV[0]->operator[](d)[i] ;	// let the new vertex lie
			p1[i+3] = this->m_attrV[1]->operator[](d)[i] ; // on either one
			p2[i] = this->m_attrV[0]->operator[](dd)[i] ;	// of the two
			p2[i+3] = this->m_attrV[1]->operator[](dd)[i] ; // endpoints
		}
		VEC6 p12 = (p1 + p2) / 2.0f ;	// or the middle of the edge

		REAL e1 = quad(p1) ;
		REAL e2 = quad(p2) ;
		REAL e12 = quad(p12) ;
		REAL minerr = std::min(std::min(e1, e2), e12) ;	// consider only the one for
		if(minerr == e12)
			res = p12 ;		// which the error is minimal
		else if(minerr == e1)
			res = p1 ;
		else
			res = p2 ;
	}

	// copy res into m_approx
	for (unsigned int i = 0 ; i < 3 ; ++i)
	{
		this->m_approx[0][d][i] = res[i] ;
		this->m_approx[1][d][i] = res[i+3] ;
	}
}

/************************************************************************************
 *                    GEOM + COLOR OPTIMIZED ERROR METRIC                           *
 ************************************************************************************/
template <typename PFP>
bool Approximator_GeomColOpt<PFP>::init()
{
	m_quadric = this->m_map.template getAttribute<Utils::Quadric<REAL>, VERTEX, MAP>("QEMquadric") ;
	// Does not require to be valid (if it is not, altenatives will be used).

	if(this->m_predictor)
	{
		return false ;
	}

	return m_position->isValid() && m_color->isValid() ;
}

template <typename PFP>
void Approximator_GeomColOpt<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	// get some darts
	Dart dd = m.phi2(d) ;

	// POSITION
	Utils::Quadric<REAL> q1, q2 ;
	if(!m_quadric.isValid()) // if the selector is not QEM, compute local error quadrics
	{
		// compute the error quadric associated to v1
		Dart it = d ;
		do
		{
			Utils::Quadric<REAL> q(this->m_attrV[0]->operator[](it), this->m_attrV[0]->operator[](m.phi1(it)), this->m_attrV[0]->operator[](m.phi_1(it))) ;
			q1 += q ;
			it = m.phi2_1(it) ;
		} while(it != d) ;

		// compute the error quadric associated to v2
		it = dd ;
		do
		{
			Utils::Quadric<REAL> q(this->m_attrV[0]->operator[](it), this->m_attrV[0]->operator[](m.phi1(it)), this->m_attrV[0]->operator[](m.phi_1(it))) ;
			q2 += q ;
			it = m.phi2_1(it) ;
		} while(it != dd) ;
	}
	else // if the selector is QEM, use the error quadrics computed by the selector
	{
		q1 = m_quadric[d] ;
		q2 = m_quadric[dd] ;
	}

	Utils::Quadric<REAL> quad ;
	quad += q1 ;	// compute the sum of the
	quad += q2 ;	// two vertices quadrics

	VEC3 res ;
	bool opt = quad.findOptimizedPos(res) ;	// try to compute an optimized position for the contraction of this edge

	const VEC3& p0 = this->m_attrV[0]->operator[](d) ;    // let the new vertex lie
	const VEC3& p1 = this->m_attrV[0]->operator[](dd) ;   // on either one of the two endpoints

	if(false && !opt)
	{
		VEC3 p12 = (p0 + p1) / 2.0f ;   // or the middle of the edge
		REAL e1 = quad(p0) ;
		REAL e2 = quad(p1) ;
		REAL e12 = quad(p12) ;
		REAL minerr = std::min(std::min(e1, e2), e12) ; // consider only the one for
		if(minerr == e12)		this->m_approx[0][d] = p12 ;             // which the error is minimal
		else if(minerr == e1)	this->m_approx[0][d] = p0 ;
		else					this->m_approx[0][d] = p1 ;
	}
	// copy res into m_approx
	else
	{
		this->m_approx[0][d] = res ;
	}
	const VEC3& p = this->m_approx[0][d] ;

	// COLOR
	const VEC3& c1 = this->m_attrV[1]->operator[](d) ;    // let the new vertex lie
	const VEC3& c2 = this->m_attrV[1]->operator[](dd) ;   // on either one of the two endpoints

	VEC3 e = p1 - p0 ;
	VEC3 e1 = p - p0 ;
	const REAL normE1 = e1.normalize() ;
	const REAL normE = e.normalize() ;
	REAL ratio = (e * e1)*normE1/normE ;
	ratio = std::max(REAL(0),std::min(REAL(1),ratio)) ;

	this->m_approx[1][d] = ratio*c1 + (1-ratio)*c2 ;
}


} //namespace Decimation

}

} //namespace Algo

} //namespace CGoGN
