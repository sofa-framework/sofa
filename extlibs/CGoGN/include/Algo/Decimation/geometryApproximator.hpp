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
#include "Algo/Topo/embedding.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Decimation
{

/************************************************************************************
 *                            QUADRIC ERROR METRIC                                  *
 ************************************************************************************/
template <typename PFP>
bool Approximator_QEM<PFP>::init()
{
	m_quadric = this->m_map.template getAttribute<Utils::Quadric<REAL>, VERTEX, MAP>("QEMquadric") ;
	// Does not require to be valid (if it is not, altenatives will be used).

	if(this->m_predictor)
	{
		return false ;
	}
	return true ;
}

template <typename PFP>
void Approximator_QEM<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	// get some darts
	Dart dd = m.phi2(d) ;

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
	quad += q1 ;    // compute the sum of the
	quad += q2 ;    // two vertices quadrics

	VEC3 res ;
	bool opt = quad.findOptimizedPos(res) ; // try to compute an optimized position for the contraction of this edge
	if(!opt)
	{
		VEC3 p1 = this->m_attrV[0]->operator[](d) ;    // let the new vertex lie
		VEC3 p2 = this->m_attrV[0]->operator[](dd) ;   // on either one of the two endpoints
		VEC3 p12 = (p1 + p2) / 2.0f ;   // or the middle of the edge
		REAL e1 = quad(p1) ;
		REAL e2 = quad(p2) ;
		REAL e12 = quad(p12) ;
		REAL minerr = std::min(std::min(e1, e2), e12) ; // consider only the one for
		if(minerr == e12) this->m_approx[0][d] = p12 ;             // which the error is minimal
		else if(minerr == e1) this->m_approx[0][d] = p1 ;
		else this->m_approx[0][d] = p2 ;
	}
	else
		this->m_approx[0][d] = res ;
}

/************************************************************************************
 *                            QUADRIC ERROR METRIC (for half-edge criteria)         *
 ************************************************************************************/

template <typename PFP>
bool Approximator_QEMhalfEdge<PFP>::init()
{
	m_quadric = this->m_map.template getAttribute<Utils::Quadric<REAL>, VERTEX, MAP>("QEMquadric") ;

	if(this->m_predictor)
	{
		return false ;
	}
	return true ;
}

template <typename PFP>
void Approximator_QEMhalfEdge<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	// get some darts
	Dart dd = m.phi2(d) ;

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
	if(!opt)
		this->m_approx[0][d] = this->m_attrV[0]->operator[](d) ;
	else
		this->m_approx[0][d] = res ;
}

/************************************************************************************
 *							         MID EDGE                                       *
 ************************************************************************************/

template <typename PFP>
bool Approximator_MidEdge<PFP>::init()
{
	if(this->m_predictor)
	{
		if(! (	this->m_predictor->getType() == P_TangentPredict1
			 || this->m_predictor->getType() == P_TangentPredict2 ) )
		{
			return false ;
		}
	}
	return true ;
}

template <typename PFP>
void Approximator_MidEdge<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	// get some darts
	Dart dd = m.phi2(d) ;

	// get the contracted edge vertices positions
	VEC3 v1 = this->m_attrV[0]->operator[](d) ;
	VEC3 v2 = this->m_attrV[0]->operator[](dd) ;

	// Compute the approximated position
	this->m_approx[0][d] = (v1 + v2) / REAL(2) ;

	if(this->m_predictor)
	{
		Dart dd = m.phi2(d) ;
		Dart d2 = m.phi2(m.phi_1(d)) ;
		Dart dd2 = m.phi2(m.phi_1(dd)) ;

		// VEC3 v2 = this->m_attrV[0]->operator[](dd) ;

		// temporary edge collapse
		m.extractTrianglePair(d) ;
//		unsigned int newV = m.template setOrbitEmbeddingOnNewCell<VERTEX>(d2) ;
		unsigned int newV = Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(m,d2);
		this->m_attrV[0]->operator[](newV) = this->m_approx[0][d] ;

		// compute the detail vector
		this->m_predictor->predict(d2, dd2) ;
		this->m_detail[0][d] = v1 - this->m_predictor->getPredict(0) ;

		// vertex split to reset the initial connectivity and embeddings
		m.insertTrianglePair(d, d2, dd2) ;
//		m.template setOrbitEmbedding<VERTEX>(d, m.template getEmbedding<VERTEX>(d)) ;
//		m.template setOrbitEmbedding<VERTEX>(dd, m.template getEmbedding<VERTEX>(dd)) ;
		Algo::Topo::setOrbitEmbedding<VERTEX>(m, d, m.template getEmbedding<VERTEX>(d)) ;
		Algo::Topo::setOrbitEmbedding<VERTEX>(m, dd, m.template getEmbedding<VERTEX>(dd)) ;
	}
}

/************************************************************************************
 *							       HALF COLLAPSE                                    *
 ************************************************************************************/

template <typename PFP>
bool Approximator_HalfCollapse<PFP>::init()
{
	if(this->m_predictor)
	{
		if(! ( this->m_predictor->getType() == P_HalfCollapse ) )
		{
			return false ;
		}
	}
	return true ;
}

template <typename PFP>
void Approximator_HalfCollapse<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	for (unsigned int i = 0 ; i < this->m_attrV.size() ; ++i)
		this->m_approx[i][d] = this->m_attrV[i]->operator[](d) ;

	if(this->m_predictor)
	{
		Dart dd = m.phi2(d) ;
		Dart d2 = m.phi2(m.phi_1(d)) ;
		Dart dd2 = m.phi2(m.phi_1(dd)) ;

		VEC3 v2 = this->m_attrV[0]->operator[](dd) ;

		// temporary edge collapse
		m.extractTrianglePair(d) ;
//		unsigned int newV = m.template setOrbitEmbeddingOnNewCell<VERTEX>(d2) ;
		unsigned int newV = Algo::Topo::setOrbitEmbeddingOnNewCell<VERTEX>(m,d2) ;
		for (unsigned int i = 0 ; i < this->m_attrV.size() ; ++i)
		{
			this->m_attrV[i]->operator[](newV) = this->m_approx[i][d] ;
		}

		// compute the detail vector
		this->m_predictor->predict(d2, dd2) ;
		for (unsigned int i = 0 ; i < this->m_attrV.size() ; ++i)
		{
			this->m_detail[i][d] = v2 - this->m_predictor->getPredict(1) ;
		}

		// vertex split to reset the initial connectivity and embeddings
		m.insertTrianglePair(d, d2, dd2) ;
//		m.template setOrbitEmbedding<VERTEX>(d, m.template getEmbedding<VERTEX>(d)) ;
//		m.template setOrbitEmbedding<VERTEX>(dd, m.template getEmbedding<VERTEX>(dd)) ;
		Algo::Topo::setOrbitEmbedding<VERTEX>(m, d, m.template getEmbedding<VERTEX>(d)) ;
		Algo::Topo::setOrbitEmbedding<VERTEX>(m, dd, m.template getEmbedding<VERTEX>(dd)) ;

	}
}

/************************************************************************************
 *							      CORNER CUTTING                                    *
 ************************************************************************************/

template <typename PFP>
bool Approximator_CornerCutting<PFP>::init()
{
	if(this->m_predictor)
	{
		if(! ( this->m_predictor->getType() == P_CornerCutting ) )
		{
			return false ;
		}
	}
	return true ;
}

template <typename PFP>
void Approximator_CornerCutting<PFP>::approximate(Dart d)
{
	MAP& m = this->m_map ;

	// get some darts
	Dart dd = m.phi2(d) ;
	// Dart d1 = m.phi2(m.phi1(d)) ;
	Dart d2 = m.phi2(m.phi_1(d)) ;
	// Dart dd1 = m.phi2(m.phi1(dd)) ;
	Dart dd2 = m.phi2(m.phi_1(dd)) ;

	// get the contracted edge vertices positions
	VEC3 v1 = this->m_attrV[0]->operator[](d) ;
	VEC3 v2 = this->m_attrV[0]->operator[](dd) ;

	// compute the alpha value according to vertices valences
	REAL k1 = 0 ;
	Dart it = d ;
	do
	{
		++k1 ;
		it = m.phi2_1(it) ;
	} while(it != d) ;
	REAL k2 = 0 ;
	it = dd ;
	do
	{
		++k2 ;
		it = m.phi2_1(it) ;
	} while(it != dd) ;
	REAL alpha = (k1-1) * (k2-1) / (k1*k2-1) ;

	// Compute the mean of v1 half-ring
	VEC3 m1(0) ;
	unsigned int count = 0 ;
	it = d2 ;
	do
	{
		m1 += this->m_attrV[0]->operator[](m.phi1(it));
		it = m.phi2_1(it) ;
		++count ;
	} while (it != d) ;
	m1 /= REAL(count) ;

	// Compute the mean of v2 half-ring
	VEC3 m2(0) ;
	count = 0 ;
	it = dd2 ;
	do
	{
		m2 += this->m_attrV[0]->operator[](m.phi1(it));
		it = m.phi2_1(it) ;
		++count ;
	} while (it != dd) ;
	m2 /= REAL(count) ;

	// Compute the a1 approximation
	VEC3 a1 = ( REAL(1) / (REAL(1) - alpha) ) * ( v1 - (alpha * m1) ) ;
	// Compute the a2 approximation
	VEC3 a2 = ( REAL(1) / (REAL(1) - alpha) ) * ( v2 - (alpha * m2) ) ;

	// Compute the final approximated position
	this->m_approx[0][d] = (a1 + a2) / REAL(2) ;

	if(this->m_predictor)
	{
		this->m_detail[0][d] = (REAL(1) - alpha) * ( (a1 - a2) / REAL(2) ) ;
	}
}

/************************************************************************************
 *                            NORMAL AREA METRIC                                    *
 ************************************************************************************/
template <typename PFP>
bool Approximator_NormalArea<PFP>::init()
{
	edgeMatrix = this->m_map.template getAttribute<Geom::Matrix<3,3,REAL>, EDGE, MAP>("NormalAreaMatrix") ;
	assert(edgeMatrix.isValid());

	if(this->m_predictor)
	{
		return false ;
	}
	return true ;
}

template <typename PFP>
void Approximator_NormalArea<PFP>::approximate(Dart d)
{
	typedef typename PFP::REAL REAL;
	typedef Geom::Matrix<3,3,REAL> MATRIX;
	typedef Eigen::Matrix<REAL,3,1> E_VEC3;
	typedef Eigen::Matrix<REAL,3,3> E_MATRIX;

	MAP& m = this->m_map ;
	Dart dd = m.phi2(d);
	MATRIX M1; // init zero included
	MATRIX M2; // init zero included

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

	const VEC3 & v1 = (*this->m_attrV[0])[d] ;
	const VEC3 & v2 = (*this->m_attrV[0])[dd] ;

	/* version plus sûre : sans cast avec recopie
	E_MATRIX A ;
	A << M1(0,0)+M2(0,0) , M1(0,1)+M2(0,1) , M1(0,2)+M2(0,2) , M1(1,0)+M2(1,0) , M1(1,1)+M2(1,1) , M1(1,2)+M2(1,2) , M1(2,0)+M2(2,0) , M1(2,1)+M2(2,1) , M1(2,2)+M2(2,2) ;

	VEC3 mb = M1*v1 + M2*v2 ;
	E_VEC3 b (mb[0],mb[1],mb[2]);

	Eigen::LDLT<E_MATRIX> decompo (A);
	E_VEC3 x = decompo.solve(b);

	this->m_approx[0][d] = VEC3 (x(0),x(1),x(2)) ;
	 */

	/* version legerement moins gourmande et plus risquee : avec cast sans recopie */
	VEC3 mb = M1*v1 + M2*v2 ;
	M1 += M2;

	Eigen::LDLT<E_MATRIX> decompo (Utils::convertRef<E_MATRIX>(M1));
	E_VEC3 x = decompo.solve(Utils::convertRef<E_VEC3>(mb));

	this->m_approx[0][d] = Utils::convertRef<VEC3>(x) ;
}


} //namespace Decimation

}

} //namespace Algo

} //namespace CGoGN
