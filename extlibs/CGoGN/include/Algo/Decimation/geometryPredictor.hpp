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
 *							       HALF COLLAPSE                                    *
 ************************************************************************************/

template <typename PFP>
void Predictor_HalfCollapse<PFP>::predict(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;

	this->m_predict.clear() ;

	// get some darts
	// Dart d1 = m.phi2(d2) ;
	// Dart dd1 = m.phi2(dd2) ;

	REAL k2 = REAL(1) ;
	VEC3 s2_1(0) ;
	Dart it = dd2 ;
	do {
		s2_1 += this->m_attrV[m.phi1(it)] ;
		it = m.phi2_1(it) ;
		++k2 ;
	} while (m.phi2(it) != m.phi_1(d2)) ;

	// get the current coarse position
	VEC3 a = this->m_attrV[d2] ;

	s2_1 += a ;
	s2_1 /= k2 ;

	this->m_predict.push_back(a) ;
	this->m_predict.push_back(s2_1) ;
}

/************************************************************************************
 *							      CORNER CUTTING                                    *
 ************************************************************************************/

template <typename PFP>
typename PFP::REAL Predictor_CornerCutting<PFP>::autoAlpha(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;

	// get some darts
	// Dart d1 = m.phi2(d2) ;
	// Dart dd1 = m.phi2(dd2) ;

	REAL k1 = 2 ;				// compute the alpha
	REAL k2 = 2 ;				// value according to
	Dart it = d2 ;				// vertices valences
	do
	{
		++k1 ;
		it = m.phi2_1(it) ;
	} while(it != dd2) ;
	do
	{
		++k2 ;
		it = m.phi2_1(it) ;
	} while(it != d2) ;
	return (k1-1) * (k2-1) / (k1*k2-1) ;
}

template <typename PFP>
void Predictor_CornerCutting<PFP>::predict(Dart d2, Dart dd2)
{
	MAP& m = this->m_map ;

	this->m_predict.clear() ;

	// get some darts
	// Dart d1 = m.phi2(d2) ;
	// Dart dd1 = m.phi2(dd2) ;

	REAL alpha = autoAlpha(d2, dd2) ;

	// get the current coarse position
	VEC3 a = this->m_attrV[d2] ;

	// Compute the mean of v1 half-ring
	VEC3 m1(0) ;
	unsigned int count = 0 ;
	Dart it = d2 ;
	do {
		m1 += this->m_attrV[m.phi1(it)] ;
		it = m.phi2_1(it) ;
		++count ;
	} while (m.phi2(it) != m.phi_1(dd2)) ;
	m1 /= REAL(count) ;

	// compute the predicted position of v1
	this->m_predict.push_back( (( REAL(1) - alpha ) * a) + ( alpha * m1 ) ) ;

	// Compute the mean of v2 half-ring
	VEC3 m2(0) ;
	count = 0 ;
	it = dd2 ;
	do {
		m2 += this->m_attrV[m.phi1(it)] ;
		it = m.phi2_1(it) ;
		++count ;
	} while (m.phi2(it) != m.phi_1(d2)) ;
	m2 /= REAL(count) ;

	// compute the predicted position of v2
	this->m_predict.push_back( (( REAL(1) - alpha ) * a) + ( alpha * m2 ) ) ;
}

/************************************************************************************
 *							     TANGENT PREDICT 1                                  *
 ************************************************************************************/

template <typename PFP>
void Predictor_TangentPredict1<PFP>::predictedTangent(Dart d2, Dart dd2, VEC3& displ, REAL& k1, REAL& k2)
{
	MAP& m = this->m_map ;

	k1 = 1 ;
	k2 = 1 ;

	VEC3 s1_1(0) ;
	Dart it = d2 ;
	do {
		s1_1 += this->m_attrV[m.phi1(it)] ;
		it = m.phi2_1(it) ;
		++k1 ;
	} while (m.phi2(it) != m.phi_1(dd2)) ;

	VEC3 s2_1(0) ;
	it = dd2 ;
	do {
		s2_1 += this->m_attrV[m.phi1(it)] ;
		it = m.phi2_1(it) ;
		++k2 ;
	} while (m.phi2(it) != m.phi_1(d2)) ;

	displ = ( s1_1 / (2*(k1 + 1)) ) - ( s2_1 / (2*(k2 + 1)) ) ;
}

template <typename PFP>
void Predictor_TangentPredict1<PFP>::predict(Dart d2, Dart dd2)
{
	this->m_predict.clear() ;

	VEC3 displ(0) ;
	REAL k1, k2 ;
	predictedTangent(d2, dd2, displ, k1, k2) ;

	// get the current coarse position
	VEC3 a = this->m_attrV[d2] ;

	VEC3 a1 = a * ( REAL(1) + ((k2-k1) / ((k1+1)*(k2+1))) ) ;
	this->m_predict.push_back( a1 + displ ) ;

	VEC3 a2 = a * ( REAL(1) - ((k2-k1) / ((k1+1)*(k2+1))) ) ;
	this->m_predict.push_back( a2 - displ ) ;
}

/************************************************************************************
 *							     TANGENT PREDICT 2                                  *
 ************************************************************************************/

template <typename PFP>
void Predictor_TangentPredict2<PFP>::predictedTangent(Dart d2, Dart dd2, VEC3& displ, REAL& k1, REAL& k2)
{
	MAP& m = this->m_map ;

	// get some darts
	Dart d1 = m.phi2(d2) ;
	Dart dd1 = m.phi2(dd2) ;

	float alpha = 1.15f ;
	k1 = 1 ;
	k2 = 1 ;

	VEC3 s1_1(0) ;
	Dart it = d2 ;
	do {
		s1_1 += this->m_attrV[m.phi1(it)] ;
		it = m.phi2_1(it) ;
		++k1 ;
	} while (m.phi2(it) != m.phi_1(dd2)) ;

	VEC3 s1_2(0) ;
	it = d2 ;
	do {
		Dart p1 = m.phi1(it) ;
		Dart p2 = m.phi2(p1) ;
		if(p2 != p1)
			s1_2 += this->m_attrV[m.phi_1(p2)] ;
		it = m.phi2_1(it) ;
	} while (it != dd2) ;
	s1_2 += this->m_attrV[m.phi_1(d1)] ;
	s1_2 += this->m_attrV[m.phi_1(dd2)] ;

	VEC3 s2_1(0) ;
	it = dd2 ;
	do {
		s2_1 += this->m_attrV[m.phi2(it)] ;
		it = m.phi2_1(it) ;
		++k2 ;
	} while (m.phi2(it) != m.phi_1(d2)) ;

	VEC3 s2_2(0) ;
	it = dd2 ;
	do {
		Dart p1 = m.phi1(it) ;
		Dart p2 = m.phi2(p1) ;
		if(p2 != p1)
			s2_2 += this->m_attrV[m.phi_1(p2)] ;
		it = m.phi2_1(it) ;
	} while (it != d2) ;
	s2_2 += this->m_attrV[m.phi_1(dd1)] ;
	s2_2 += this->m_attrV[m.phi_1(d2)] ;

	VEC3 tmp1 = (alpha * s1_1) + ((REAL(1) - alpha) * s1_2) ;
	tmp1 /= 2 * (k1 + alpha) ;

	VEC3 tmp2 = (alpha * s2_1) + ((REAL(1) - alpha) * s2_2) ;
	tmp2 /= 2 * (k2 + alpha) ;

	displ = tmp1 - tmp2 ;
}

template <typename PFP>
void Predictor_TangentPredict2<PFP>::predict(Dart d2, Dart dd2)
{
	this->m_predict.clear() ;

	VEC3 displ(0) ;
	REAL k1, k2 ;
	predictedTangent(d2, dd2, displ, k1, k2) ;

	// get the current coarse position
	VEC3 a = this->m_attrV[d2] ;

	REAL alpha = 1.15f ;

	VEC3 a1 = a * ( REAL(1) + (alpha * (k2-k1) / ((k1+alpha)*(k2+alpha))) ) ;
	this->m_predict.push_back( a1 + displ ) ;

	VEC3 a2 = a * ( REAL(1) - (alpha * (k2-k1) / ((k1+alpha)*(k2+alpha))) ) ;
	this->m_predict.push_back( a2 - displ ) ;
}

} // namespace Decimation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

