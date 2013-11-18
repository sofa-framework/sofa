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

#ifndef QUADRICLF_HPP_
#define QUADRICLF_HPP_

template <typename REAL>
std::string QuadricRGBfunctions<REAL>::CGoGNnameOfType() {
	return std::string("QuadricColFuncs");
}

template <typename REAL>
QuadricRGBfunctions<REAL>::QuadricRGBfunctions() {
	for (unsigned i = 0; i < 6; ++i)
		for (unsigned j = 0; j < 6; ++j)
			A(i,j) = REAL(0) ;

	for (unsigned col = RED; col < BLUE+1 ; ++col) {
		for (unsigned i = 0; i < 6; ++i) {
			b[col][i] = REAL(0) ;
		}

		c[col] = REAL(0);
	}
}

template <typename REAL>
QuadricRGBfunctions<REAL>::QuadricRGBfunctions(int i) {
	QuadricRGBfunctions();
}

template <typename REAL>
QuadricRGBfunctions<REAL>::QuadricRGBfunctions(const QuadricRGBfunctions& q) {
	for (unsigned i = 0; i < 6; ++i)
		for (unsigned j = 0; j < 6; ++j)
			A(i,j) = q.A(i,j) ;

	for (unsigned col = RED; col < BLUE+1 ; ++col) {

		for (unsigned i = 0; i < 6; ++i) {
			b[col][i] = q.b[col][i] ;
		}

		c[col] = q.c[col] ;
	}
}

template <typename REAL>
QuadricRGBfunctions<REAL>::QuadricRGBfunctions(const MATRIX36& cf, const REAL gamma, const REAL alpha) {
	MATRIX66 R1,R2_b,R2_c ;

	buildRotateMatrix(R1,gamma); // Rotation 1

	buildIntegralMatrix_A(A,alpha); // Parameterized integral matrix A
	buildIntegralMatrix_b(R2_b,alpha); // Parameterized integral matrix b
	buildIntegralMatrix_c(R2_c,alpha); // Parameterized integral matrix c

	// Quadric (A,b,c) => L*A*Lt - 2*b*Lt + c = ERROR
	for (unsigned col = RED ; col < BLUE+1 ; ++col) {

		Geom::Vector<6,REAL> function ; // get function coefficients
		if (!cf.getSubVectorH(col,0,function))
			assert(!"QuadricRGBfunctions::constructor") ;

		function = R1 * function ; // Rotation 1

		b[col] = R2_b * function ;	// Vector b : integral + rotation on 1 vector
		c[col] = function * (R2_c * function) ;	// Scalar c : integral + rotation on 2 vectors
	}
}

template <typename REAL>
REAL QuadricRGBfunctions<REAL>::operator() (const MATRIX36& cf) const {
	REAL res = REAL(0);

	for (unsigned col = RED; col < BLUE+1; ++col) {
		Geom::Vector<6,REAL> function ; // Get function coefficients
		if (!cf.getSubVectorH(col,0,function))
			assert (!"QuadricRGBfunctions::getSubVectorH") ;

		REAL res_local = REAL(0) ;
		res_local += function * (A * function) ; // l*A*lt
		res_local -= 2 * (function * b[col]) ; // -2*l*b
		res_local += c[col] ; // c
		// res = l*A*lT - 2*l*b + c
		res += res_local;
	}

	return res;
}

template <typename REAL>
bool QuadricRGBfunctions<REAL>::findOptimizedRGBfunctions(MATRIX36& cf) const {
	MATRIX66 Ainv ;

	REAL det = A.invert(Ainv) ; // Invert matrix
	if(det > -1e-8 && det < 1e-8)
		return false ; // invert failed
	Ainv.transpose() ;

	for (unsigned  col = RED; col < BLUE+1 ; ++col) {
		VEC6 function = Ainv * b[col]; // function = A^(-1) * b

		if (!cf.setSubVectorH(col,0,function)) // save in argument cf
			assert (!"QuadricRGBfunctions::findOptimizedRGBfunctions(cf) setSubVector failed") ;
	}

	return true;
}

template <typename REAL>
void QuadricRGBfunctions<REAL>::buildIntegralMatrix_A(MATRIX66 &M, const REAL alpha) const {
	// Int(phi=0..pi)(theta=0..pi-alpha) variables^2 dTheta dPhi      if alpha > 0
	// Int(phi=0..pi)(theta= -alpha..pi) variables^2 dTheta dPhi      if alpha < 0

	REAL alphaAbs = alpha > 0 ? alpha : -alpha;

	const REAL pi = 3.141592 ;

	const REAL cosinus = cos(alpha);
	const REAL cos2 = cosinus*cosinus;
	const REAL cos3 = cos2*cosinus;
	const REAL sinus = sin(alpha);
	const REAL sinAbs = sin(alphaAbs); // = - sin(alpha) si alpha < 0

	// Line 1
	M(0,0) = 2.0*(pi-alphaAbs)/5.0f;
	M(0,1) = 2.0 * (pi - alphaAbs - cosinus*sinAbs) / 15.0 ;
	M(0,2) = 0;
	M(0,3) = 0;
	M(0,4) = sinAbs*pi / 8.0;
	M(0,5) = 2.0*(pi-alphaAbs)/3.0;

	// Line 2
	M(1,0) = M(0,1);
	M(1,1) = (-4.0 * sinAbs*cos3 + 6.0 * (pi - cosinus*sinAbs - alphaAbs)) / 15.0 ;
	M(1,2) = REAL(0);
	M(1,3) = REAL(0);
	M(1,4) = (sinus*cos2*pi + 2*sinus*pi) / 8.0;
	M(1,5) = 2.0*(pi-cosinus*sinAbs-alphaAbs)/3.0;

	// LINE 3
	M(2,0) = REAL(0);
	M(2,1) = REAL(0);
	M(2,2) = 2.0*(pi - alphaAbs - cosinus*sinAbs)/15.0;
	M(2,3) = pi*sinus/8.0f;
	M(2,4) = REAL(0) ;
	M(2,5) = REAL(0) ;

	// Line 4
	M(3,0) = REAL(0) ;
	M(3,1) = REAL(0) ;
	M(3,2) = M(2,3);
	M(3,3) = 2.0*(pi-alphaAbs)/3.0 ;
	M(3,4) = REAL(0) ;
	M(3,5) = REAL(0) ;

	// Line 5
	M(4,0) = M(0,4) ;
	M(4,1) = M(1,4) ;
	M(4,2) = REAL(0) ;
	M(4,3) = REAL(0) ;
	M(4,4) = 2.0 * (pi - cosinus*sinAbs - alphaAbs ) / 3.0 ;
	M(4,5) = pi*sinus / 2.0 ;

	// Line 6
	M(5,0) = M(0,5) ;
	M(5,1) = M(1,5) ;
	M(5,2) = REAL(0) ;
	M(5,3) = REAL(0) ;
	M(5,4) = M(4,5) ;
	M(5,5) = 2.0*(pi-alphaAbs);
}


template <typename REAL>
void QuadricRGBfunctions<REAL>::buildIntegralMatrix_b(MATRIX66 &M, const REAL alpha) const {
	// Int(phi=0..pi)(theta=0..pi-alpha) variables*variablesRotated dTheta dPhi * coefs     if alpha > 0
	// Int(phi=0..pi)(theta= -alpha..pi) variables*variablesRotated dTheta dPhi * coefs     if alpha < 0
	REAL alphaAbs = alpha > 0 ? alpha : -alpha;

	const REAL pi = 3.141592 ;

	const REAL cosinus = cos(alpha) ;
	const REAL cos2 = cosinus*cosinus ;
	const REAL cos3 = cos2*cosinus ;
	const REAL cos4 = cos3*cosinus ;
	const REAL cos5 = cos4*cosinus ;
	const REAL sinus = sin(alpha) ;
	const REAL sinAbs = sin(alphaAbs) ; // = - sin(alpha) si alpha < 0

	// Line 1
	M(0,0) = 2.0*(pi-alphaAbs)/5.0f;
	M(0,1) = ( 6*cosinus*sinAbs - 8*sinAbs*cos3 - 2*alphaAbs + 2*pi ) / 15.0 ;
	M(0,2) = 0;
	M(0,3) = 0;
	M(0,4) = (sinus*pi + 2*pi*cosinus*sinus) / 8.0 ;
	M(0,5) = 2.0*(pi-alphaAbs)/3.0;

	// Line 2
	M(1,0) = 2 * (pi - cosinus*sinAbs - alphaAbs ) / 15.0 ;
	M(1,1) = ( 6*cosinus*sinAbs - 2*alphaAbs + 2*pi - 16*sinAbs*cos5+4*cos2*pi  - 4*alphaAbs * cos2 ) / 15.0 ;
	M(1,2) = 0;
	M(1,3) = 0;
	M(1,4) = (sinus*pi + 2* (pi*sinus*cos3+pi*cosinus*sinus)) / 8.0 ;
	M(1,5) = 2.0*(pi-cosinus*sinAbs-alphaAbs) / 3.0;

	// LINE 3
	M(2,0) = REAL(0);
	M(2,1) = REAL(0);
	M(2,2) = 2 * (sinAbs - cosinus*alphaAbs + cosinus*pi - 2*cos2*sinAbs) / 15.0 ;
	M(2,3) = pi*sinus / 8.0f;
	M(2,4) = REAL(0) ;
	M(2,5) = REAL(0) ;

	// Line 4
	M(3,0) = REAL(0) ;
	M(3,1) = REAL(0) ;
	M(3,2) = ( sinus*pi + 2*pi*cosinus*sinus ) / 8.0 ;
	M(3,3) = 2.0*(pi-alphaAbs)/3.0 ;
	M(3,4) = REAL(0) ;
	M(3,5) = REAL(0) ;

	// Line 5
	M(4,0) = pi*sinus / 8.0 ;
	M(4,1) = (sinus*pi + 4 * sinus * cos4 + 2*pi*cosinus*sinus) / 8.0 ;
	M(4,2) = REAL(0) ;
	M(4,3) = REAL(0) ;
	M(4,4) = 2*(sinAbs - cosinus*alphaAbs + cosinus*pi - 2*cos2*sinAbs) / 3.0 ;
	M(4,5) = pi*sinus / 2.0 ;

	// Line 6
	M(5,0) = M(0,5) ;
	M(5,1) = 2*cosinus*sinAbs + 2*(pi-4*sinAbs*cos3-alphaAbs) / 3.0 ;
	M(5,2) = REAL(0) ;
	M(5,3) = REAL(0) ;
	M(5,4) = sinus*pi / 2.0 + pi*cosinus*sinus ;
	M(5,5) = 2.0*(pi-alphaAbs);
}

template <typename REAL>
void QuadricRGBfunctions<REAL>::buildIntegralMatrix_c(MATRIX66 &M, const REAL alpha) const {
	// coefs * Int(phi=0..pi)(theta=0..pi-alpha) variablesRotated^2 dTheta dPhi * coefs     if alpha > 0
	// coefs * Int(phi=0..pi)(theta= -alpha..pi) variablesRotated^2 dTheta dPhi * coefs     if alpha < 0

	REAL alphaAbs = alpha > 0 ? alpha : -alpha;

	const REAL pi = 3.141592 ;

	const REAL cosinus = cos(alpha);
	const REAL cos2 = cosinus*cosinus ;
	const REAL cos3 = cos2*cosinus ;
	const REAL cos5 = cos2*cos3 ;
	const REAL cos7 = cos2*cos5 ;
	const REAL sinus = sin(alpha);
	const REAL sinAbs = sin(alphaAbs); // = - sin(alpha) si alpha < 0

	// Line 1
	M(0,0) = 2.0*(pi-alphaAbs)/5.0f;
	M(0,1) = 2.0 * (3*cosinus*sinAbs-4*sinAbs*cos3-alphaAbs+pi) / 15.0 ;
	M(0,2) = 0;
	M(0,3) = 0;
	M(0,4) = (sinus*pi + 2*pi*cosinus*sinus) / 8.0;
	M(0,5) = 2.0*(pi-alphaAbs)/3.0;

	// Line 2
	M(1,0) = M(0,1);
	M(1,1) = (96*sinAbs*cos5 - 64*cos7*sinAbs+26*cosinus*sinAbs-68*sinAbs*cos3-6*alphaAbs+6*pi) / 15.0 ;
	M(1,2) = 0;
	M(1,3) = 0;
	M(1,4) = (2*sinus*pi + pi * sinus*cos5 - pi*sinus*cos3 + 6*pi*cosinus*sinus + sinus*cos2*pi) / 8.0 ;
	M(1,5) = 2*cosinus*sinAbs + (2*pi-8*sinAbs*cos3 - 2*alphaAbs) / 3.0 ;

	// LINE 3
	M(2,0) = REAL(0);
	M(2,1) = REAL(0);
	M(2,2) = 2.0*(3*cosinus*sinAbs - 4*sinAbs*cos3 - alphaAbs + pi)/15.0;
	M(2,3) = pi*sinus/8.0f + cosinus*sinus*pi / 4.0f;
	M(2,4) = REAL(0) ;
	M(2,5) = REAL(0) ;

	// Line 4
	M(3,0) = REAL(0) ;
	M(3,1) = REAL(0) ;
	M(3,2) = M(2,3);
	M(3,3) = 2.0*(pi-alphaAbs)/3.0 ;
	M(3,4) = REAL(0) ;
	M(3,5) = REAL(0) ;

	// Line 5
	M(4,0) = M(0,4) ;
	M(4,1) = M(1,4) ;
	M(4,2) = REAL(0) ;
	M(4,3) = REAL(0) ;
	M(4,4) = M(1,5) ;
	M(4,5) = pi*sinus / 2.0 + pi*cosinus*sinus ;

	// Line 6
	M(5,0) = M(0,5) ;
	M(5,1) = M(1,5) ;
	M(5,2) = REAL(0) ;
	M(5,3) = REAL(0) ;
	M(5,4) = M(4,5) ;
	M(5,5) = 2.0*(pi-alphaAbs);
}


template <typename REAL>
void QuadricRGBfunctions<REAL>::buildRotateMatrix(MATRIX66 &N, const REAL gamma) const {
	REAL cosinus = cos(gamma), cos2 = cosinus*cosinus;
	REAL sinus = sin(gamma), sin2 = sinus*sinus;
	REAL sincos = sinus*cosinus;

	N(0,0) = cos2;
	N(0,1) = sin2;
	N(0,2) = 2*sincos;
	N(0,3) = REAL(0);
	N(0,4) = REAL(0);
	N(0,5) = REAL(0);

	N(1,0) = sin2;
	N(1,1) = cos2;
	N(1,2) = -2*sincos;
	N(1,3) = REAL(0);
	N(1,4) = REAL(0);
	N(1,5) = REAL(0);

	N(2,0) = -sincos;
	N(2,1) = sincos;
	N(2,2) = cos2-sin2;
	N(2,3) = REAL(0);
	N(2,4) = REAL(0);
	N(2,5) = REAL(0);

	N(3,0) = REAL(0);
	N(3,1) = REAL(0);
	N(3,2) = REAL(0);
	N(3,3) = cosinus;
	N(3,4) = sinus;
	N(3,5) = REAL(0);

	N(4,0) = REAL(0);
	N(4,1) = REAL(0);
	N(4,2) = REAL(0);
	N(4,3) = -sinus;
	N(4,4) = cosinus;
	N(4,5) = REAL(0);

	N(5,0) = REAL(0);
	N(5,1) = REAL(0);
	N(5,2) = REAL(0);
	N(5,3) = REAL(0);
	N(5,4) = REAL(0);
	N(5,5) = REAL(1);
}

template <typename REAL>
void QuadricRGBfunctions<REAL>::operator += (const QuadricRGBfunctions& q) {
	for (unsigned i = 0; i < 6; ++i)
		for (unsigned j = 0; j < 6; ++j)
			A(i,j) += q.A(i,j);

	for (unsigned col = RED; col < BLUE+1 ; ++col) {

		for (unsigned i = 0; i < 6; ++i) {
			b[col][i] += q.b[col][i];
		}
		c[col] += q.c[col];

	}
}

template <typename REAL>
void QuadricRGBfunctions<REAL>::operator -= (const QuadricRGBfunctions& q) {
	for (unsigned i = 0; i < 6; ++i)
		for (unsigned j = 0; j < 6; ++j)
			A(i,j) -= q.A(i,j);

	for (unsigned col = RED; col < BLUE+1 ; ++col) {
		for (unsigned i = 0; i < 6; ++i) {
			b[col][i] -= q.b[col][i];
		}

		c[col] -= q.c[col];
	}
}

template <typename REAL>
void QuadricRGBfunctions<REAL>::operator *= (const REAL v) {
	for (unsigned i = 0; i < 6; ++i)
		for (unsigned j = 0; j < 6; ++j)
			A(i,j) *= v;

	for (unsigned col = RED; col < BLUE+1 ; ++col) {
		for (unsigned i = 0; i < 6; ++i) {
			b[col][i] *= v;
		}

		c[col] *= v;
	}
}

template <typename REAL>
void QuadricRGBfunctions<REAL>::operator /= (const REAL v) {
	if (v==REAL(0))
		return ;
	for (unsigned i = 0; i < 6; ++i)
		for (unsigned j = 0; j < 6; ++j)
			A(i,j) /= v;

	for (unsigned col = RED; col < BLUE+1 ; ++col) {
		for (unsigned i = 0; i < 6; ++i) {
			b[col][i] /= v;
		}

		c[col] /= v;
	}
}

template <typename REAL>
void QuadricRGBfunctions<REAL>::zero () {
	A.zero();
	for (unsigned int i = 0 ; i < COLCHANNELS ; ++i) {
		b[i].zero();
		c[i] = REAL(0) ;
	}
}

#endif /* QUADRICLF_HPP_ */
