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

#ifndef QUADRICLF_H_
#define QUADRICLF_H_


#include "Geometry/vector_gen.h"
#include "Geometry/matrix.h"
#include "Utils/cgognStream.h"

using namespace CGoGN;

enum channel {RED=0, GREEN, BLUE};
#define COLCHANNELS 3

template <typename REAL>
class QuadricRGBfunctions {
public:
    typedef typename Geom::Vector<3,REAL>::type VEC3;
    typedef typename Geom::Vector<6,REAL>::type VEC6;
	typedef Geom::Matrix<6,6,REAL> MATRIX66;

	typedef Geom::Matrix<3,3,REAL> MATRIX33;
	typedef Geom::Matrix<3,6,REAL> MATRIX36;

private:
	MATRIX66 A;
	VEC6 b[COLCHANNELS];
	REAL c[COLCHANNELS];


public:
	static std::string CGoGNnameOfType() ;

	QuadricRGBfunctions();
	QuadricRGBfunctions(int i);
	QuadricRGBfunctions(const QuadricRGBfunctions&);
	QuadricRGBfunctions(const MATRIX36&, const REAL gamma = REAL(0), const REAL alpha = REAL(0)) ;

	virtual ~QuadricRGBfunctions() {} ;

	REAL operator() (const MATRIX36&) const;

	bool findOptimizedRGBfunctions(MATRIX36& lff) const;

	void operator += (const QuadricRGBfunctions&) ;
	void operator -= (const QuadricRGBfunctions&) ;
	void operator *= (const REAL v) ;
	void operator /= (const REAL v) ;

	void zero () ;

	friend std::ostream& operator<< (std::ostream &out, const QuadricRGBfunctions& q) {
		out << "quadricRGBf : " << std::endl ;
		out << "q.A" << "= " << q.A << std::endl ;
		for (unsigned int i = 0 ; i < 3 ; ++i) {
			out << "q.b["<<i<<"] = " << q.b[i] << std::endl ;
			out << "q.c["<<i<<"] = " << q.c[i] << std::endl ;
		}
		return out ;
	} ;
	friend std::istream& operator>> (std::istream &in, const QuadricRGBfunctions&) {return in;};

private :
	void buildIntegralMatrix_A(MATRIX66 &, const REAL alpha) const;
	void buildIntegralMatrix_b(MATRIX66 &, const REAL alpha) const;
	void buildIntegralMatrix_c(MATRIX66 &, const REAL alpha) const;
	void buildRotateMatrix(MATRIX66 &N, const REAL gamma) const;

};


#include "quadricRGBfunctions.hpp"

#endif /* QUADRICLF_H_ */
