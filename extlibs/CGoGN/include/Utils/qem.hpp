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

#include <cmath>

namespace CGoGN {

namespace Utils {

template <typename REAL>
Quadric<REAL>::Quadric()
{
	A.zero() ;
}

template <typename REAL>
Quadric<REAL>::Quadric(int)
{
	A.zero() ;
}

template <typename REAL>
Quadric<REAL>::Quadric(const VEC3& p1, const VEC3& p2, const VEC3& p3)
{
	// compute the equation of the plane of the 3 points
	Geom::Plane3D<REAL> plane(p1, p2, p3) ;
	const VEC3& n = plane.normal() ;

	Geom::Vector<4,double> p = Geom::Vector<4,double>(n[0],n[1],n[2],plane.d()) ;
	A = Geom::transposed_vectors_mult(p,p) ;
}

template <typename REAL>
void
Quadric<REAL>::zero()
{
	A.zero() ;
}

template <typename REAL>
void
Quadric<REAL>::operator= (const Quadric<REAL>& q)
{
	A = q.A ;
}

template <typename REAL>
Quadric<REAL>&
Quadric<REAL>::operator+= (const Quadric<REAL>& q)
{
	A += q.A ;
	return *this ;
}

template <typename REAL>
Quadric<REAL>&
Quadric<REAL>::operator -= (const Quadric<REAL>& q)
{
	A -= q.A ;
	return *this ;
}

template <typename REAL>
Quadric<REAL>&
Quadric<REAL>::operator *= (const REAL& v)
{
	A *= v ;
	return *this ;
}

template <typename REAL>
Quadric<REAL>&
Quadric<REAL>::operator /= (const REAL& v)
{
	A /= v ;
	return *this ;
}

template <typename REAL>
REAL
Quadric<REAL>::operator() (const VEC4& v) const
{
	return evaluate(v) ;
}

template <typename REAL>
REAL
Quadric<REAL>::operator() (const VEC3& v) const
{
	VEC4 hv(v[0], v[1], v[2], 1.0f) ;
	return evaluate(hv) ;
}

template <typename REAL>
bool
Quadric<REAL>::findOptimizedPos(VEC3& v)
{
	VEC4 hv ;
	bool b = optimize(hv) ;
	if(b)
	{
		v[0] = hv[0] ;
		v[1] = hv[1] ;
		v[2] = hv[2] ;
	}
	return b ;
}

template <typename REAL>
REAL
Quadric<REAL>::evaluate(const VEC4& v) const
{
	// Double computation is crucial for stability
	Geom::Vector<4, double> vd(v);
	Geom::Vector<4, double> Av = A * vd ;
	return REAL(vd * Av) ;
}

template <typename REAL>
bool
Quadric<REAL>::optimize(VEC4& v) const
{
#ifdef WIN32
	if (A(0, 0) != A(0, 0))
#else
	if (std::isnan(A(0, 0)))
#endif
		return false ;

	MATRIX44 A2(A) ;
	for(int i = 0; i < 3; ++i)
		A2(3,i) = 0.0f ;
	A2(3,3) = 1.0f ;
	MATRIX44 Ainv ;
	REAL det = A2.invert(Ainv) ;
	if(det > -1e-6 && det < 1e-6)
		return false ;
//	VEC4 right(0,0,0,1) ;
	Geom::Vector<4, double> right(0,0,0,1) ;
	v = VEC4(Ainv * right) ;

	return true;
}

template <typename REAL, unsigned int N>
QuadricNd<REAL,N>::QuadricNd()
{
	A.zero() ;
	b.zero() ;
	c = 0 ;
}

template <typename REAL, unsigned int N>
QuadricNd<REAL,N>::QuadricNd(int)
{
	A.zero() ;
	b.zero() ;
	c = 0 ;
}

template <typename REAL, unsigned int N>
QuadricNd<REAL,N>::QuadricNd(const VECN& p1_r, const VECN& p2_r, const VECN& p3_r)
{
	const Geom::Vector<N,double>& p1 = p1_r ;
	const Geom::Vector<N,double>& p2 = p2_r ;
	const Geom::Vector<N,double>& p3 = p3_r ;

	Geom::Vector<N,double> e1 = p2 - p1 ; 					e1.normalize() ;
	Geom::Vector<N,double> e2 = (p3 - p1) - (e1*(p3-p1))*e1 ; 	e2.normalize() ;

	A.identity() ;
	A -= Geom::transposed_vectors_mult(e1,e1) + Geom::transposed_vectors_mult(e2,e2) ;

	b = (p1*e1)*e1 + (p1*e2)*e2 - p1 ;

	c = p1*p1 - pow((p1*e1),2) - pow((p1*e2),2) ;
}

template <typename REAL, unsigned int N>
void
QuadricNd<REAL,N>::zero()
{
	A.zero() ;
	b.zero() ;
	c = 0 ;
}

template <typename REAL, unsigned int N>
void
QuadricNd<REAL,N>::operator= (const QuadricNd<REAL,N>& q)
{
	A = q.A ;
	b = q.b ;
	c = q.c ;
}

template <typename REAL, unsigned int N>
QuadricNd<REAL,N>&
QuadricNd<REAL,N>::operator+= (const QuadricNd<REAL,N>& q)
{
	A += q.A ;
	b += q.b ;
	c += q.c ;
	return *this ;
}

template <typename REAL, unsigned int N>
QuadricNd<REAL,N>&
QuadricNd<REAL,N>::operator -= (const QuadricNd<REAL,N>& q)
{
	A -= q.A ;
	b -= q.b ;
	c -= q.c ;
	return *this ;
}

template <typename REAL, unsigned int N>
QuadricNd<REAL,N>&
QuadricNd<REAL,N>::operator *= (REAL v)
{
	A *= v ;
	b *= v ;
	c *= v ;
	return *this ;
}

template <typename REAL, unsigned int N>
QuadricNd<REAL,N>&
QuadricNd<REAL,N>::operator /= (REAL v)
{
	A /= v ;
	b /= v ;
	c /= v ;
	return *this ;
}

template <typename REAL, unsigned int N>
REAL
QuadricNd<REAL,N>::operator() (const VECNp& v) const
{
	VECN hv ;
	for (unsigned int i = 0 ; i < N ; ++i)
		hv[i] = v[i] ;

	return evaluate(v) ;
}

template <typename REAL, unsigned int N>
REAL
QuadricNd<REAL,N>::operator() (const VECN& v) const
{
	return evaluate(v) ;
}

template <typename REAL, unsigned int N>
bool
QuadricNd<REAL,N>::findOptimizedVec(VECN& v)
{
	return optimize(v) ;
}

template <typename REAL, unsigned int N>
REAL
QuadricNd<REAL,N>::evaluate(const VECN& v) const
{
	Geom::Vector<N, double> v_d = v ;
	return v_d*A*v_d + 2.*(b*v_d) + c ;
}

template <typename REAL, unsigned int N>
bool
QuadricNd<REAL,N>::optimize(VECN& v) const
{
#ifdef WIN32
	if (A(0, 0) != A(0, 0))
#else
	if (std::isnan(A(0, 0)))
#endif
		return false ;

	Geom::Matrix<N,N,double> Ainv ;
	double det = A.invert(Ainv) ;

	if(det > -1e-6 && det < 1e-6)
		return false ;

	v.zero() ;
	v -= Ainv * b ;

	return true ;
}

template <typename REAL>
QuadricHF<REAL>::QuadricHF():
m_noAlphaRot(false)
{}

template <typename REAL>
QuadricHF<REAL>::QuadricHF(int i):
m_noAlphaRot(false)
{
	m_A.resize(i,i) ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		m_b[c].resize(i) ;
		m_c[c] = 0 ;
	}
}

template <typename REAL>
QuadricHF<REAL>::QuadricHF(const std::vector<VEC3>& v, const REAL& gamma, const REAL& alpha)
{
	Geom::Tensor3d *T = tensorsFromCoefs(v) ;
	*this = QuadricHF(T, gamma, alpha) ;
	delete[] T ;
}

template <typename REAL>
QuadricHF<REAL>::QuadricHF(const Geom::Tensor3d* T, const REAL& gamma, const REAL& alpha):
m_noAlphaRot(fabs(alpha) < 1e-13)
{
	const unsigned int nbcoefs = ((T[0].order() + 1) * (T[0].order() + 2)) / 2. ;

	// 2D rotation
	const Geom::Matrix33d R = buildRotateMatrix(gamma) ;
	Geom::Tensor3d* Trot = new Geom::Tensor3d[3] ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
		Trot[c] = rotate(T[c],R) ;
	m_coefs = coefsFromTensors(Trot) ;
	delete[] Trot ;

	// parameterized integral on intersection

	// build A, b and c
	m_A = buildIntegralMatrix_A(alpha,nbcoefs) ; // Parameterized integral matrix A
	Eigen::MatrixXd integ_b = buildIntegralMatrix_B(alpha,nbcoefs) ; // Parameterized integral matrix b
	Eigen::MatrixXd integ_c = buildIntegralMatrix_C(alpha,nbcoefs) ; // Parameterized integral matrix c

	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		Eigen::VectorXd v ;	v.resize(nbcoefs) ;
		for (unsigned int i = 0 ; i < nbcoefs ; ++i) // copy into vector
			v[i] = m_coefs[i][c] ;

		m_b[c] = integ_b * v ; // Vector b
		m_c[c] = v.transpose() * (integ_c * v) ; // Constant c
	}
}

template <typename REAL>
QuadricHF<REAL>::~QuadricHF()
{}

template <typename REAL>
void
QuadricHF<REAL>::zero()
{
	m_A.setZero() ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		m_b[c].setZero() ;
		m_c[c] = 0 ;
	}
	m_coefs.clear() ;
	m_noAlphaRot = false ;
}

template <typename REAL>
QuadricHF<REAL>&
QuadricHF<REAL>::operator= (const QuadricHF<REAL>& q)
{
	m_A = q.m_A ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		m_b[c] = q.m_b[c] ;
		m_c[c] = q.m_c[c] ;
	}
	m_coefs = q.m_coefs ;
	m_noAlphaRot = q.m_noAlphaRot ;

	return *this ;
}

template <typename REAL>
QuadricHF<REAL>&
QuadricHF<REAL>::operator+= (const QuadricHF<REAL>& q)
{
	assert(((m_A.cols() == q.m_A.cols()) && (m_A.rows() == q.m_A.rows()) && m_b[0].size() == q.m_b[0].size()) || !"Incompatible add of matrices") ;
	if (!(m_A.cols() == q.m_A.cols()) && (m_A.rows() == q.m_A.rows()) && (m_b[0].size() == q.m_b[0].size()))
		return *this ;

	m_A += q.m_A ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		m_b[c] += q.m_b[c] ;
		m_c[c] += q.m_c[c] ;
	}
	m_coefs.resize(m_coefs.size()) ;
	for (unsigned int i = 0 ; i < m_coefs.size() ; ++i)
		m_coefs[i] += q.m_coefs[i] ;

	m_noAlphaRot &= q.m_noAlphaRot ;

	return *this ;
}

template <typename REAL>
QuadricHF<REAL>&
QuadricHF<REAL>::operator -= (const QuadricHF<REAL>& q)
{
	assert(((m_A.cols() == q.m_A.cols()) && (m_A.rows() == q.m_A.rows()) && m_b[0].size() == q.m_b[0].size()) || !"Incompatible substraction of matrices") ;
	if ((m_A.cols() == q.m_A.cols()) && (m_A.rows() == q.m_A.rows()) && (m_b[0].size() == q.m_b[0].size()))
		return *this ;

	m_A -= q.m_A ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		m_b[c] -= q.m_b[c] ;
		m_c[c] -= q.m_c[c] ;
	}

	m_coefs.resize(m_coefs.size()) ;
	for (unsigned int i = 0 ; i < m_coefs.size() ; ++i)
		m_coefs[i] -= q.m_coefs[i] ;

	m_noAlphaRot &= q.m_noAlphaRot ;

	return *this ;
}

template <typename REAL>
QuadricHF<REAL>&
QuadricHF<REAL>::operator *= (const REAL& v)
{
	std::cout << "Warning: QuadricHF<REAL>::operator *= should not be used !" << std::endl ;
	m_A *= v ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		m_b[c] *= v ;
		m_c[c] *= v ;
	}

	return *this ;
}

template <typename REAL>
QuadricHF<REAL>&
QuadricHF<REAL>::operator /= (const REAL& v)
{
	std::cout << "Warning: QuadricHF<REAL>::operator /= should not be used !" << std::endl ;
	const REAL& inv = 1. / v ;

	(*this) *= inv ;

	return *this ;
}

template <typename REAL>
REAL
QuadricHF<REAL>::operator() (const std::vector<VEC3>& coefs) const
{
	return evaluate(coefs) ;
}

template <typename REAL>
bool
QuadricHF<REAL>::findOptimizedCoefs(std::vector<VEC3>& coefs)
{
	coefs.resize(m_b[0].size()) ;

	if (fabs(m_A.determinant()) < 1e-10 )
	{
		coefs = m_coefs ;
		return m_noAlphaRot ; // if true inversion failed (normal!) and m_coefs forms a valid solution
	}
	Eigen::MatrixXd Ainv = m_A.inverse() ;

	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		Eigen::VectorXd tmp(m_b[0].size()) ;
		tmp = Ainv * m_b[c] ;
		for (unsigned int i = 0 ; i < m_b[c].size() ; ++i)
			coefs[i][c] = tmp[i] ;
	}

	return true ;
}

template <typename REAL>
REAL
QuadricHF<REAL>::evaluate(const std::vector<VEC3>& coefs) const
{
	VEC3 res ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		Eigen::VectorXd tmp(coefs.size()) ;
		for (unsigned int i = 0 ; i < coefs.size() ; ++i)
            tmp[i] = coefs[i][c] ;
		res[c] = tmp.transpose() * m_A * tmp ;		// A
		res[c] -= 2. * (m_b[c]).transpose() * tmp ;	// - 2b
		res[c] += m_c[c] ;							// + c
	}

	res /= 2*M_PI ; // max integral value over hemisphere

	return (res[0] + res[1] + res[2]) / 3. ;
}

template <typename REAL>
typename QuadricHF<REAL>::VEC3
QuadricHF<REAL>::evalR3(const std::vector<VEC3>& coefs) const
{
	VEC3 res ;
	for (unsigned int c = 0 ; c < 3 ; ++c)
	{
		Eigen::VectorXd tmp(coefs.size()) ;
		for (unsigned int i = 0 ; i < coefs.size() ; ++i)
			tmp[i] = coefs[i][c] ;
		res[c] = tmp.transpose() * m_A * tmp ;		// A
		res[c] -= 2. * (m_b[c]).transpose() * tmp ;	// - 2b
		res[c] += m_c[c] ;							// + c
	}

	res /= 2*M_PI ; // max integral value over hemisphere

	return res ;
}

template <typename REAL>
Geom::Matrix33d
QuadricHF<REAL>::buildRotateMatrix(const REAL& gamma)
{
	Geom::Matrix33d R ;
	R.identity() ;
	R(0,0) = cos(gamma) ;
	R(0,1) = -sin(gamma) ;
	R(1,0) = sin(gamma) ;
	R(1,1) = cos(gamma) ;

	return R ;
}

template <typename REAL>
Geom::Tensor3d
QuadricHF<REAL>::rotate(const Geom::Tensor3d& T, const Geom::Matrix33d& R)
{
	Geom::Tensor3d Tp(T.order()) ;
	std::vector<unsigned int> p ; p.resize(T.order(), 0) ;
	for (unsigned int i = 0 ; i < T.nbElem() ; ++i)
	{
		REAL S = 0 ;
		std::vector<unsigned int> q ; q.resize(T.order(), 0) ;
		for (unsigned int j = 0 ; j < T.nbElem() ; ++j)
		{
			REAL P = T[j] ;
			for (unsigned int k = 0 ; k < T.order() ; ++k)
				P *= R(q[k],p[k]) ;
			S += P ;
			Geom::Tensor3d::incremIndex(q) ;
		}
		Tp[i] = S ;
		Geom::Tensor3d::incremIndex(p) ;
	}

	return Tp ;
}

template <typename REAL>
void
QuadricHF<REAL>::fillSymmetricMatrix(Eigen::MatrixXd& M)
{
	assert(M.cols() == M.rows() || !"QuadricHF<REAL>::fillSymmetricMatrix: matrix to fill should be a square matrix") ;
	for (unsigned int c = 0 ; c < M.cols() ; ++c)
		for (unsigned int l = 0 ; l < c ; ++l)
			M( l, c) = M (c, l) ;
}

template <typename REAL>
Eigen::MatrixXd
QuadricHF<REAL>::buildIntegralMatrix_A(const REAL& alpha, unsigned int size)
{
	Eigen::MatrixXd res = buildLowerLeftIntegralMatrix_A(alpha,size) ;
	fillSymmetricMatrix(res) ;

	return res ;
}

template <typename REAL>
Eigen::MatrixXd
QuadricHF<REAL>::buildIntegralMatrix_C(const REAL& alpha, unsigned int size)
{
	Eigen::MatrixXd res = buildLowerLeftIntegralMatrix_C(alpha,size) ;
	fillSymmetricMatrix(res) ;

	return res ;
}


template <typename REAL>
Eigen::MatrixXd
QuadricHF<REAL>::buildLowerLeftIntegralMatrix_A(const REAL& alpha, unsigned int size)
{
	Eigen::MatrixXd A(size,size) ;

	A( 0 , 0 ) = 2*(M_PI-alpha) ;

	A( 1 , 0 ) = M_PI*sin(alpha)/2.0 ;
	A( 1 , 1 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.0 ;

	A( 2 , 0 ) = 0 ;
	A( 2 , 1 ) = 0 ;
	A( 2 , 2 ) = 2.0*(M_PI-alpha)/3.0 ;

	A( 3 , 0 ) = 0 ;
	A( 3 , 1 ) = 0 ;
	A( 3 , 2 ) = M_PI*sin(alpha)/8.0 ;
	A( 3 , 3 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/1.5E+1 ;

	A( 4 , 0 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.0 ;
	A( 4 , 1 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/8.0 ;
	A( 4 , 2 ) = 0 ;
	A( 4 , 3 ) = 0 ;
	A( 4 , 4 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/3.0E+1 ;

	A( 5 , 0 ) = 2.0*(M_PI-alpha)/3.0 ;
	A( 5 , 1 ) = M_PI*sin(alpha)/8.0 ;
	A( 5 , 2 ) = 0 ;
	A( 5 , 3 ) = 0 ;
	A( 5 , 4 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/1.5E+1 ;
	A( 5 , 5 ) = 2.0*(M_PI-alpha)/5.0 ;

	if (size < 7)
		return A ;

	A( 6 , 0 ) = 0 ;
	A( 6 , 1 ) = 0 ;
	A( 6 , 2 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/1.5E+1 ;
	A( 6 , 3 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/4.8E+1 ;
	A( 6 , 4 ) = 0 ;
	A( 6 , 5 ) = 0 ;
	A( 6 , 6 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/2.1E+2 ;

	A( 7 , 0 ) = M_PI*sin(alpha)/8.0 ;
	A( 7 , 1 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/1.5E+1 ;
	A( 7 , 2 ) = 0 ;
	A( 7 , 3 ) = 0 ;
	A( 7 , 4 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/4.8E+1 ;
	A( 7 , 5 ) = M_PI*sin(alpha)/1.6E+1 ;
	A( 7 , 6 ) = 0 ;
	A( 7 , 7 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.5E+1 ;

	A( 8 , 0 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/8.0 ;
	A( 8 , 1 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/3.0E+1 ;
	A( 8 , 2 ) = 0 ;
	A( 8 , 3 ) = 0 ;
	A( 8 , 4 ) = M_PI*(3*pow(sin(alpha),5)-10*pow(sin(alpha),3)+15*sin(alpha))/4.8E+1 ;
	A( 8 , 5 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/4.8E+1 ;
	A( 8 , 6 ) = 0 ;
	A( 8 , 7 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/2.1E+2 ;
	A( 8 , 8 ) = -(9*sin(4*alpha)-4*pow(sin(2*alpha),3)+48*sin(2*alpha)+60*alpha-60*M_PI )/2.1E+2 ;

	A( 9 , 0 ) = 0 ;
	A( 9 , 1 ) = 0 ;
	A( 9 , 2 ) = 2.0*(M_PI-alpha)/5.0 ;
	A( 9 , 3 ) = M_PI*sin(alpha)/1.6E+1 ;
	A( 9 , 4 ) = 0 ;
	A( 9 , 5 ) = 0 ;
	A( 9 , 6 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.5E+1 ;
	A( 9 , 7 ) = 0 ;
	A( 9 , 8 ) = 0 ;
	A( 9 , 9 ) = 2.0*(M_PI-alpha)/7.0 ;

	if (size < 11)
		return A ;

	A( 10 , 0 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/1.5E+1 ;
	A( 10 , 1 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/4.8E+1 ;
	A( 10 , 2 ) = 0 ;
	A( 10 , 3 ) = 0 ;
	A( 10 , 4 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/2.1E+2 ;
	A( 10 , 5 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.5E+1 ;
	A( 10 , 6 ) = 0 ;
	A( 10 , 7 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/1.28E+2 ;
	A( 10 , 8 ) = M_PI*(3*pow(sin(alpha),5)-10*pow(sin(alpha),3)+15*sin(alpha))/3.84E+2 ;
	A( 10 , 9 ) = 0 ;
	A( 10 , 10 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/6.3E+2 ;

	A( 11 , 0 ) = 0 ;
	A( 11 , 1 ) = 0 ;
	A( 11 , 2 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/4.8E+1 ;
	A( 11 , 3 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/2.1E+2 ;
	A( 11 , 4 ) = 0 ;
	A( 11 , 5 ) = 0 ;
	A( 11 , 6 ) = M_PI*(3*pow(sin(alpha),5)-10*pow(sin(alpha),3)+15*sin(alpha))/3.84E+2 ;
	A( 11 , 7 ) = 0 ;
	A( 11 , 8 ) = 0 ;
	A( 11 , 9 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/1.28E+2 ;
	A( 11 , 10 ) = 0 ;
	A( 11 , 11 ) = -(9*sin(4*alpha)-4*pow(sin(2*alpha),3)+48*sin(2*alpha)+60*alpha-60*M_PI )/1.89E+3 ;

	A( 12 , 0 ) = 0 ;
	A( 12 , 1 ) = 0 ;
	A( 12 , 2 ) = M_PI*sin(alpha)/1.6E+1 ;
	A( 12 , 3 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.5E+1 ;
	A( 12 , 4 ) = 0 ;
	A( 12 , 5 ) = 0 ;
	A( 12 , 6 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/1.28E+2 ;
	A( 12 , 7 ) = 0 ;
	A( 12 , 8 ) = 0 ;
	A( 12 , 9 ) = 5.0*M_PI*sin(alpha)/1.28E+2 ;
	A( 12 , 10 ) = 0 ;
	A( 12 , 11 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/6.3E+2 ;
	A( 12 , 12 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/6.3E+1 ;

	A( 13 , 0 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/3.0E+1 ;
	A( 13 , 1 ) = M_PI*(3*pow(sin(alpha),5)-10*pow(sin(alpha),3)+15*sin(alpha))/4.8E+1 ;
	A( 13 , 2 ) = 0 ;
	A( 13 , 3 ) = 0 ;
	A( 13 , 4 ) = -(9*sin(4*alpha)-4*pow(sin(2*alpha),3)+48*sin(2*alpha)+60*alpha-60*M_PI )/2.1E+2 ;
	A( 13 , 5 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/2.1E+2 ;
	A( 13 , 6 ) = 0 ;
	A( 13 , 7 ) = M_PI*(3*pow(sin(alpha),5)-10*pow(sin(alpha),3)+15*sin(alpha))/3.84E+2 ;
	A( 13 , 8 ) = -M_PI*(5*pow(sin(alpha),7)-21*pow(sin(alpha),5)+35*pow(sin(alpha),3)-35*sin(alpha))/1.28E+2 ;
	A( 13 , 9 ) = 0 ;
	A( 13 , 10 ) = -(9*sin(4*alpha)-4*pow(sin(2*alpha),3)+48*sin(2*alpha)+60*alpha-60*M_PI )/1.89E+3 ;
	A( 13 , 11 ) = 0 ;
	A( 13 , 12 ) = 0 ;
	A( 13 , 13 ) = -(3*sin(8*alpha)+168*sin(4*alpha)-128*pow(sin(2*alpha),3)+768*sin(2*alpha)+840*alpha-840*M_PI)/3.78E+3 ;

	A( 14 , 0 ) = 2.0*(M_PI-alpha)/5.0 ;
	A( 14 , 1 ) = M_PI*sin(alpha)/1.6E+1 ;
	A( 14 , 2 ) = 0 ;
	A( 14 , 3 ) = 0 ;
	A( 14 , 4 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.5E+1 ;
	A( 14 , 5 ) = 2.0*(M_PI-alpha)/7.0 ;
	A( 14 , 6 ) = 0 ;
	A( 14 , 7 ) = 5.0*M_PI*sin(alpha)/1.28E+2 ;
	A( 14 , 8 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/1.28E+2 ;
	A( 14 , 9 ) = 0 ;
	A( 14 , 10 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/6.3E+1 ;
	A( 14 , 11 ) = 0 ;
	A( 14 , 12 ) = 0 ;
	A( 14 , 13 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/6.3E+2 ;
	A( 14 , 14 ) = 2.0*(M_PI-alpha)/9.0 ;

	return A ;
}

template <typename REAL>
Eigen::MatrixXd
QuadricHF<REAL>::buildIntegralMatrix_B(const REAL& alpha, unsigned int size)
{
	Eigen::MatrixXd B(size,size) ;

	 B( 0 , 0 ) = 2*(M_PI-alpha) ;
	 B( 0 , 1 ) = M_PI*(sin(2*alpha)+sin(alpha))/2.0 ;
	 B( 0 , 2 ) = 0 ;
	 B( 0 , 3 ) = 0 ;
	 B( 0 , 4 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.0 ;
	 B( 0 , 5 ) = 2.0*(M_PI-alpha)/3.0 ;

	 B( 1 , 0 ) = M_PI*sin(alpha)/2.0 ;
	 B( 1 , 1 ) = 4.0*(sin(alpha)/4.0-(sin(3*alpha)+(2*alpha-2*M_PI)*cos(alpha) )/4.0)/3.0 ;
	 B( 1 , 2 ) = 0 ;
	 B( 1 , 3 ) = 0 ;
	 B( 1 , 4 ) = 3.0*M_PI*((sin(5*alpha)+3*sin(3*alpha)+6*sin(alpha))/1.2E+1+sin( 2*alpha)/3.0)/8.0 ;
	 B( 1 , 5 ) = M_PI*sin(alpha)/8.0 ;

	 B( 2 , 0 ) = 0 ;
	 B( 2 , 1 ) = 0 ;
	 B( 2 , 2 ) = 2.0*(M_PI-alpha)/3.0 ;
	 B( 2 , 3 ) = M_PI*(sin(2*alpha)+sin(alpha))/8.0 ;
	 B( 2 , 4 ) = 0 ;
	 B( 2 , 5 ) = 0 ;

	 B( 3 , 0 ) = 0 ;
	 B( 3 , 1 ) = 0 ;
	 B( 3 , 2 ) = M_PI*sin(alpha)/8.0 ;
	 B( 3 , 3 ) = 4.0*(sin(alpha)/4.0-(sin(3*alpha)+(2*alpha-2*M_PI)*cos(alpha) )/4.0)/1.5E+1 ;
	 B( 3 , 4 ) = 0 ;
	 B( 3 , 5 ) = 0 ;

	 B( 4 , 0 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.0 ;
	 B( 4 , 1 ) = 3.0*M_PI*((sin(4*alpha)+6*sin(2*alpha))/1.2E+1+sin(alpha)/3.0 )/8.0 ;
	 B( 4 , 2 ) = 0 ;
	 B( 4 , 3 ) = 0 ;
	 B( 4 , 4 ) = 1.6E+1*(5.0*sin(2*alpha)/3.2E+1-(sin(6*alpha)+4*sin(4*alpha)+4* sin(2*alpha)+(4*alpha-4*M_PI)*cos(2*alpha)+8*alpha-8*M_PI)/3.2E+1 )/1.5E+1 ;
	 B( 4 , 5 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/1.5E+1 ;

	 B( 5 , 0 ) = 2.0*(M_PI-alpha)/3.0 ;
	 B( 5 , 1 ) = M_PI*(sin(2*alpha)+sin(alpha))/8.0 ;
	 B( 5 , 2 ) = 0 ;
	 B( 5 , 3 ) = 0 ;
	 B( 5 , 4 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/1.5E+1 ;
	 B( 5 , 5 ) = 2.0*(M_PI-alpha)/5.0 ;

	 if (size < 7)
		 return B ;

	 B( 0 , 6 ) = 0 ;
	 B( 0 , 7 ) = M_PI*(sin(2*alpha)+sin(alpha))/8.0 ;
	 B( 0 , 8 ) = 3.0*M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin(alpha))/3.0)/8.0 ;
	 B( 0 , 9 ) = 0 ;

	 B( 1 , 6 ) = 0 ;
	 B( 1 , 7 ) = 4.0*(sin(alpha)/4.0-(sin(3*alpha)+(2*alpha-2*M_PI)*cos(alpha) )/4.0)/1.5E+1 ;
	 B( 1 , 8 ) = 1.6E+1*((3*sin(3*alpha)+6*sin(alpha))/3.2E+1-(sin(7*alpha)+2*sin(5 *alpha)+6*sin(3*alpha)+(12*alpha-12*M_PI)*cos(alpha))/3.2E+1)/1.5E+1 ;
	 B( 1 , 9 ) = 0 ;

	 B( 2 , 6 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/1.5E+1 ;
	 B( 2 , 7 ) = 0 ;
	 B( 2 , 8 ) = 0 ;
	 B( 2 , 9 ) = 2.0*(M_PI-alpha)/5.0 ;

	 B( 3 , 6 ) = M_PI*((sin(5*alpha)+3*sin(3*alpha)+6*sin(alpha))/1.2E+1+sin(2*alpha )/3.0)/1.6E+1 ;
	 B( 3 , 7 ) = 0 ;
	 B( 3 , 8 ) = 0 ;
	 B( 3 , 9 ) = M_PI*sin(alpha)/1.6E+1 ;

	 B( 4 , 6 ) = 0 ;
	 B( 4 , 7 ) = M_PI*((sin(4*alpha)+6*sin(2*alpha))/1.2E+1+sin(alpha)/3.0)/1.6E+1 ;
	 B( 4 , 8 ) = 5.0*M_PI*((3*sin(8*alpha)+10*sin(6*alpha)+30*sin(4*alpha)+90*sin (2*alpha))/2.4E+2+(7*sin(3*alpha)+15*sin(alpha))/6.0E+1)/1.6E+1 ;
	 B( 4 , 9 ) = 0 ;

	 B( 5 , 6 ) = 0 ;
	 B( 5 , 7 ) = M_PI*(sin(2*alpha)+sin(alpha))/1.6E+1 ;
	 B( 5 , 8 ) = M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin (alpha))/3.0)/1.6E+1 ;
	 B( 5 , 9 ) = 0 ;

	 B( 6 , 0 ) = 0 ;
	 B( 6 , 1 ) = 0 ;
	 B( 6 , 2 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/1.5E+1 ;
	 B( 6 , 3 ) = M_PI*((sin(4*alpha)+6*sin(2*alpha))/1.2E+1+sin(alpha)/3.0)/1.6E+1 ;
	 B( 6 , 4 ) = 0 ;
	 B( 6 , 5 ) = 0 ;
	 B( 6 , 6 ) = 1.6E+1*(5.0*sin(2*alpha)/3.2E+1-(sin(6*alpha)+4*sin(4*alpha)+4* sin(2*alpha)+(4*alpha-4*M_PI)*cos(2*alpha)+8*alpha-8*M_PI)/3.2E+1 )/1.05E+2 ;
	 B( 6 , 7 ) = 0 ;
	 B( 6 , 8 ) = 0 ;
	 B( 6 , 9 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.5E+1 ;

	 B( 7 , 0 ) = M_PI*sin(alpha)/8.0 ;
	 B( 7 , 1 ) = 4.0*(sin(alpha)/4.0-(sin(3*alpha)+(2*alpha-2*M_PI)*cos(alpha) )/4.0)/1.5E+1 ;
	 B( 7 , 2 ) = 0 ;
	 B( 7 , 3 ) = 0 ;
	 B( 7 , 4 ) = M_PI*((sin(5*alpha)+3*sin(3*alpha)+6*sin(alpha))/1.2E+1+sin(2*alpha )/3.0)/1.6E+1 ;
	 B( 7 , 5 ) = M_PI*sin(alpha)/1.6E+1 ;
	 B( 7 , 6 ) = 0 ;
	 B( 7 , 7 ) = 4.0*(sin(alpha)/4.0-(sin(3*alpha)+(2*alpha-2*M_PI)*cos(alpha) )/4.0)/3.5E+1 ;
	 B( 7 , 8 ) = 1.6E+1*((3*sin(3*alpha)+6*sin(alpha))/3.2E+1-(sin(7*alpha)+2*sin(5 *alpha)+6*sin(3*alpha)+(12*alpha-12*M_PI)*cos(alpha))/3.2E+1)/1.05E+2 ;
	 B( 7 , 9 ) = 0 ;

	 B( 8 , 0 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/8.0 ;
	 B( 8 , 1 ) = 1.6E+1*(5.0*sin(alpha)/3.2E+1-(sin(5*alpha)+6*sin(3*alpha)+2*sin(alpha)+(12*alpha-12*M_PI)*cos(alpha))/3.2E+1)/1.5E+1 ;
	 B( 8 , 2 ) = 0 ;
	 B( 8 , 3 ) = 0 ;
	 B( 8 , 4 ) = 5.0*M_PI*((3*sin(7*alpha)+15*sin(5*alpha)+55*sin(3*alpha)+75*sin (alpha))/2.4E+2+sin(2*alpha)/5.0)/1.6E+1 ;
	 B( 8 , 5 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/4.8E+1 ;
	 B( 8 , 6 ) = 0 ;
	 B( 8 , 7 ) = 1.6E+1*(5.0*sin(alpha)/3.2E+1-(sin(5*alpha)+6*sin(3*alpha)+2*sin(alpha)+(12*alpha-12*M_PI)*cos(alpha))/3.2E+1)/1.05E+2 ;
	 B( 8 , 8 ) = 3.2E+1*((29*sin(3*alpha)+45*sin(alpha))/3.84E+2-(2*sin(9*alpha)+9*sin(7*alpha)+27*sin(5*alpha)+54*sin(3*alpha)+(12*alpha-12*M_PI)*cos(3*alpha)+18*sin(alpha)+(108*alpha-108*M_PI)*cos(alpha))/3.84E+2)/3.5E+1 ;
	 B( 8 , 9 ) = 0 ;

	 B( 9 , 0 ) = 0 ;
	 B( 9 , 1 ) = 0 ;
	 B( 9 , 2 ) = 2.0*(M_PI-alpha)/5.0 ;
	 B( 9 , 3 ) = M_PI*(sin(2*alpha)+sin(alpha))/1.6E+1 ;
	 B( 9 , 4 ) = 0 ;
	 B( 9 , 5 ) = 0 ;
	 B( 9 , 6 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.5E+1 ;
	 B( 9 , 7 ) = 0 ;
	 B( 9 , 8 ) = 0 ;
	 B( 9 , 9 ) = 2.0*(M_PI-alpha)/7.0 ;

	 if (size < 11)
		 return B ;

	 B( 0 , 10 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/1.5E+1 ;
	 B( 0 , 11 ) = 0 ;
	 B( 0 , 12 ) = 0 ;
	 B( 0 , 13 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/1.5E+1 ;
	 B( 0 , 14 ) = 2.0*(M_PI-alpha)/5.0 ;

	 B( 1 , 10 ) = M_PI*((sin(5*alpha)+3*sin(3*alpha)+6*sin(alpha))/1.2E+1+sin(2*alpha )/3.0)/1.6E+1 ;
	 B( 1 , 11 ) = 0 ;
	 B( 1 , 12 ) = 0 ;
	 B( 1 , 13 ) = 5.0*M_PI*((3*sin(9*alpha)+5*sin(7*alpha)+20*sin(5*alpha)+60*sin(3*alpha)+90*sin(alpha))/2.4E+2+(sin(4*alpha)+10*sin(2*alpha))/3.0E+1)/1.6E+1 ;
	 B( 1 , 14 ) = M_PI*sin(alpha)/1.6E+1 ;

	 B( 2 , 10 ) = 0 ;
	 B( 2 , 11 ) = M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin (alpha))/3.0)/1.6E+1 ;
	 B( 2 , 12 ) = M_PI*(sin(2*alpha)+sin(alpha))/1.6E+1 ;
	 B( 2 , 13 ) = 0 ;
	 B( 2 , 14 ) = 0 ;

	 B( 3 , 10 ) = 0 ;
	 B( 3 , 11 ) = 1.6E+1*((3*sin(3*alpha)+6*sin(alpha))/3.2E+1-(sin(7*alpha)+2*sin(5 *alpha)+6*sin(3*alpha)+(12*alpha-12*M_PI)*cos(alpha))/3.2E+1)/1.05E+2 ;
	 B( 3 , 12 ) = 4.0*(sin(alpha)/4.0-(sin(3*alpha)+(2*alpha-2*M_PI)*cos(alpha) )/4.0)/3.5E+1 ;
	 B( 3 , 13 ) = 0 ;
	 B( 3 , 14 ) = 0 ;

	 B( 4 , 10 ) = 1.6E+1*(5.0*sin(2*alpha)/3.2E+1-(sin(6*alpha)+4*sin(4*alpha)+4* sin(2*alpha)+(4*alpha-4*M_PI)*cos(2*alpha)+8*alpha-8*M_PI)/3.2E+1 )/1.05E+2 ;
	 B( 4 , 11 ) = 0 ;
	 B( 4 , 12 ) = 0 ;
	 B( 4 , 13 ) = 3.2E+1*((7*sin(4*alpha)+30*sin(2*alpha))/1.92E+2-(sin(10*alpha)+3* sin(8*alpha)+9*sin(6*alpha)+24*sin(4*alpha)+18*sin(2*alpha)+(24 *alpha-24*M_PI)*cos(2*alpha)+36*alpha-36*M_PI)/1.92E+2)/3.5E+1 ;
	 B( 4 , 14 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.5E+1 ;

	 B( 5 , 10 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.5E+1 ;
	 B( 5 , 11 ) = 0 ;
	 B( 5 , 12 ) = 0 ;
	 B( 5 , 13 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/1.05E+2 ;
	 B( 5 , 14 ) = 2.0*(M_PI-alpha)/7.0 ;

	 B( 6 , 10 ) = 0 ;
	 B( 6 , 11 ) = 5.0*M_PI*((3*sin(8*alpha)+10*sin(6*alpha)+30*sin(4*alpha)+90*sin (2*alpha))/2.4E+2+(7*sin(3*alpha)+15*sin(alpha))/6.0E+1)/1.28E+2 ;
	 B( 6 , 12 ) = 3.0*M_PI*((sin(4*alpha)+6*sin(2*alpha))/1.2E+1+sin(alpha)/3.0 )/1.28E+2 ;
	 B( 6 , 13 ) = 0 ;
	 B( 6 , 14 ) = 0 ;

	 B( 7 , 10 ) = 3.0*M_PI*((sin(5*alpha)+3*sin(3*alpha)+6*sin(alpha))/1.2E+1+sin( 2*alpha)/3.0)/1.28E+2 ;
	 B( 7 , 11 ) = 0 ;
	 B( 7 , 12 ) = 0 ;
	 B( 7 , 13 ) = 5.0*M_PI*((3*sin(9*alpha)+5*sin(7*alpha)+20*sin(5*alpha)+60*sin(3*alpha)+90*sin(alpha))/2.4E+2+(sin(4*alpha)+10*sin(2*alpha))/3.0E+1)/1.28E+2 ;
	 B( 7 , 14 ) = 5.0*M_PI*sin(alpha)/1.28E+2 ;

	 B( 8 , 10 ) = 5.0*M_PI*((3*sin(7*alpha)+15*sin(5*alpha)+55*sin(3*alpha)+75*sin (alpha))/2.4E+2+sin(2*alpha)/5.0)/1.28E+2 ;
	 B( 8 , 11 ) = 0 ;
	 B( 8 , 12 ) = 0 ;
	 B( 8 , 13 ) = 3.5E+1*M_PI*((5*sin(11*alpha)+21*sin(9*alpha)+63*sin(7*alpha)+175*sin(5*alpha)+490*sin(3*alpha)+490*sin(alpha))/2.24E+3+(3*sin(4*alpha)+14*sin(2*alpha))/7.0E+1)/1.28E+2 ;
	 B( 8 , 14 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/1.28E+2 ;

	 B( 9 , 10 ) = 0 ;
	 B( 9 , 11 ) = 3.0*M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin(alpha))/3.0)/1.28E+2 ;
	 B( 9 , 12 ) = 5.0*M_PI*(sin(2*alpha)+sin(alpha))/1.28E+2 ;
	 B( 9 , 13 ) = 0 ;
	 B( 9 , 14 ) = 0 ;

	 B( 10 , 0 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/1.5E+1 ;
	 B( 10 , 1 ) = M_PI*((sin(4*alpha)+6*sin(2*alpha))/1.2E+1+sin(alpha)/3.0)/1.6E+1 ;
	 B( 10 , 2 ) = 0 ;
	 B( 10 , 3 ) = 0 ;
	 B( 10 , 4 ) = 1.6E+1*(5.0*sin(2*alpha)/3.2E+1-(sin(6*alpha)+4*sin(4*alpha)+4* sin(2*alpha)+(4*alpha-4*M_PI)*cos(2*alpha)+8*alpha-8*M_PI)/3.2E+1 )/1.05E+2 ;
	 B( 10 , 5 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/3.5E+1 ;
	 B( 10 , 6 ) = 0 ;
	 B( 10 , 7 ) = 3.0*M_PI*((sin(4*alpha)+6*sin(2*alpha))/1.2E+1+sin(alpha)/3.0 )/1.28E+2 ;
	 B( 10 , 8 ) = 5.0*M_PI*((3*sin(8*alpha)+10*sin(6*alpha)+30*sin(4*alpha)+90*sin (2*alpha))/2.4E+2+(7*sin(3*alpha)+15*sin(alpha))/6.0E+1)/1.28E+2 ;
	 B( 10 , 9 ) = 0 ;
	 B( 10 , 10 ) = 1.6E+1*(5.0*sin(2*alpha)/3.2E+1-(sin(6*alpha)+4*sin(4*alpha)+4* sin(2*alpha)+(4*alpha-4*M_PI)*cos(2*alpha)+8*alpha-8*M_PI)/3.2E+1 )/3.15E+2 ;
	 B( 10 , 11 ) = 0 ;
	 B( 10 , 12 ) = 0 ;
	 B( 10 , 13 ) = 3.2E+1*((7*sin(4*alpha)+30*sin(2*alpha))/1.92E+2-(sin(10*alpha)+3* sin(8*alpha)+9*sin(6*alpha)+24*sin(4*alpha)+18*sin(2*alpha)+(24 *alpha-24*M_PI)*cos(2*alpha)+36*alpha-36*M_PI)/1.92E+2)/3.15E+2 ;
	 B( 10 , 14 ) = -(sin(2*alpha)+2*alpha-2*M_PI)/6.3E+1 ;

	 B( 11 , 0 ) = 0 ;
	 B( 11 , 1 ) = 0 ;
	 B( 11 , 2 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/4.8E+1 ;
	 B( 11 , 3 ) = 1.6E+1*(5.0*sin(alpha)/3.2E+1-(sin(5*alpha)+6*sin(3*alpha)+2*sin(alpha)+(12*alpha-12*M_PI)*cos(alpha))/3.2E+1)/1.05E+2 ;
	 B( 11 , 4 ) = 0 ;
	 B( 11 , 5 ) = 0 ;
	 B( 11 , 6 ) = 5.0*M_PI*((3*sin(7*alpha)+15*sin(5*alpha)+55*sin(3*alpha)+75*sin (alpha))/2.4E+2+sin(2*alpha)/5.0)/1.28E+2 ;
	 B( 11 , 7 ) = 0 ;
	 B( 11 , 8 ) = 0 ;
	 B( 11 , 9 ) = -M_PI*(pow(sin(alpha),3)-3*sin(alpha))/1.28E+2 ;
	 B( 11 , 10 ) = 0 ;
	 B( 11 , 11 ) = 3.2E+1*((29*sin(3*alpha)+45*sin(alpha))/3.84E+2-(2*sin(9*alpha)+9*sin(7*alpha)+27*sin(5*alpha)+54*sin(3*alpha)+(12*alpha-12*M_PI)*cos(3*alpha)+18*sin(alpha)+(108*alpha-108*M_PI)*cos(alpha))/3.84E+2)/3.15E+2 ;
	 B( 11 , 12 ) = 1.6E+1*(5.0*sin(alpha)/3.2E+1-(sin(5*alpha)+6*sin(3*alpha)+2*sin(alpha)+(12*alpha-12*M_PI)*cos(alpha))/3.2E+1)/3.15E+2 ;
	 B( 11 , 13 ) = 0 ;
	 B( 11 , 14 ) = 0 ;

	 B( 12 , 0 ) = 0 ;
	 B( 12 , 1 ) = 0 ;
	 B( 12 , 2 ) = M_PI*sin(alpha)/1.6E+1 ;
	 B( 12 , 3 ) = 4.0*(sin(alpha)/4.0-(sin(3*alpha)+(2*alpha-2*M_PI)*cos(alpha) )/4.0)/3.5E+1 ;
	 B( 12 , 4 ) = 0 ;
	 B( 12 , 5 ) = 0 ;
	 B( 12 , 6 ) = 3.0*M_PI*((sin(5*alpha)+3*sin(3*alpha)+6*sin(alpha))/1.2E+1+sin( 2*alpha)/3.0)/1.28E+2 ;
	 B( 12 , 7 ) = 0 ;
	 B( 12 , 8 ) = 0 ;
	 B( 12 , 9 ) = 5.0*M_PI*sin(alpha)/1.28E+2 ;
	 B( 12 , 10 ) = 0 ;
	 B( 12 , 11 ) = 1.6E+1*((3*sin(3*alpha)+6*sin(alpha))/3.2E+1-(sin(7*alpha)+2*sin(5 *alpha)+6*sin(3*alpha)+(12*alpha-12*M_PI)*cos(alpha))/3.2E+1)/3.15E+2 ;
	 B( 12 , 12 ) = 4.0*(sin(alpha)/4.0-(sin(3*alpha)+(2*alpha-2*M_PI)*cos(alpha) )/4.0)/6.3E+1 ;
	 B( 12 , 13 ) = 0 ;
	 B( 12 , 14 ) = 0 ;

	 B( 13 , 0 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/3.0E+1 ;
	 B( 13 , 1 ) = 5.0*M_PI*((3*sin(6*alpha)+20*sin(4*alpha)+95*sin(2*alpha))/2.4E+2+sin(alpha)/5.0)/1.6E+1 ;
	 B( 13 , 2 ) = 0 ;
	 B( 13 , 3 ) = 0 ;
	 B( 13 , 4 ) = 3.2E+1*(1.1E+1*sin(2*alpha)/9.6E+1-(sin(8*alpha)+6*sin(6*alpha)+21 *sin(4*alpha)+24*sin(2*alpha)+(24*alpha-24*M_PI)*cos(2*alpha)+36 *alpha-36*M_PI)/1.92E+2)/3.5E+1 ;
	 B( 13 , 5 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/2.1E+2 ;
	 B( 13 , 6 ) = 0 ;
	 B( 13 , 7 ) = 5.0*M_PI*((3*sin(6*alpha)+20*sin(4*alpha)+95*sin(2*alpha))/2.4E+2+sin(alpha)/5.0)/1.28E+2 ;
	 B( 13 , 8 ) = 3.5E+1*M_PI*((5*sin(10*alpha)+28*sin(8*alpha)+91*sin(6*alpha)+280*sin(4*alpha)+630*sin(2*alpha))/2.24E+3+(13*sin(3*alpha)+21*sin(alpha))/1.4E+2)/1.28E+2 ;
	 B( 13 , 9 ) = 0 ;
	 B( 13 , 10 ) = 3.2E+1*(1.1E+1*sin(2*alpha)/9.6E+1-(sin(8*alpha)+6*sin(6*alpha)+21 *sin(4*alpha)+24*sin(2*alpha)+(24*alpha-24*M_PI)*cos(2*alpha)+36 *alpha-36*M_PI)/1.92E+2)/3.15E+2 ;
	 B( 13 , 11 ) = 0 ;
	 B( 13 , 12 ) = 0 ;
	 B( 13 , 13 ) = 2.56E+2*((103*sin(4*alpha)+352*sin(2*alpha))/3.072E+3-(3*sin(12*alpha)+16*sin(10*alpha)+52*sin(8*alpha)+144*sin(6*alpha)+324*sin(4*alpha)+(24*alpha-24*M_PI)*cos(4*alpha)+288*sin(2*alpha)+(384*alpha-384*M_PI)*cos(2*alpha)+432*alpha-432*M_PI)/3.072E+3)/3.15E+2 ;
	 B( 13 , 14 ) = -(sin(4*alpha)+8*sin(2*alpha)+12*alpha-12*M_PI)/6.3E+2 ;

	 B( 14 , 0 ) = 2.0*(M_PI-alpha)/5.0 ;
	 B( 14 , 1 ) = M_PI*(sin(2*alpha)+sin(alpha))/1.6E+1 ;
	 B( 14 , 2 ) = 0 ;
	 B( 14 , 3 ) = 0 ;
	 B( 14 , 4 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.5E+1 ;
	 B( 14 , 5 ) = 2.0*(M_PI-alpha)/7.0 ;
	 B( 14 , 6 ) = 0 ;
	 B( 14 , 7 ) = 5.0*M_PI*(sin(2*alpha)+sin(alpha))/1.28E+2 ;
	 B( 14 , 8 ) = 3.0*M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin(alpha))/3.0)/1.28E+2 ;
	 B( 14 , 9 ) = 0 ;
	 B( 14 , 10 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/6.3E+1 ;
	 B( 14 , 11 ) = 0 ;
	 B( 14 , 12 ) = 0 ;
	 B( 14 , 13 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/3.15E+2 ;
	 B( 14 , 14 ) = 2.0*(M_PI-alpha)/9.0 ;

	return B ;
}

template <typename REAL>
Eigen::MatrixXd
QuadricHF<REAL>::buildLowerLeftIntegralMatrix_C(const REAL& alpha, unsigned int size)
{
	Eigen::MatrixXd C(size,size) ;

	C( 0 , 0 ) = 2*(M_PI-alpha) ;

	C( 1 , 0 ) = M_PI*(sin(2*alpha)+sin(alpha))/2.0 ;
	C( 1 , 1 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.0 ;

	C( 2 , 0 ) = 0 ;
	C( 2 , 1 ) = 0 ;
	C( 2 , 2 ) = 2.0*(M_PI-alpha)/3.0 ;

	C( 3 , 0 ) = 0 ;
	C( 3 , 1 ) = 0 ;
	C( 3 , 2 ) = M_PI*(sin(2*alpha)+sin(alpha))/8.0 ;
	C( 3 , 3 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/1.5E+1 ;

	C( 4 , 0 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.0 ;
	C( 4 , 1 ) = 3.0*M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin(alpha))/3.0)/8.0 ;
	C( 4 , 2 ) = 0 ;
	C( 4 , 3 ) = 0 ;
	C( 4 , 4 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/1.5E+1 ;

	C( 5 , 0 ) = 2.0*(M_PI-alpha)/3.0 ;
	C( 5 , 1 ) = M_PI*(sin(2*alpha)+sin(alpha))/8.0 ;
	C( 5 , 2 ) = 0 ;
	C( 5 , 3 ) = 0 ;
	C( 5 , 4 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/1.5E+1 ;
	C( 5 , 5 ) = 2.0*(M_PI-alpha)/5.0 ;

	if (size < 7)
		return C ;

	C( 6 , 0 ) = 0 ;
	C( 6 , 1 ) = 0 ;
	C( 6 , 2 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/1.5E+1 ;
	C( 6 , 3 ) = M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin (alpha))/3.0)/1.6E+1 ;
	C( 6 , 4 ) = 0 ;
	C( 6 , 5 ) = 0 ;
	C( 6 , 6 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/1.05E+2 ;

	C( 7 , 0 ) = M_PI*(sin(2*alpha)+sin(alpha))/8.0 ;
	C( 7 , 1 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/1.5E+1 ;
	C( 7 , 2 ) = 0 ;
	C( 7 , 3 ) = 0 ;
	C( 7 , 4 ) = M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin (alpha))/3.0)/1.6E+1 ;
	C( 7 , 5 ) = M_PI*(sin(2*alpha)+sin(alpha))/1.6E+1 ;
	C( 7 , 6 ) = 0 ;
	C( 7 , 7 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.5E+1 ;

	C( 8 , 0 ) = 3.0*M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin(alpha))/3.0)/8.0 ;
	C( 8 , 1 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/1.5E+1 ;
	C( 8 , 2 ) = 0 ;
	C( 8 , 3 ) = 0 ;
	C( 8 , 4 ) = 5.0*M_PI*((3*pow(sin(2*alpha),5)-10*pow(sin(2*alpha),3)+15*sin(2*alpha)) /1.5E+1+(3*pow(sin(alpha),5)-10*pow(sin(alpha),3)+15*sin(alpha))/1.5E+1 )/1.6E+1 ;
	C( 8 , 5 ) = M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin (alpha))/3.0)/1.6E+1 ;
	C( 8 , 6 ) = 0 ;
	C( 8 , 7 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/1.05E+2 ;
	C( 8 , 8 ) = 3.2E+1*((9*sin(4*alpha)-4*pow(sin(2*alpha),3)+48*sin(2*alpha)+60*alpha )/1.92E+2-(9*sin(8*alpha)-4*pow(sin(4*alpha),3)+48*sin(4*alpha)+120 *alpha-60*M_PI)/1.92E+2)/3.5E+1 ;

	C( 9 , 0 ) = 0 ;
	C( 9 , 1 ) = 0 ;
	C( 9 , 2 ) = 2.0*(M_PI-alpha)/5.0 ;
	C( 9 , 3 ) = M_PI*(sin(2*alpha)+sin(alpha))/1.6E+1 ;
	C( 9 , 4 ) = 0 ;
	C( 9 , 5 ) = 0 ;
	C( 9 , 6 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.5E+1 ;
	C( 9 , 7 ) = 0 ;
	C( 9 , 8 ) = 0 ;
	C( 9 , 9 ) = 2.0*(M_PI-alpha)/7.0 ;

	if (size < 11)
		return C ;

	C( 10 , 0 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/1.5E+1 ;
	C( 10 , 1 ) = M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin (alpha))/3.0)/1.6E+1 ;
	C( 10 , 2 ) = 0 ;
	C( 10 , 3 ) = 0 ;
	C( 10 , 4 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/1.05E+2 ;
	C( 10 , 5 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.5E+1 ;
	C( 10 , 6 ) = 0 ;
	C( 10 , 7 ) = 3.0*M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin(alpha))/3.0)/1.28E+2 ;
	C( 10 , 8 ) = 5.0*M_PI*((3*pow(sin(2*alpha),5)-10*pow(sin(2*alpha),3)+15*sin(2*alpha)) /1.5E+1+(3*pow(sin(alpha),5)-10*pow(sin(alpha),3)+15*sin(alpha))/1.5E+1)/1.28E+2 ;
	C( 10 , 9 ) = 0 ;
	C( 10 , 10 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/3.15E+2 ;

	C( 11 , 0 ) = 0 ;
	C( 11 , 1 ) = 0 ;
	C( 11 , 2 ) = M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin (alpha))/3.0)/1.6E+1 ;
	C( 11 , 3 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/1.05E+2 ;
	C( 11 , 4 ) = 0 ;
	C( 11 , 5 ) = 0 ;
	C( 11 , 6 ) = 5.0*M_PI*((3*pow(sin(2*alpha),5)-10*pow(sin(2*alpha),3)+15*sin(2*alpha)) /1.5E+1+(3*pow(sin(alpha),5)-10*pow(sin(alpha),3)+15*sin(alpha))/1.5E+1 )/1.28E+2 ;
	C( 11 , 7 ) = 0 ;
	C( 11 , 8 ) = 0 ;
	C( 11 , 9 ) = 3.0*M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin(alpha))/3.0)/1.28E+2 ;
	C( 11 , 10 ) = 0 ;
	C( 11 , 11 ) = 3.2E+1*((9*sin(4*alpha)-4*pow(sin(2*alpha),3)+48*sin(2*alpha)+60*alpha )/1.92E+2-(9*sin(8*alpha)-4*pow(sin(4*alpha),3)+48*sin(4*alpha)+120 *alpha-60*M_PI)/1.92E+2)/3.15E+2 ;

	C( 12 , 0 ) = 0 ;
	C( 12 , 1 ) = 0 ;
	C( 12 , 2 ) = M_PI*(sin(2*alpha)+sin(alpha))/1.6E+1 ;
	C( 12 , 3 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.5E+1 ;
	C( 12 , 4 ) = 0 ;
	C( 12 , 5 ) = 0 ;
	C( 12 , 6 ) = 3.0*M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin(alpha))/3.0)/1.28E+2 ;
	C( 12 , 7 ) = 0 ;
	C( 12 , 8 ) = 0 ;
	C( 12 , 9 ) = 5.0*M_PI*(sin(2*alpha)+sin(alpha))/1.28E+2 ;
	C( 12 , 10 ) = 0 ;
	C( 12 , 11 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/3.15E+2 ;
	C( 12 , 12 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/6.3E+1 ;

	C( 13 , 0 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/1.5E+1 ;
	C( 13 , 1 ) = 5.0*M_PI*((3*pow(sin(2*alpha),5)-10*pow(sin(2*alpha),3)+15*sin(2*alpha)) /1.5E+1+(3*pow(sin(alpha),5)-10*pow(sin(alpha),3)+15*sin(alpha))/1.5E+1 )/1.6E+1 ;
	C( 13 , 2 ) = 0 ;
	C( 13 , 3 ) = 0 ;
	C( 13 , 4 ) = 3.2E+1*((9*sin(4*alpha)-4*pow(sin(2*alpha),3)+48*sin(2*alpha)+60*alpha )/1.92E+2-(9*sin(8*alpha)-4*pow(sin(4*alpha),3)+48*sin(4*alpha)+120 *alpha-60*M_PI)/1.92E+2)/3.5E+1 ;
	C( 13 , 5 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/1.05E+2 ;
	C( 13 , 6 ) = 0 ;
	C( 13 , 7 ) = 5.0*M_PI*((3*pow(sin(2*alpha),5)-10*pow(sin(2*alpha),3)+15*sin(2*alpha)) /1.5E+1+(3*pow(sin(alpha),5)-10*pow(sin(alpha),3)+15*sin(alpha))/1.5E+1 )/1.28E+2 ;
	C( 13 , 8 ) = 3.5E+1*M_PI*(-(5*pow(sin(2*alpha),7)-21*pow(sin(2*alpha),5)+35*pow(sin(2*alpha),3)-35*sin(2*alpha))/3.5E+1-(5*pow(sin(alpha),7)-21*pow(sin(alpha),5)+35*pow(sin(alpha),3)-35*sin(alpha))/3.5E+1)/1.28E+2 ;
	C( 13 , 9 ) = 0 ;
	C( 13 , 10 ) = 3.2E+1*((9*sin(4*alpha)-4*pow(sin(2*alpha),3)+48*sin(2*alpha)+60*alpha )/1.92E+2-(9*sin(8*alpha)-4*pow(sin(4*alpha),3)+48*sin(4*alpha)+120 *alpha-60*M_PI)/1.92E+2)/3.15E+2 ;
	C( 13 , 11 ) = 0 ;
	C( 13 , 12 ) = 0 ;
	C( 13 , 13 ) = 2.56E+2*((3*sin(8*alpha)+168*sin(4*alpha)-128*pow(sin(2*alpha),3)+768* sin(2*alpha)+840*alpha)/3.072E+3-(3*sin(16*alpha)+168*sin(8*alpha)-128*pow(sin(4*alpha),3)+768*sin(4*alpha)+1680*alpha-840*M_PI)/3.072E+3)/3.15E+2 ;

	C( 14 , 0 ) = 2.0*(M_PI-alpha)/5.0 ;
	C( 14 , 1 ) = M_PI*(sin(2*alpha)+sin(alpha))/1.6E+1 ;
	C( 14 , 2 ) = 0 ;
	C( 14 , 3 ) = 0 ;
	C( 14 , 4 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/3.5E+1 ;
	C( 14 , 5 ) = 2.0*(M_PI-alpha)/7.0 ;
	C( 14 , 6 ) = 0 ;
	C( 14 , 7 ) = 5.0*M_PI*(sin(2*alpha)+sin(alpha))/1.28E+2 ;
	C( 14 , 8 ) = 3.0*M_PI*(-(pow(sin(2*alpha),3)-3*sin(2*alpha))/3.0-(pow(sin(alpha),3)-3*sin(alpha))/3.0)/1.28E+2 ;
	C( 14 , 9 ) = 0 ;
	C( 14 , 10 ) = 4.0*((sin(2*alpha)+2*alpha)/4.0-(sin(4*alpha)+4*alpha-2*M_PI) /4.0)/6.3E+1 ;
	C( 14 , 11 ) = 0 ;
	C( 14 , 12 ) = 0 ;
	C( 14 , 13 ) = 1.6E+1*((sin(4*alpha)+8*sin(2*alpha)+12*alpha)/3.2E+1-(sin(8*alpha )+8*sin(4*alpha)+24*alpha-12*M_PI)/3.2E+1)/3.15E+2 ;
	C( 14 , 14 ) = 2.0*(M_PI-alpha)/9.0 ;

	return C ;
}

template <typename REAL>
Geom::Tensor3d*
QuadricHF<REAL>::tensorsFromCoefs(const std::vector<VEC3>& coefs)
{
	const unsigned int& N = coefs.size() ;
	const unsigned int& degree = (sqrt(1+8*N) - 3) / REAL(2) ;
	Geom::Tensor3d *A = new Geom::Tensor3d[3] ;

	for (unsigned int col = 0 ; col < 3 ; ++col)
	{
		A[col] = Geom::Tensor3d(degree) ;
		//A[col].setConst(CONST_VAL) ;
	}

	std::vector<unsigned int> index ;
	if (N > 0)
	{
		index.resize(degree,2) ;
		for (unsigned int col = 0 ; col < 3 ; ++col)
			A[col](index) = coefs[0][col] ; // constant term (2,2,2,2)
		if (N > 2)
		{
			//				index.resize(degree,2) ;
			index[0] = 1 ;
			for (unsigned int col = 0 ; col < 3 ; ++col)
				A[col](index) = coefs[1][col] / degree ; // v (1,2,2,2)
			index[0] = 0 ;
			for (unsigned int col = 0 ; col < 3 ; ++col)
				A[col](index) = coefs[2][col] / degree ; // u (0,2,2,2)
			if (N > 5)
			{
				//					index.resize(degree,2) ;
				index[0] = 0 ; index[1] = 1 ;
				for (unsigned int col = 0 ; col < 3 ; ++col)
					A[col](index) = coefs[3][col] / (degree * (degree - 1)); // uv (0,1,2,2)

				index[0] = 1 ;
				for (unsigned int col = 0 ; col < 3 ; ++col)
					A[col](index) = coefs[4][col] / ((degree * (degree - 1)) / 2.) ; // v² (1,1,2,2)

				index[0] = 0 ; index[1] = 0 ;
				for (unsigned int col = 0 ; col < 3 ; ++col)
					A[col](index) = coefs[5][col] / ((degree * (degree - 1)) / 2.) ; // u² (0,0,2,2)
				if (N > 9)
				{
					//						index.resize(degree,2) ;
					index[0] = 0 ; index[1] = 1 ; index[2] = 1 ;
					for (unsigned int col = 0 ; col < 3 ; ++col)
						A[col](index) = coefs[6][col] / ((degree * (degree - 1) * (degree - 2)) / 2.) ; // uv**2 (0,1,1,2)
					index[0] = 0 ; index[1] = 0 ; index[2] = 1 ;
					for (unsigned int col = 0 ; col < 3 ; ++col)
						A[col](index) = coefs[7][col] / ((degree * (degree - 1) * (degree - 2)) / 2.) ; // u**2v (0,0,1,2)
					index[0] = 1 ; index[1] = 1 ; index[2] = 1 ;
					for (unsigned int col = 0 ; col < 3 ; ++col)
						A[col](index) = coefs[8][col] / ((degree * (degree - 1) * (degree - 2)) / 6.) ; // v**3 (1,1,1,2)
					index[0] = 0 ; index[1] = 0 ; index[2] = 0 ;
					for (unsigned int col = 0 ; col < 3 ; ++col)
						A[col](index) = coefs[9][col] / ((degree * (degree - 1) * (degree - 2)) / 6.) ; // u**3 (0,0,0,2)
					if (N > 14)
					{
						assert(degree == 4) ;
						//							index.resize(degree,2)
						index[0] = 0 ; index[1] = 0 ; index[2] = 1 ; index[3] = 1 ;
						for (unsigned int col = 0 ; col < 3 ; ++col)
							A[col](index) = coefs[10][col] / 6 ; // u**2v**2 (0,0,1,1)
						index[0] = 0 ; index[1] = 0 ; index[2] = 0 ; index[3] = 1 ;
						for (unsigned int col = 0 ; col < 3 ; ++col)
							A[col](index) = coefs[11][col] / 4 ; // uv**3 (0,0,0,1)
						index[0] = 0 ; index[1] = 1 ; index[2] = 1 ; index[3] = 1 ;
						for (unsigned int col = 0 ; col < 3 ; ++col)
							A[col](index) = coefs[12][col] / 4 ; // u**3v (0,1,1,1)
						index[0] = 1 ; index[1] = 1 ; index[2] = 1 ; index[3] = 1 ;
						for (unsigned int col = 0 ; col < 3 ; ++col)
							A[col](index) = coefs[13][col] ; // v**4 (1,1,1,1)
						index[0] = 0 ; index[1] = 0 ; index[2] = 0 ; index[3] = 0 ;
						for (unsigned int col = 0 ; col < 3 ; ++col)
							A[col](index) = coefs[14][col] ; // u**4 (0,0,0,0)
					}
				}
			}
		}
	}

	for (unsigned int col = 0 ; col < 3 ; ++col)
		A[col].completeSymmetricTensor() ;

	return A ;
}

template <typename REAL>
std::vector<typename QuadricHF<REAL>::VEC3>
QuadricHF<REAL>::coefsFromTensors(Geom::Tensor3d* A)
{
	const unsigned int& degree = A[0].order() ;
	std::vector<VEC3> coefs ;
	coefs.resize(((degree + 1) * (degree + 2)) / REAL(2)) ;

	std::vector<unsigned int> index ;
	index.resize(degree,2) ;
	for (unsigned int col = 0 ; col < 3 ; ++col)
		coefs[0][col] = A[col](index) ; // constant term (2,2,2,2)
	if (degree > 0)
	{
		index[0] = 1 ;
		for (unsigned int col = 0 ; col < 3 ; ++col)
			coefs[1][col] = A[col](index) * degree ; // v (1,2,2,2)
		index[0] = 0 ;
		for (unsigned int col = 0 ; col < 3 ; ++col)
			coefs[2][col] = A[col](index) * degree ; // u (0,2,2,2)
		if (degree > 1)
		{
			//					index.resize(degree,2) ;
			index[0] = 0 ; index[1] = 1 ;
			for (unsigned int col = 0 ; col < 3 ; ++col)
				coefs[3][col]  = A[col](index) * (degree * (degree - 1)); // uv (0,1,2,2)

			index[0] = 1 ;
			for (unsigned int col = 0 ; col < 3 ; ++col)
				coefs[4][col] = A[col](index) * ((degree * (degree - 1)) / 2.) ; // v² (1,1,2,2)

			index[0] = 0 ; index[1] = 0 ;
			for (unsigned int col = 0 ; col < 3 ; ++col)
				coefs[5][col] = A[col](index) * ((degree * (degree - 1)) / 2.) ; // u² (0,0,2,2)
			if (degree > 2)
			{
				//						index.resize(degree,2) ;
				index[0] = 0 ; index[1] = 1 ; index[2] = 1 ;
				for (unsigned int col = 0 ; col < 3 ; ++col)
					coefs[6][col] = A[col](index) * ((degree * (degree - 1) * (degree - 2)) / 2.) ; // uv**2 (0,1,1,2)
				index[0] = 0 ; index[1] = 0 ; index[2] = 1 ;
				for (unsigned int col = 0 ; col < 3 ; ++col)
					coefs[7][col] = A[col](index) * ((degree * (degree - 1) * (degree - 2)) / 2.) ; // u**2v (0,0,1,2)
				index[0] = 1 ; index[1] = 1 ; index[2] = 1 ;
				for (unsigned int col = 0 ; col < 3 ; ++col)
					coefs[8][col] = A[col](index) * ((degree * (degree - 1) * (degree - 2)) / 6.) ; // v**3 (1,1,1,2)
				index[0] = 0 ; index[1] = 0 ; index[2] = 0 ;
				for (unsigned int col = 0 ; col < 3 ; ++col)
					coefs[9][col] = A[col](index) * ((degree * (degree - 1) * (degree - 2)) / 6.) ; // u**3 (0,0,0,2)
				if (degree > 3)
				{
					assert(degree == 4) ;
					//							index.resize(degree,2)
					index[0] = 0 ; index[1] = 0 ; index[2] = 1 ; index[3] = 1 ;
					for (unsigned int col = 0 ; col < 3 ; ++col)
						coefs[10][col] = A[col](index) * 6 ; // u**2v**2 (0,0,1,1)
					index[0] = 0 ; index[1] = 0 ; index[2] = 0 ; index[3] = 1 ;
					for (unsigned int col = 0 ; col < 3 ; ++col)
						coefs[11][col] = A[col](index) * 4 ; // uv**3 (0,0,0,1)
					index[0] = 0 ; index[1] = 1 ; index[2] = 1 ; index[3] = 1 ;
					for (unsigned int col = 0 ; col < 3 ; ++col)
						coefs[12][col] = A[col](index) * 4 ; // u**3v (0,1,1,1)
					index[0] = 1 ; index[1] = 1 ; index[2] = 1 ; index[3] = 1 ;
					for (unsigned int col = 0 ; col < 3 ; ++col)
						coefs[13][col] = A[col](index) ; // v**4 (1,1,1,1)
					index[0] = 0 ; index[1] = 0 ; index[2] = 0 ; index[3] = 0 ;
					for (unsigned int col = 0 ; col < 3 ; ++col)
						coefs[14][col] = A[col](index) ; // u**4 (0,0,0,0)
				}
			}
		}
	}

	return coefs ;
}

} // Utils

} // CGOGN

