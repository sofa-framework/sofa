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

/*! \file qem.h
* Header file for Quadric Error Metric classes.
*/

#ifndef __QEM__
#define __QEM__

#include "Utils/os_spec.h" // allow compilation under windows
#include <cmath>

#include "Geometry/vector_gen.h"
#include "Geometry/matrix.h"
#include "Geometry/tensor.h"
#include "Geometry/plane_3d.h"

// Eigen includes
#include <Eigen/Dense>

/*! \namespace CGoGN
 * \brief namespace for all elements composing the CGoGN library
 */
namespace CGoGN {

/*! \namespace Utils
 * \brief namespace for tool classes used by CGoGN and its applications
 */
namespace Utils {

/*! \class Quadric
 *
 * \brief Quadric for computing the quadric error metric (QEM)
 * introduced by Garland and Heckbert in 1997.
 */
template <typename REAL>
class Quadric
{
public:
	/**
	 * \brief get CGoGN name of current type
	 *
	 * \return name of the CGoGN type
	 */
	static std::string CGoGNnameOfType()
	{
		return "Quadric" ;
	}

    typedef typename Geom::Vector<3,REAL>::type VEC3 ;
    typedef typename Geom::Vector<4,REAL>::type VEC4 ;
	typedef Geom::Matrix<4,4,double> MATRIX44 ; // double is crucial here !

	/*!
	 * \brief Default constructor
	 *
	 * Initializes empty members.
	 */
	Quadric() ;

	/*!
	 * \brief Constructor
	 *
	 * Initializes empty members (idem. default constructor).
	 * Exists for compatibility reasons.
	 */
	Quadric(int i) ;

	/*!
	 * \brief Constructor building a quadric given three points (defining a plane);.
	 *
	 * \param p1 first point
	 * \param p2 second point
	 * \param p3 third point
	 */
	Quadric(const VEC3& p1, const VEC3& p2, const VEC3& p3) ;


	/*!
	 * \brief destructor
	 */
	~Quadric() {} ;

	/*!
	 * \brief set members to zero
	 */
	void zero() ;

	/*!
	 * \brief affectation operator (by copy)
	 *
	 * \param q the Quadric to copy
	 */
	void operator= (const Quadric<REAL>& q) ;

	/*!
	 * \brief sum of Quadric operator
	 *
	 * \param q the Quadric to sum
	 */
	Quadric& operator+= (const Quadric<REAL>& q) ;

	/*!
	 * \brief substract of Quadric operator
	 *
	 * \param q the Quadric to substract
	 */
	Quadric& operator -= (const Quadric<REAL>& q) ;

	/*!
	 * \brief scalar product operator
	 *
	 * \param v the scalar to multiply the Quadric with
	 */
	Quadric& operator *= (const REAL& v) ;

	/*!
	 * \brief scalar division operator
	 *
	 * \param v the scalar to divide the Quadric with
	 */
	Quadric& operator /= (const REAL& v) ;

	/*!
	 * \brief error evaluation operator
	 *
	 * \param v a point expressed in homogeneous coordinates in space
	 *
	 * \return the error
	 */
	REAL operator() (const VEC4& v) const ;

	/*!
	 * \brief error evaluation operator
	 *
	 * \param v a point in space
	 *
	 * \param the error
	 */
	REAL operator() (const VEC3& v) const ;

	/*!
	 * \brief Write to stream operator
	 *
	 * \param out the stream to write to
	 * \param q the Quadric to write in the stream
	 *
	 * \return the stream reference
	 */
	friend std::ostream& operator<<(std::ostream& out, const Quadric<REAL>& q)
	{
		out << q.A ;
		return out ;
	} ;

	/*!
	 * \brief Read from stream operator
	 *
	 * \param int the stream to read from
	 * \param q the Quadric to write the data that has been read in
	 *
	 * \return the stream reference
	 */
	friend std::istream& operator>>(std::istream& in, Quadric<REAL>& q)
	{
		in >> q.A ;
		return in ;
	} ;

	/*!
	 * \brief Method to deduce a position in space that minimizes the error
	 *
	 * \param v the ideal position (if it can be computed)
	 *
	 * \return true if the ideal position has been computed correctly
	 */
	bool findOptimizedPos(VEC3& v) ;

private:
	MATRIX44 A ; /*!< The Quadric matrix */

	/*!
	 * \brief method to evaluate the error at a given point in space (homogeneous coordinates)
	 *
	 * \param v the given point
	 *
	 * \return the error
	 */
	REAL evaluate(const VEC4& v) const ;

	/*!
	 * \brief method to deduce an optimal position in space (homogeneous coordinates)
	 * w.r.t. the current Quadric.
	 *
	 * \param v will contain the optimal position (if it can be computed)
	 *
	 * \return true if an optimal position was correctly computed
	 */
	bool optimize(VEC4& v) const ;
} ;

/*! \class QuadricNd
 * \brief extension of Quadric (which is 3D) to nD points.
 * This was published by Garland and Heckbert in 1998 and is meant to define a quadric
 * for a nD-space which contains geometry (3D) + other attributes like color (3D),
 * normals (3D) or texture coordinates (2D) for instance.
 */
template <typename REAL, unsigned int N>
class QuadricNd
{
public:
	/**
	 * \brief get CGoGN name of current type
	 *
	 * \return name of the CGoGN type
	 */
	static std::string CGoGNnameOfType()
	{
		return "QuadricNd" ;
	}

    typedef typename Geom::Vector<N,REAL>::type VECN ;
    typedef typename Geom::Vector<N+1,REAL>::type VECNp ;

	/*!
	 * \brief Default constructor
	 *
	 * Initializes empty members.
	 */
	QuadricNd() ;

	/*!
	 * \brief Constructor
	 *
	 * Initializes empty members (idem. default constructor).
	 * Exists for compatibility reasons.
	 */
	QuadricNd(int i) ;

	/*!
	 * \brief Constructor building a quadricNd given three points (defining a plane);
	 *
	 * \param p1 first point
	 * \param p2 second point
	 * \param p3 third point
	 */
	QuadricNd(const VECN& p1_r, const VECN& p2_r, const VECN& p3_r) ;

	/*!
	 * \brief destructor
	 */
	~QuadricNd() {} ;

	/*!
	 * \brief set members to zero
	 */
	void zero() ;

	/*!
	 * \brief affectation operator (by copy)
	 *
	 * \param q the QuadricNd to copy
	 */
	void operator= (const QuadricNd<REAL,N>& q) ;

	/*!
	 * \brief sum of QuadricNd operator
	 *
	 * \param q the QuadricNd to sum
	 */
	QuadricNd& operator+= (const QuadricNd<REAL,N>& q) ;

	/*!
	 * \brief substract of QuadricNd operator
	 *
	 * \param q the QuadricNd to substract
	 */
	QuadricNd& operator -= (const QuadricNd<REAL,N>& q) ;

	/*!
	 * \brief scalar product operator
	 *
	 * \param v the scalar to multiply the QuadricNd with
	 */
	QuadricNd& operator *= (REAL v) ;

	/*!
	 * \brief scalar division operator
	 *
	 * \param v the scalar to divide the QuadricNd with
	 */
	QuadricNd& operator /= (REAL v) ;

	/*!
	 * \brief error evaluation operator
	 *
	 * \param v a point expressed in homogeneous coordinates in nD space
	 *
	 * \return the error
	 */
	REAL operator() (const VECNp& v) const ;

	/*!
	 * \brief error evaluation operator
	 *
	 * \param v a point in nD space
	 *
	 * \param the error
	 */
	REAL operator() (const VECN& v) const ;

	/*!
	 * \brief Write to stream operator
	 *
	 * \param out the stream to write to
	 * \param q the QuadricNd to write in the stream
	 *
	 * \return the stream reference
	 */
	friend std::ostream& operator<<(std::ostream& out, const QuadricNd<REAL,N>& q)
	{
		out << "(" << q.A << ", " << q.b << ", " << q.c << ")" ;
		return out ;
	} ;

	/*!
	 * \brief Read from stream operator
	 *
	 * \param int the stream to read from
	 * \param q the QuadricNd to write the data that has been read in
	 *
	 * \return the stream reference
	 */
	friend std::istream& operator>>(std::istream& in, QuadricNd<REAL,N>& q)
	{
		in >> q.A ;
		in >> q.b ;
		in >> q.c ;
		return in ;
	} ;

	/*!
	 * \brief Method to deduce a position in space that minimizes the error
	 *
	 * \param v the ideal position (if it can be computed)
	 *
	 * \return true if the ideal position has been computed correctly
	 */
	bool findOptimizedVec(VECN& v) ;

private:
	// Double computation is crucial for stability
	Geom::Matrix<N,N,double> A ; /*!< The first QuadricNd member matrix A */
    typename Geom::Vector<N,double>::type b ; /*!< The second QuadricNd member vector b */
	double c ;/*!< The third QuadricNd member scalar c */

	/*!
	 * \brief method to evaluate the error at a given nD point in space (homogeneous coordinates)
	 *
	 * \param v the given point
	 *
	 * \return the error
	 */
	REAL evaluate(const VECN& v) const ;

	/*!
	 * \brief method to deduce an optimal position in space (homogeneous coordinates)
	 * w.r.t. the current QuadricNd.
	 *
	 * \param v will contain the optimal position (if it can be computed)
	 *
	 * \return true if an optimal position was correctly computed
	 */
	bool optimize(VECN& v) const ;
} ;

/*! \class QuadricHF
 * \brief quadric used for measuring a lightfield metric.
 * This was defined by Vanhoey, Sauvage and Dischler in 2012.
 * This implementation works only for polynomial basis functions.
 */
template <typename REAL>
class QuadricHF
{
public:
	/**
	 * \brief get CGoGN name of current type
	 *
	 * \return name of the CGoGN type
	 */
	static std::string CGoGNnameOfType() { return "QuadricHF" ; }

    typedef typename Geom::Vector<3,REAL>::type VEC3 ;

	/*!
	 * \brief Constructor
	 *
	 * Initializes empty members
	 */
	QuadricHF() ;

	/*!
	 * \brief Constructor
	 *
	 * Initializes empty members (idem. default constructor)
	 * Exists for compatibility reasons
	 */
	QuadricHF(int i) ;

	/*!
	 * \brief Constructor building a QuadricHF given a lightfield function and the two angles gamma and alpha
	 *
	 * \param v the lightfield function
	 * \param gamma
	 * \param alpha
	 */
	QuadricHF(const std::vector<VEC3>& v, const REAL& gamma, const REAL& alpha) ;

	/*!
	 * \brief Constructor building a QuadricHF given a lightfield function and the two angles gamma and alpha
	 *
	 * \param v the lightfield function expressed as a Geom::Tensor
	 * \param gamma
	 * \param alpha
	 */
	QuadricHF(const Geom::Tensor3d* T, const REAL& gamma, const REAL& alpha) ;

	/*!
	 * Destructor
	 */
	~QuadricHF() ;

	/*!
	 * \brief set members to zero
	 */
	void zero() ;

	/*!
	 * \brief affectation operator (by copy)
	 *
	 * \param q the QuadricHF to copy
	 */
	QuadricHF& operator= (const QuadricHF<REAL>& q) ;

	/*!
	 * \brief sum of QuadricHF operator
	 *
	 * \param q the QuadricHF to sum
	 */
	QuadricHF& operator+= (const QuadricHF<REAL>& q) ;

	/*!
	 * \brief substract of QuadricHF operator
	 *
	 * \param q the QuadricHF to substract
	 */
	QuadricHF& operator -= (const QuadricHF<REAL>& q) ;

	/*!
	 * \brief scalar product operator
	 *
	 * \param v the scalar to multiply the QuadricHF with
	 */
	QuadricHF& operator *= (const REAL& v) ;

	/*!
	 * \brief scalar division operator
	 *
	 * \param v the scalar to divide the QuadricHF with
	 */
	QuadricHF& operator /= (const REAL& v) ;

	/*!
	 * \brief error evaluation operator
	 *
	 * \param coefs a lightfield function
	 *
	 * \param the error
	 */
	REAL operator() (const std::vector<VEC3>& coefs) const ;


	/*!
	 * \brief method to evaluate the error for a given lightfield function
	 *
	 * \param coefs the given function
	 *
	 * \return the squared error per color channel
	 */
	VEC3 evalR3(const std::vector<VEC3>& coefs) const ;

	/*!
	 * \brief Write to stream operator
	 *
	 * \param out the stream to write to
	 * \param q the QuadricHF to write in the stream
	 *
	 * \return the stream reference
	 */
	friend std::ostream& operator<<(std::ostream& out, const QuadricHF<REAL>& q)
	{
        out << "(" << q.m_A << ", " << q.m_b << ", " << q.m_c << ")" ;
		return out ;
    }

	/*!
	 * \brief Read from stream operator
	 *
	 * \param int the stream to read from
	 * \param q the QuadricHF to write the data that has been read in
	 *
	 * \return the stream reference
	 */
	friend std::istream& operator>>(std::istream& in, QuadricHF<REAL>& q)
	{
		// TODO
		//		in >> q.A ;
		//		in >> q.b ;
		//		in >> q.c ;
		return in ;
	} ;

	/*!
	 * \brief Method to deduce a position in space that minimizes the error
	 *
	 * \param v the ideal position (if it can be computed)
	 *
	 * \return true if the ideal position has been computed correctly
	 */
	bool findOptimizedCoefs(std::vector<VEC3>& coefs) ;

	/*!
	 * \brief method to convert a lightfield in tensor format to a coefficient vector format
	 *
	 * \param coefs vector of coefficients representing a lightfield function
	 *
	 * \return a tensor representing the same lightfield function
	 *
	 */
	static Geom::Tensor3d* tensorsFromCoefs(const std::vector<VEC3>& coefs) ;

	/*!
	 * \brief method to convert a lightfield in coefficient vector format to a tensor format
	 *
	 * \param T the tensor to convert
	 *
	 * \return a vector of coefficients representing the same lightfield function
	 */
	static std::vector<VEC3> coefsFromTensors(Geom::Tensor3d* T) ;

	/*!
	 * \brief method to complete a symmetric matrix that was
	 * only filled in its first half (line >= column)
	 *
	 * \param M the matrix to fill
	 */
	static void fillSymmetricMatrix(Eigen::MatrixXd& M) ;

	/*!
	 * \brief method to rotate a tensor representing a polynomial light field
	 *
	 * \param T the tensor representing a polynomial light field
	 * \param R the 3x3 matrix representing a rotation in (u,v,1)-space
	 *
	 * \return a new rotated tensor representing a polynomial light field.
	 */
	static Geom::Tensor3d rotate(const Geom::Tensor3d& T, const Geom::Matrix33d& R) ;

private:
	// Double computation is crucial for stability
	Eigen::MatrixXd m_A ; /*!< The first QuadricHF member matrix A */
	Eigen::VectorXd m_b[3] ; /*!< The second QuadricHF member vector b */
	double m_c[3] ; /*!< The third QuadricHF member scalar c */
	std::vector<VEC3> m_coefs ; /*!< The coefficients in cas optim fails */
	bool m_noAlphaRot ; /*!< If alpha = 0 then optim will fail */

	/*!
	 * \brief method to evaluate the error for a given lightfield function
	 *
	 * \param coefs the given function
	 *
	 * \return the norm of the squared error per color channel
	 */
	REAL evaluate(const std::vector<VEC3>& coefs) const ;

	/*!
	 * \brief method to build a rotate matrix (rotation in tangent plane)
	 * given angle gamma
	 *
	 * \param gamma the rotation angle
	 *
	 * \return the rotation matrix
	 */
	Geom::Matrix33d buildRotateMatrix(const REAL& gamma) ;

	/*!
	 * \brief method to build the first integral matrix A
	 * given angle alpha
	 *
	 * \param alpha angle
	 * \param size the amount of monomes in a function
	 *
	 * \return the integral of product of monomes
	 */
	Eigen::MatrixXd buildIntegralMatrix_A(const REAL& alpha, unsigned int size) ;

	/*!
	 * \brief method to build the first integral matrix B
	 * given angle alpha
	 *
	 * \param alpha angle
	 * * \param size the amount of monomes in a function
	 *
	 * \return the integral of product of monomes
	 */
	Eigen::MatrixXd buildIntegralMatrix_B(const REAL& alpha, unsigned int size) ;

	/*!
	 * \brief method to build the third integral matrix C
	 * given angle alpha
	 *
	 * \param alpha angle
	 * \param size the amount of monomes in a function
	 *
	 * \return the integral of product of monomes
	 */
	Eigen::MatrixXd buildIntegralMatrix_C(const REAL& alpha, unsigned int size) ;

	Eigen::MatrixXd buildLowerLeftIntegralMatrix_A(const REAL& alpha, unsigned int size) ;
	Eigen::MatrixXd buildLowerLeftIntegralMatrix_C(const REAL& alpha, unsigned int size) ;
} ;

} // Utils

} // CGOGN

#include "Utils/qem.hpp"

#endif
