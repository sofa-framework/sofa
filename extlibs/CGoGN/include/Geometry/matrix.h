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

#ifndef __MATRIX__
#define __MATRIX__

#include "Geometry/vector_gen.h"

namespace CGoGN
{

namespace Geom
{

/*
 * Class for the representation of rectangular matrices
 */
template <unsigned int M, unsigned int N, typename T>
class Matrix
{
	public:
		typedef T DATA_TYPE ;

		static std::string CGoGNnameOfType() ;

		/**********************************************/
		/*                CONSTRUCTORS                */
		/**********************************************/

		Matrix();

		Matrix(const Matrix<M,N,T>& m);

		Matrix(T v);

		void set(T a);

		void zero();

		void identity();

		template <unsigned int K, unsigned int L>
		bool setSubMatrix(unsigned int ii, unsigned int jj, const Matrix<K,L,T>& m);

		template <unsigned int K>
                bool setSubVectorV(unsigned int ii, unsigned int jj, const typename Vector<K,T>::type& v) ;

		template <unsigned int L>
                bool setSubVectorH(unsigned int ii, unsigned int jj, const typename Vector<L,T>::type& v) ;

		/**********************************************/
		/*                 ACCESSORS                  */
		/**********************************************/

		T& operator()(unsigned int i, unsigned int j);

		const T& operator()(unsigned int i, unsigned int j) const;

		template <unsigned int K, unsigned int L>
		bool getSubMatrix(unsigned int ii, unsigned int jj, Matrix<K,L,T>& m) const;

		template <unsigned int K>
                bool getSubVectorV(unsigned int ii, unsigned int jj, typename Vector<K,T>::type& v) const;

		template <unsigned int L>
                bool getSubVectorH(unsigned int ii, unsigned int jj, typename Vector<L,T>::type& v) const;

		unsigned int m() const;

		unsigned int n() const;

		/**********************************************/
		/*         ARITHMETIC SELF-OPERATORS          */
		/**********************************************/

		Matrix<M,N,T>& operator+=(const Matrix<M,N,T>& m);

		Matrix<M,N,T>& operator-=(const Matrix<M,N,T>& m);

		Matrix<M,N,T>& operator*=(T a);

		Matrix<M,N,T>& operator/=(T a);

		/**********************************************/
		/*            ARITHMETIC OPERATORS            */
		/**********************************************/

		Matrix<M,N,T> operator+(const Matrix<M,N,T>& m) const;

		Matrix<M,N,T> operator-(const Matrix<M,N,T>& m) const;

		// Matrix / Matrix multiplication
		template <unsigned int P>
		Matrix<M,P,T> operator*(const Matrix<N,P,T>& m) const;

		// Matrix / Vector multiplication
                typename Vector<M,T>::type operator*(const typename Vector<N,T>::type& v) const;

		// Matrix / Scalar multiplication
		Matrix<M,N,T> operator*(T s) const;

		// Matrix / Scalar division
		Matrix<M,N,T> operator/(T s) const;

		/**********************************************/
		/*             UTILITY FUNCTIONS              */
		/**********************************************/

		// transpose the matrix
		// ** Works only for square matrices **
		void transpose();

		// return a new matrix which is the transpose of the matrix
		Matrix<N,M,T> transposed() const;

		// Invert the matrix using Gauss-Jordan elimination
		// The determinant of the matrix is returned
		// (in case of singular matrix (determinant=0),
		// trash values are leaved in the result)
		// ** Works only for square matrices **
		T invert(Matrix<M,M,T>& result) const;

		// Equal
		bool operator==(const Matrix<M,N,T>& m) const ;

		/**********************************************/
		/*             STREAM OPERATORS               */
		/**********************************************/

		template <unsigned int MM, unsigned int NN, typename TT>
		friend std::ostream& operator<<(std::ostream& out, const Matrix<MM,NN,TT>& m);

		template <unsigned int MM, unsigned int NN, typename TT>
		friend std::istream& operator>>(std::istream& in, Matrix<MM,NN,TT>& m);

	private:
		T m_data[M][N] ;
} ;

/**********************************************/
/*           EXTERNAL OPERATORS               */
/**********************************************/

// Vector / Matrix multiplication
template <unsigned int M, unsigned int N, typename T>
typename Vector<N,T>::type operator*(const typename Vector<M,T>::type& v, const Matrix<M,N,T>& m) ;

// Matrix / Vector multiplication
template <unsigned int M, unsigned int N, typename T>
typename Vector<M,T>::type operator*(const Matrix<M,N,T>& m,const typename Vector<N,T>::type & v) ;

// Scalar / Matrix multiplication
template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T> operator*(T s, const Matrix<M,N,T>& m) ;

// Vector / Transposed vector multiplication
template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T> transposed_vectors_mult(const typename Vector<M,T>::type& v1, const typename Vector<N,T>::type& v2) ;


/**********************************************/
/*           SOME USEFUL TYPEDEFS             */
/**********************************************/

typedef Matrix<2,2,float> Matrix22f ;
typedef Matrix<2,2,double> Matrix22d ;
typedef Matrix<2,2,unsigned int> Matrix22ui ;
typedef Matrix<2,2,int> Matrix22i ;
typedef Matrix<2,2,unsigned char> Matrix22uc ;

typedef Matrix<3,3,float> Matrix33f ;
typedef Matrix<3,3,double> Matrix33d ;
typedef Matrix<3,3,unsigned int> Matrix33ui ;
typedef Matrix<3,3,int> Matrix33i ;
typedef Matrix<3,3,unsigned char> Matrix33uc ;

typedef Matrix<4,4,float> Matrix44f ;
typedef Matrix<4,4,double> Matrix44d ;
typedef Matrix<4,4,unsigned int> Matrix44ui ;
typedef Matrix<4,4,int> Matrix44i ;
typedef Matrix<4,4,unsigned char> Matrix44uc ;

}

}

#include "matrix.hpp"

#endif
