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

namespace CGoGN
{

namespace Geom
{

template <unsigned int M, unsigned int N, typename T>
std::string Matrix<M,N,T>::CGoGNnameOfType()
{
	std::stringstream ss ;
	ss << "Geom::Matrix<" ;
	ss << M ;
	ss << "," ;
	ss << N ;
	ss << "," ;
	ss << nameOfType(T()) ;
	ss << ">" ;

	return ss.str() ;
}

/**********************************************/
/*                CONSTRUCTORS                */
/**********************************************/

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T>::Matrix()
{
	CGoGN_STATIC_ASSERT(M > 0, invalid_zero_dimensional_Matrix) ;
	CGoGN_STATIC_ASSERT(N > 0, invalid_zero_dimensional_Matrix) ;
	zero() ;
}

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T>::Matrix(const Matrix<M,N,T>& m)
{
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			m_data[i][j] = m(i,j) ;
}

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T>::Matrix(T v)
{
	set(v) ;
}

template <unsigned int M, unsigned int N, typename T>
void Matrix<M,N,T>::set(T a)
{
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			m_data[i][j] = a ;
}

template <unsigned int M, unsigned int N, typename T>
void Matrix<M,N,T>::zero()
{
	set(T(0)) ;
}

template <unsigned int M, unsigned int N, typename T>
void Matrix<M,N,T>::identity()
{
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			m_data[i][j] = (i==j) ? T(1) : T(0) ;
}

template <unsigned int M, unsigned int N, typename T>
template <unsigned int K, unsigned int L>
bool Matrix<M,N,T>::setSubMatrix(unsigned int ii, unsigned int jj, const Matrix<K,L,T>& m)
{
	if(ii + K <= M && jj + L <= N)
	{
		for(unsigned int i = 0; i < K; ++i)
			for(unsigned int j = 0; j < L; ++j)
				m_data[ii+i][jj+j] = m(i,j) ;
		return true ;
	}
	return false ;
}

template <unsigned int M, unsigned int N, typename T>
template <unsigned int K>
bool Matrix<M,N,T>::setSubVectorV(unsigned int ii, unsigned int jj, const typename Vector<K,T>::type& v)
{
	if(ii + K <= M && jj <= N)
	{
		for(unsigned int i = 0; i < K; ++i)
			m_data[ii+i][jj] = v[i] ;
		return true ;
	}
	return false ;
}

template <unsigned int M, unsigned int N, typename T>
template <unsigned int L>
bool Matrix<M,N,T>::setSubVectorH(unsigned int ii, unsigned int jj, const typename Vector<L,T>::type& v)
{
	if(ii <= M && jj + L <= N)
	{
		for(unsigned int j = 0; j < L; ++j)
			m_data[ii][jj+j] = v[j] ;
		return true ;
	}
	return false ;
}

/**********************************************/
/*                 ACCESSORS                  */
/**********************************************/

template <unsigned int M, unsigned int N, typename T>
T& Matrix<M,N,T>::operator()(unsigned int i, unsigned int j)
{
	assert(i < M) ;
	assert(j < N) ;
	return m_data[i][j] ;
}

template <unsigned int M, unsigned int N, typename T>
const T& Matrix<M,N,T>::operator()(unsigned int i, unsigned int j) const
{
	assert(i < M) ;
	assert(j < N) ;
	return m_data[i][j] ;
}

template <unsigned int M, unsigned int N, typename T>
template <unsigned int K, unsigned int L>
bool Matrix<M,N,T>::getSubMatrix(unsigned int ii, unsigned int jj, Matrix<K,L,T>& m) const
{
	if(ii + K <= M && jj + L <= N)
	{
		for(unsigned int i = 0; i < K; ++i)
			for(unsigned int j = 0; j < L; ++j)
				m(i,j) = m_data[ii+i][jj+j] ;
		return true ;
	}
	return false ;
}

template <unsigned int M, unsigned int N, typename T>
template <unsigned int K>
bool Matrix<M,N,T>::getSubVectorV(unsigned int ii, unsigned int jj, typename Vector<K,T>::type& v) const
{
	if(ii + K <= M && jj <= N)
	{
		for(unsigned int i = 0; i < K; ++i)
			v[i] = m_data[ii+i][jj] ;
		return true ;
	}
	return false ;
}

template <unsigned int M, unsigned int N, typename T>
template <unsigned int L>
bool Matrix<M,N,T>::getSubVectorH(unsigned int ii, unsigned int jj, typename Vector<L,T>::type& v) const
{
	if(ii <= M && jj + L <= N)
	{
		for(unsigned int j = 0; j < L; ++j)
			v[j] = m_data[ii][jj+j] ;
		return true ;
	}
	return false ;
}

template <unsigned int M, unsigned int N, typename T>
unsigned int Matrix<M,N,T>::m() const
{
	return M ;
}

template <unsigned int M, unsigned int N, typename T>
unsigned int Matrix<M,N,T>::n() const
{
	return N ;
}

/**********************************************/
/*         ARITHMETIC SELF-OPERATORS          */
/**********************************************/

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T>& Matrix<M,N,T>::operator+=(const Matrix<M,N,T>& m)
{
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			m_data[i][j] += m(i,j) ;
	return *this ;
}

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T>& Matrix<M,N,T>::operator-=(const Matrix<M,N,T>& m)
{
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			m_data[i][j] -= m(i,j) ;
	return *this ;
}

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T>& Matrix<M,N,T>::operator*=(T a)
{
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			m_data[i][j] *= a ;
	return *this ;
}

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T>& Matrix<M,N,T>::operator/=(T a)
{
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			m_data[i][j] /= a ;
	return *this ;
}

/**********************************************/
/*            ARITHMETIC OPERATORS            */
/**********************************************/

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T> Matrix<M,N,T>::operator+(const Matrix<M,N,T>& m) const
{
	Matrix<M,N,T> res ;
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			res(i,j) = m_data[i][j] + m(i,j) ;
	return res ;
}

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T> Matrix<M,N,T>::operator-(const Matrix<M,N,T>& m) const
{
	Matrix<M,N,T> res ;
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			res(i,j) = m_data[i][j] - m(i,j) ;
	return res ;
}

template <unsigned int M, unsigned int N, typename T>
template <unsigned int P>
Matrix<M,P,T> Matrix<M,N,T>::operator*(const Matrix<N,P,T>& m) const
{
	Matrix<M,P,T> res ;
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < P; ++j)
			for(unsigned int k = 0; k < N; ++k)
				res(i,j) += m_data[i][k] * m(k,j) ;
	return res ;
}

template <unsigned int M, unsigned int N, typename T>
typename Vector<M,T>::type Matrix<M,N,T>::operator*(const typename Vector<N,T>::type& v) const
{
        typename Vector<N,T>::type res ;
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			res[i] += m_data[i][j] * v[j] ;
	return res ;
}

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T> Matrix<M,N,T>::operator*(T s) const
{
	Matrix<M,N,T> res ;
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			res(i,j) = m_data[i][j] * s ;
	return res ;
}

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T> Matrix<M,N,T>::operator/(T s) const
{
	Matrix<M,N,T> res ;
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			res(i,j) = m_data[i][j] / s ;
	return res ;
}

/**********************************************/
/*             UTILITY FUNCTIONS              */
/**********************************************/

template <unsigned int M, unsigned int N, typename T>
void Matrix<M,N,T>::transpose()
{
	CGoGN_STATIC_ASSERT(M == N, Matrix_self_transpose_only_available_for_square_matrices) ;
	Matrix<M,M,T> m ;
	for(unsigned int i = 1; i < M; ++i)
		for(unsigned int j = 0; j < i; ++j)
		{
			T tmp = m_data[i][j] ;
			m_data[i][j] = m_data[j][i] ;
			m_data[j][i] = tmp ;
		}
}

template <unsigned int M, unsigned int N, typename T>
Matrix<N,M,T> Matrix<M,N,T>::transposed() const
{
	Matrix<N,M,T> m ;
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			m(j,i) = m_data[i][j] ;
	return m ;
}

template <unsigned int M, unsigned int N, typename T>
T Matrix<M,N,T>::invert(Matrix<M,M,T>& result) const
{
	CGoGN_STATIC_ASSERT(M == N, Matrix_invert_only_available_for_square_matrices) ;
	Matrix<M,M,T> mat = (*this) ;
	result.identity() ;
	T det = T(1) ;

	for (unsigned int i = 0; i < M; ++i)
	{
		T pivot = mat(i,i) ;
		unsigned int ind = i ;
		for (unsigned int j = i + 1; j < M; ++j)
		{
			if (fabs(mat(j,i)) > fabs(pivot))
			{
				ind = j ;
				pivot = mat(j,i) ;
			}
		}

		det *= pivot ;
		if(det == 0.0)
			return det ;

		if (ind != i)
		{
			for (unsigned int j = 0; j < M; ++j)
			{
				T tmp = result(i,j) ;
				result(i,j) = result(ind,j) ;
				result(ind,j) = tmp ;

				tmp = mat(i,j) ;
				mat(i,j) = mat(ind,j) ;
				mat(ind,j) = tmp ;

				det = -det ;
			}
		}

		for (unsigned int j = 0; j < M; ++j)
		{
			mat(i,j)	/= pivot ;
			result(i,j) /= pivot ;
		}

		for (unsigned int j = 0; j < M; ++j)
		{
			if (j == i)
				continue ;
			T t = mat(j,i);
			for (unsigned int k = 0; k < M; ++k)
			{
				mat(j,k)    -= mat(i,k)    * t ;
				result(j,k) -= result(i,k) * t ;
			}
		}
	}

	return det ;
}

template <unsigned int M, unsigned int N, typename T>
bool Matrix<M,N,T>::operator==(const Matrix<M,N,T>& m) const {
	for (unsigned int i = 0 ; i < M ; ++i)
		for (unsigned int j = 0 ; j < N ; ++j)
			if (m(i,j) != m_data[i][j]) return false ;
	return true ;
}

/**********************************************/
/*             STREAM OPERATORS               */
/**********************************************/

template <unsigned int M, unsigned int N, typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<M,N,T>& m)
{
	for(unsigned int i = 0; i < M; ++i)
	{
		for(unsigned int j = 0; j < N; ++j)
			out << m(i,j) << " " ;
		out << std::endl ;
	}
	return out ;
}

template <unsigned int M, unsigned int N, typename T>
std::istream& operator>>(std::istream& in, Matrix<M,N,T>& m)
{
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			in >> m(i,j) ;
	return in ;
}

/**********************************************/
/*           EXTERNAL OPERATORS               */
/**********************************************/

// Vector / Matrix multiplication
template <unsigned int M, unsigned int N, typename T>
typename Vector<N,T>::type operator*(const typename Vector<M,T>::type& v, const Matrix<M,N,T>& m)
{
        typename Vector<N,T>::type res;
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			res[j] += v[i] * m(i,j) ;
	return res ;
}

// Matrix / Vector multiplication
template <unsigned int M, unsigned int N, typename T>
typename Vector<M,T>::type operator*(const Matrix<M,N,T>& m,const Vector<N,T>& v) {
        typename Vector<M,T>::type res (0);
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			res[i] += m(i,j) * v[j] ;
	return res ;
}



// Scalar / Matrix multiplication
template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T> operator*(T s, const Matrix<M,N,T>& m)
{
	Matrix<M,N,T> res ;
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			res(i,j) = m(i,j) * s ;
	return res ;
}

template <unsigned int M, unsigned int N, typename T>
Matrix<M,N,T> transposed_vectors_mult(const typename Vector<M,T>::type& v1, const Vector<N,T>& v2) {
	Matrix<M,N,T> res ;
	for(unsigned int i = 0; i < M; ++i)
		for(unsigned int j = 0; j < N; ++j)
			res(i,j) = v1[i] * v2[j] ;
	return res ;
}

} // namespace Geom

} // namespace CGoGN
