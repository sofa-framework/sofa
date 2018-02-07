/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_MAT_H
#define SOFA_DEFAULTTYPE_MAT_H

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/Vec.h>
#include <iostream>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace defaulttype
{

template <int L, int C, class real=float>
class Mat : public helper::fixed_array<VecNoInit<C,real>,L>
    //class Mat : public Vec<L,Vec<C,real> >
{
public:

    enum { N = L*C };

    // Cheap and hacky workaround: Mat inherits from fixed_array - which uses
    // std:size_t - but it defines its own operator[] with int, and this causes
    // warnings when calling mat[(size_type)foo].
    typedef int size_type;

    typedef real Real;
    typedef Vec<C,real> Line;
    typedef VecNoInit<C,real> LineNoInit;
    typedef Vec<L,real> Col;

    static const int nbLines = L;
    static const int nbCols  = C;

    Mat()
    {
        clear();
    }

    explicit Mat(NoInit)
    {
    }

    /*
      /// Specific constructor with a single line.
      explicit Mat(Line r1)
      {
        static_assert(L == 1, "");
        this->elems[0]=r1;
      }
    */

    /// Specific constructor with 2 lines.
    Mat(Line r1, Line r2)
    {
        static_assert(L == 2, "");
        this->elems[0]=r1;
        this->elems[1]=r2;
    }

    /// Specific constructor with 3 lines.
    Mat(Line r1, Line r2, Line r3)
    {
        static_assert(L == 3, "");
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
    }

    /// Specific constructor with 4 lines.
    Mat(Line r1, Line r2, Line r3, Line r4)
    {
        static_assert(L == 4, "");
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
    }

    /// Constructor from an element
    explicit Mat(const real& v)
    {
        for( int i=0; i<L; i++ )
            for( int j=0; j<C; j++ )
                this->elems[i][j] = v;
    }

    /// Constructor from another matrix
    template<typename real2>
    Mat(const Mat<L,C,real2>& m)
    {
        std::copy(m.begin(), m.begin()+L, this->begin());
    }

    /// Constructor from another matrix with different size (with null default entries and ignoring outside entries)
    template<int L2, int C2, typename real2>
    explicit Mat(const Mat<L2,C2,real2>& m)
    {
        int maxL = std::min( L, L2 );
        int maxC = std::min( C, C2 );

        for( int l=0 ; l<maxL ; ++l )
        {
            for( int c=0 ; c<maxC ; ++c )
                this->elems[l][c] = (real)m[l][c];
            for( int c=maxC ; c<C ; ++c )
                this->elems[l][c] = 0;
        }

        for( int l=maxL ; l<L ; ++l )
            for( int c=0 ; c<C ; ++c )
                this->elems[l][c] = 0;
    }

    /// Constructor from an array of elements (stored per line).
    template<typename real2>
    explicit Mat(const real2* p)
    {
        std::copy(p, p+N, this->begin()->begin());
    }

    /// number of lines
    int getNbLines() const
    {
        return L;
    }

    /// number of colums
    int getNbCols() const
    {
        return C;
    }


    /// Assignment from an array of elements (stored per line).
    void operator=(const real* p)
    {
        std::copy(p, p+N, this->begin()->begin());
    }

    /// Assignment from another matrix
    template<typename real2> void operator=(const Mat<L,C,real2>& m)
    {
        std::copy(m.begin(), m.begin()+L, this->begin());
    }

    /// Assignment from a matrix of different size.
    template<int L2, int C2> void operator=(const Mat<L2,C2,real>& m)
    {
        std::copy(m.begin(), m.begin()+(L>L2?L2:L), this->begin());
    }

    template<int L2, int C2> void getsub(int L0, int C0, Mat<L2,C2,real>& m) const
    {
        for (int i=0; i<L2; i++)
            for (int j=0; j<C2; j++)
                m[i][j] = this->elems[i+L0][j+C0];
    }

    template<int L2, int C2> void setsub(int L0, int C0, const Mat<L2,C2,real>& m)
    {
        for (int i=0; i<L2; i++)
            for (int j=0; j<C2; j++)
                this->elems[i+L0][j+C0] = m[i][j];
    }

    template<int L2> void setsub(int L0, int C0, const Vec<L2,real>& v)
    {
        assert( C0<C );
        assert( L0+L2-1<L );
        for (int i=0; i<L2; i++)
            this->elems[i+L0][C0] = v[i];
    }


    /// Sets each element to 0.
    void clear()
    {
        for (int i=0; i<L; i++)
            this->elems[i].clear();
    }

    /// Sets each element to r.
    void fill(real r)
    {
        for (int i=0; i<L; i++)
            this->elems[i].fill(r);
    }

    /// Read-only access to line i.
    const Line& line(int i) const
    {
        return this->elems[i];
    }

    /// Copy of column j.
    Col col(int j) const
    {
        Col c;
        for (int i=0; i<L; i++)
            c[i]=this->elems[i][j];
        return c;
    }

    /// Write acess to line i.
    LineNoInit& operator[](int i)
    {
        return this->elems[i];
    }

    /// Read-only access to line i.
    const LineNoInit& operator[](int i) const
    {
        return this->elems[i];
    }

    /// Write acess to line i.
    LineNoInit& operator()(int i)
    {
        return this->elems[i];
    }

    /// Read-only access to line i.
    const LineNoInit& operator()(int i) const
    {
        return this->elems[i];
    }

    /// Write access to element (i,j).
    real& operator()(int i, int j)
    {
        return this->elems[i][j];
    }

    /// Read-only access to element (i,j).
    const real& operator()(int i, int j) const
    {
        return this->elems[i][j];
    }

    /// Cast into a standard C array of lines (read-only).
    const Line* lptr() const
    {
        return this->elems;
    }

    /// Cast into a standard C array of lines.
    Line* lptr()
    {
        return this->elems;
    }

    /// Cast into a standard C array of elements (stored per line) (read-only).
    const real* ptr() const
    {
        return this->elems[0].ptr();;
    }

    /// Cast into a standard C array of elements (stored per line).
    real* ptr()
    {
        return this->elems[0].ptr();
    }

    /// Special access to first line.
    Line& x() { static_assert(L >= 1, ""); return this->elems[0]; }
    /// Special access to second line.
    Line& y() { static_assert(L >= 2, ""); return this->elems[1]; }
    /// Special access to third line.
    Line& z() { static_assert(L >= 3, ""); return this->elems[2]; }
    /// Special access to fourth line.
    Line& w() { static_assert(L >= 4, ""); return this->elems[3]; }

    /// Special access to first line (read-only).
    const Line& x() const { static_assert(L >= 1, ""); return this->elems[0]; }
    /// Special access to second line (read-only).
    const Line& y() const { static_assert(L >= 2, ""); return this->elems[1]; }
    /// Special access to thrid line (read-only).
    const Line& z() const { static_assert(L >= 3, ""); return this->elems[2]; }
    /// Special access to fourth line (read-only).
    const Line& w() const { static_assert(L >= 4, ""); return this->elems[3]; }

    /// Set matrix to identity.
    void identity()
    {
        static_assert(L == C, "");
        clear();
        for (int i=0; i<L; i++)
            this->elems[i][i]=1;
    }

    /// precomputed identity matrix of size (L,L)
    static Mat<L,L,real> s_identity;

    /// Returns the identity matrix
    static Mat<L,L,real> Identity()
    {
        static_assert(L == C, "");
        Mat<L,L,real> id;
        for (int i=0; i<L; i++)
            id[i][i]=1;
        return id;
    }


    template<int S>
    static bool canSelfTranspose(const Mat<S, S, real>& lhs, const Mat<S, S, real>& rhs)
    {
        return &lhs == &rhs;
    }

    template<int I, int J>
    static bool canSelfTranspose(const Mat<I, J, real>& /*lhs*/, const Mat<J, I, real>& /*rhs*/)
    {
        return false;
    }

    /// Set matrix as the transpose of m.
    void transpose(const Mat<C,L,real> &m)
    {
        if (canSelfTranspose(*this, m))
        {
            for (int i=0; i<L; i++)
            {
                for (int j=i+1; j<C; j++)
                {
                    std::swap(this->elems[i][j], this->elems[j][i]);
                }
            }
        }
        else
        {
            for (int i=0; i<L; i++)
                for (int j=0; j<C; j++)
                    this->elems[i][j]=m[j][i];
        }
    }

    /// Return the transpose of m.
    Mat<C,L,real> transposed() const
    {
        Mat<C,L,real> m(NOINIT);
        for (int i=0; i<L; i++)
            for (int j=0; j<C; j++)
                m[j][i]=this->elems[i][j];
        return m;
    }

    /// Transpose the square matrix.
    void transpose()
    {
        static_assert(L == C, "Cannot self-transpose a non-square matrix. Use transposed() instead");
        for (int i=0; i<L; i++)
        {
            for (int j=i+1; j<C; j++)
            {
                std::swap(this->elems[i][j], this->elems[j][i]);
            }
        }
    }

    /// @name Tests operators
    /// @{

    bool operator==(const Mat<L,C,real>& b) const
    {
        for (int i=0; i<L; i++)
            if (!(this->elems[i]==b[i])) return false;
        return true;
    }

    bool operator!=(const Mat<L,C,real>& b) const
    {
        for (int i=0; i<L; i++)
            if (this->elems[i]!=b[i]) return true;
        return false;
    }


    bool isSymmetric() const
    {
        for (int i=0; i<L; i++)
            for (int j=i+1; j<C; j++)
                if( fabs( this->elems[i][j] - this->elems[j][i] ) > EQUALITY_THRESHOLD ) return false;
        return true;
    }

    bool isDiagonal() const
    {
        for (int i=0; i<L; i++)
        {
            for (int j=0; j<i-1; j++)
                if( helper::rabs( this->elems[i][j] ) > EQUALITY_THRESHOLD ) return false;
            for (int j=i+1; j<C; j++)
                if( helper::rabs( this->elems[i][j] ) > EQUALITY_THRESHOLD ) return false;
        }
        return true;
    }


    /// @}

    // LINEAR ALGEBRA

    /// Matrix multiplication operator.
    template <int P>
    Mat<L,P,real> operator*(const Mat<C,P,real>& m) const
    {
        Mat<L,P,real> r(NOINIT);
        for(int i=0; i<L; i++)
            for(int j=0; j<P; j++)
            {
                r[i][j]=(*this)[i][0] * m[0][j];
                for(int k=1; k<C; k++)
                    r[i][j] += (*this)[i][k] * m[k][j];
            }
        return r;
    }

    /// Matrix addition operator.
    Mat<L,C,real> operator+(const Mat<L,C,real>& m) const
    {
        Mat<L,C,real> r(NOINIT);
        for(int i = 0; i < L; i++)
            r[i] = (*this)[i] + m[i];
        return r;
    }

    /// Matrix subtraction operator.
    Mat<L,C,real> operator-(const Mat<L,C,real>& m) const
    {
        Mat<L,C,real> r(NOINIT);
        for(int i = 0; i < L; i++)
            r[i] = (*this)[i] - m[i];
        return r;
    }

    /// Matrix negation operator.
    Mat<L,C,real> operator-() const
    {
        Mat<L,C,real> r(NOINIT);
        for(int i = 0; i < L; i++)
            r[i] = -(*this)[i];
        return r;
    }

    /// Multiplication operator Matrix * Line.
    Col operator*(const Line& v) const
    {
        Col r(NOINIT);
        for(int i=0; i<L; i++)
        {
            r[i]=(*this)[i][0] * v[0];
            for(int j=1; j<C; j++)
                r[i] += (*this)[i][j] * v[j];
        }
        return r;
    }


    /// Multiplication with a diagonal Matrix CxC represented as a vector of size C
    Mat<L,C,real> multDiagonal(const Line& d) const
    {
        Mat<L,C,real> r(NOINIT);
        for(int i=0; i<L; i++)
            for(int j=0; j<C; j++)
                r[i][j]=(*this)[i][j] * d[j];
        return r;
    }

    /// Multiplication of the transposed Matrix * Column
    Line multTranspose(const Col& v) const
    {
        Line r(NOINIT);
        for(int i=0; i<C; i++)
        {
            r[i]=(*this)[0][i] * v[0];
            for(int j=1; j<L; j++)
                r[i] += (*this)[j][i] * v[j];
        }
        return r;
    }


    /// Transposed Matrix multiplication operator.
    template <int P>
    Mat<C,P,real> multTranspose(const Mat<L,P,real>& m) const
    {
        Mat<C,P,real> r(NOINIT);
        for(int i=0; i<C; i++)
            for(int j=0; j<P; j++)
            {
                r[i][j]=(*this)[0][i] * m[0][j];
                for(int k=1; k<L; k++)
                    r[i][j] += (*this)[k][i] * m[k][j];
            }
        return r;
    }

    /// Multiplication with the transposed of the given matrix operator \returns this * mt
    template <int P>
    Mat<L,P,real> multTransposed(const Mat<P,C,real>& m) const
    {
        Mat<L,P,real> r(NOINIT);
        for(int i=0; i<L; i++)
            for(int j=0; j<P; j++)
            {
                r[i][j]=(*this)[i][0] * m[j][0];
                for(int k=1; k<C; k++)
                    r[i][j] += (*this)[i][k] * m[j][k];
            }
        return r;
    }

    /// Addition with the transposed of the given matrix operator \returns this + mt
    Mat<L,C,real> plusTransposed(const Mat<C,L,real>& m) const
    {
        Mat<L,C,real> r(NOINIT);
        for(int i=0; i<L; i++)
            for(int j=0; j<C; j++)
                r[i][j] = (*this)[i][j] + m[j][i];
        return r;
    }

    /// Substraction with the transposed of the given matrix operator \returns this - mt
    Mat<L,C,real>minusTransposed(const Mat<C,L,real>& m) const
    {
        Mat<L,C,real> r(NOINIT);
        for(int i=0; i<L; i++)
            for(int j=0; j<C; j++)
                r[i][j] = (*this)[i][j] - m[j][i];
        return r;
    }


    /// Scalar multiplication operator.
    Mat<L,C,real> operator*(real f) const
    {
        Mat<L,C,real> r(NOINIT);
        for(int i=0; i<L; i++)
            for(int j=0; j<C; j++)
                r[i][j] = (*this)[i][j] * f;
        return r;
    }

    /// Scalar matrix multiplication operator.
    friend Mat<L,C,real> operator*(real r, const Mat<L,C,real>& m)
    {
        return m*r;
    }

    /// Scalar division operator.
    Mat<L,C,real> operator/(real f) const
    {
        Mat<L,C,real> r(NOINIT);
        for(int i=0; i<L; i++)
            for(int j=0; j<C; j++)
                r[i][j] = (*this)[i][j] / f;
        return r;
    }

    /// Scalar multiplication assignment operator.
    void operator *=(real r)
    {
        for(int i=0; i<L; i++)
            this->elems[i]*=r;
    }

    /// Scalar division assignment operator.
    void operator /=(real r)
    {
        for(int i=0; i<L; i++)
            this->elems[i]/=r;
    }

    /// Addition assignment operator.
    void operator +=(const Mat<L,C,real>& m)
    {
        for(int i=0; i<L; i++)
            this->elems[i]+=m[i];
    }

    /// Addition of the transposed of m
    void addTransposed(const Mat<C,L,real>& m)
    {
        for(int i=0; i<L; i++)
            for(int j=0; j<C; j++)
                (*this)[i][j] += m[j][i];
    }

    /// Substraction of the transposed of m
    void subTransposed(const Mat<C,L,real>& m)
    {
        for(int i=0; i<L; i++)
            for(int j=0; j<C; j++)
                (*this)[i][j] -= m[j][i];
    }

    /// Substraction assignment operator.
    void operator -=(const Mat<L,C,real>& m)
    {
        for(int i=0; i<L; i++)
            this->elems[i]-=m[i];
    }


    /// invert this
    Mat<L,C,real> inverted() const
    {
        static_assert(L == C, "Cannot invert a non-square matrix");
        Mat<L,C,real> m = *this;
        invertMatrix(m, *this);
        return m;
    }

    /// Invert square matrix m
    bool invert(const Mat<L,C,real>& m)
    {
        static_assert(L == C, "Cannot invert a non-square matrix");
        if (&m == this)
        {
            Mat<L,C,real> mat = m;
            bool res = invertMatrix(*this, mat);
            return res;
        }
        return invertMatrix(*this, m);
    }

    static Mat<L,C,real> transformTranslation(const Vec<C-1,real>& t)
    {
        Mat<L,C,real> m;
        m.identity();
        for (int i=0; i<C-1; ++i)
            m.elems[i][C-1] = t[i];
        return m;
    }

    static Mat<L,C,real> transformScale(real s)
    {
        Mat<L,C,real> m;
        m.identity();
        for (int i=0; i<C-1; ++i)
            m.elems[i][i] = s;
        return m;
    }

    static Mat<L,C,real> transformScale(const Vec<C-1,real>& s)
    {
        Mat<L,C,real> m;
        m.identity();
        for (int i=0; i<C-1; ++i)
            m.elems[i][i] = s[i];
        return m;
    }

    template<class Quat>
    static Mat<L,C,real> transformRotation(const Quat& q)
    {
        Mat<L,C,real> m;
        m.identity();
        q.toMatrix(m);
        return m;
    }

    /// @return True if and only if the Matrix is a transformation matrix
    bool isTransform() const
    {
        for (int j=0;j<C-1;++j)
            if (fabs((*this)(L-1,j)) > EQUALITY_THRESHOLD)
                return false;
        if (fabs((*this)(L-1,C-1) - 1.) > EQUALITY_THRESHOLD)
            return false;
        return true;
    }

    /// Multiplication operator Matrix * Vector considering the matrix as a transformation.
    Vec<C-1,real> transform(const Vec<C-1,real>& v) const
    {
        Vec<C-1,real> r(NOINIT);
        for(int i=0; i<C-1; i++)
        {
            r[i]=(*this)[i][0] * v[0];
            for(int j=1; j<C-1; j++)
                r[i] += (*this)[i][j] * v[j];
            r[i] += (*this)[i][C-1];
        }
        return r;
    }

    /// Invert transformation matrix m
    bool transformInvert(const Mat<L,C,real>& m)
    {
        return transformInvertMatrix(*this, m);
    }

    /// for square matrices
    /// @warning in-place simple symmetrization
    /// this = ( this + this.transposed() ) / 2.0
    void symmetrize()
    {
        static_assert( C == L, "" );
        for(int l=0; l<L; l++)
            for(int c=l+1; c<C; c++)
                this->elems[l][c] = this->elems[c][l] = ( this->elems[l][c] + this->elems[c][l] ) * 0.5f;
    }

};



template <int L, int C, typename real> Mat<L,L,real> Mat<L,C,real>::s_identity = Mat<L,L,real>::Identity();


/// Same as Mat except the values are not initialized by default
template <int L, int C, typename real=float>
class MatNoInit : public Mat<L,C,real>
{
public:
    MatNoInit()
        : Mat<L,C,real>(NOINIT)
    {
    }

    /// Assignment from an array of elements (stored per line).
    void operator=(const real* p)
    {
        this->Mat<L,C,real>::operator=(p);
    }

    /// Assignment from another matrix
    template<int L2, int C2, typename real2> void operator=(const Mat<L2,C2,real2>& m)
    {
        this->Mat<L,C,real>::operator=(m);
    }
};

/// Determinant of a 3x3 matrix.
template<class real>
inline real determinant(const Mat<3,3,real>& m)
{
    return m(0,0)*m(1,1)*m(2,2)
            + m(1,0)*m(2,1)*m(0,2)
            + m(2,0)*m(0,1)*m(1,2)
            - m(0,0)*m(2,1)*m(1,2)
            - m(1,0)*m(0,1)*m(2,2)
            - m(2,0)*m(1,1)*m(0,2);
}

/// Determinant of a 2x2 matrix.
template<class real>
inline real determinant(const Mat<2,2,real>& m)
{
    return m(0,0)*m(1,1)
            - m(1,0)*m(0,1);
}

/// Generalized-determinant of a 2x3 matrix.
/// Mirko Radi, "About a Determinant of Rectangular 2×n Matrix and its Geometric Interpretation"
template<class real>
inline real determinant(const Mat<2,3,real>& m)
{
    return m(0,0)*m(1,1) - m(0,1)*m(1,0) - ( m(0,0)*m(1,2) - m(0,2)*m(1,0) ) + m(0,1)*m(1,2) - m(0,2)*m(1,1);
}

/// Generalized-determinant of a 3x2 matrix.
/// Mirko Radi, "About a Determinant of Rectangular 2×n Matrix and its Geometric Interpretation"
template<class real>
inline real determinant(const Mat<3,2,real>& m)
{
    return m(0,0)*m(1,1) - m(1,0)*m(0,1) - ( m(0,0)*m(2,1) - m(2,0)*m(0,1) ) + m(1,0)*m(2,1) - m(2,0)*m(1,1);
}

// one-norm of a 3 x 3 matrix
template<class real>
inline real oneNorm(const Mat<3,3,real>& A)
{
    real norm = 0.0;
    for (int i=0; i<3; i++)
    {
        real columnAbsSum = helper::rabs(A(0,i)) + helper::rabs(A(1,i)) + helper::rabs(A(2,i));
        if (columnAbsSum > norm)
            norm = columnAbsSum;
    }
    return norm;
}

// inf-norm of a 3 x 3 matrix
template<class real>
inline real infNorm(const Mat<3,3,real>& A)
{
    real norm = 0.0;
    for (int i=0; i<3; i++)
    {
        real rowSum = helper::rabs(A(i,0)) + helper::rabs(A(i,1)) + helper::rabs(A(i,2));
        if (rowSum > norm)
            norm = rowSum;
    }
    return norm;
}

/// trace of a square matrix
template<int N, class real>
inline real trace(const Mat<N,N,real>& m)
{
    real t = m[0][0];
    for( int i=1 ; i<N ; ++i ) t += m[i][i];
    return t;
}

/// diagonal of a square matrix
template<int N, class real>
inline Vec<N,real> diagonal(const Mat<N,N,real>& m)
{
    Vec<N,real> v;
    for( int i=0 ; i<N ; ++i ) v[i] = m[i][i];
    return v;
}

#define MIN_DETERMINANT  1.0e-100

/// Matrix inversion (general case).
template<int S, class real>
bool invertMatrix(Mat<S,S,real>& dest, const Mat<S,S,real>& from)
{
    int i, j, k;
    Vec<S,int> r, c, row, col;

    Mat<S,S,real> m1 = from;
    Mat<S,S,real> m2;
    m2.identity();

    for ( k = 0; k < S; k++ )
    {
        // Choosing the pivot
        real pivot = 0;
        for (i = 0; i < S; i++)
        {
            if (row[i])
                continue;
            for (j = 0; j < S; j++)
            {
                if (col[j])
                    continue;
                real t = m1[i][j]; if (t<0) t=-t;
                if ( t > pivot)
                {
                    pivot = t;
                    r[k] = i;
                    c[k] = j;
                }
            }
        }

        if (pivot <= (real) MIN_DETERMINANT)
        {
            msg_error("Mat") << "invertMatrix finds too small determinant, matrix = "<<from;
            return false;
        }

        row[r[k]] = col[c[k]] = 1;
        pivot = m1[r[k]][c[k]];

        // Normalization
        m1[r[k]] /= pivot; m1[r[k]][c[k]] = 1;
        m2[r[k]] /= pivot;

        // Reduction
        for (i = 0; i < S; i++)
        {
            if (i != r[k])
            {
                real f = m1[i][c[k]];
                m1[i] -= m1[r[k]]*f; m1[i][c[k]] = 0;
                m2[i] -= m2[r[k]]*f;
            }
        }
    }

    for (i = 0; i < S; i++)
        for (j = 0; j < S; j++)
            if (c[j] == i)
                row[i] = r[j];

    for ( i = 0; i < S; i++ )
        dest[i] = m2[row[i]];

    return true;
}

/// Matrix inversion (special case 3x3).
template<class real>
bool invertMatrix(Mat<3,3,real>& dest, const Mat<3,3,real>& from)
{
    real det=determinant(from);

    if ( -(real) MIN_DETERMINANT<=det && det<=(real) MIN_DETERMINANT)
    {
        msg_error("Mat") << "invertMatrix finds too small determinant, matrix = "<<from;
        return false;
    }

    dest(0,0)= (from(1,1)*from(2,2) - from(2,1)*from(1,2))/det;
    dest(1,0)= (from(1,2)*from(2,0) - from(2,2)*from(1,0))/det;
    dest(2,0)= (from(1,0)*from(2,1) - from(2,0)*from(1,1))/det;
    dest(0,1)= (from(2,1)*from(0,2) - from(0,1)*from(2,2))/det;
    dest(1,1)= (from(2,2)*from(0,0) - from(0,2)*from(2,0))/det;
    dest(2,1)= (from(2,0)*from(0,1) - from(0,0)*from(2,1))/det;
    dest(0,2)= (from(0,1)*from(1,2) - from(1,1)*from(0,2))/det;
    dest(1,2)= (from(0,2)*from(1,0) - from(1,2)*from(0,0))/det;
    dest(2,2)= (from(0,0)*from(1,1) - from(1,0)*from(0,1))/det;

    return true;
}

/// Matrix inversion (special case 2x2).
template<class real>
bool invertMatrix(Mat<2,2,real>& dest, const Mat<2,2,real>& from)
{
    real det=determinant(from);

    if ( -(real) MIN_DETERMINANT<=det && det<=(real) MIN_DETERMINANT)
    {
        msg_error("Mat") << "invertMatrix finds too small determinant, matrix = "<<from;
        return false;
    }

    dest(0,0)=  from(1,1)/det;
    dest(0,1)= -from(0,1)/det;
    dest(1,0)= -from(1,0)/det;
    dest(1,1)=  from(0,0)/det;

    return true;
}
#undef MIN_DETERMINANT

/// Inverse Matrix considering the matrix as a transformation.
template<int S, class real>
bool transformInvertMatrix(Mat<S,S,real>& dest, const Mat<S,S,real>& from)
{
    Mat<S-1,S-1,real> R, R_inv;
    from.getsub(0,0,R);
    bool b = invertMatrix(R_inv, R);

    Mat<S-1,1,real> t, t_inv;
    from.getsub(0,S-1,t);
    t_inv = -1.*R_inv*t;

    dest.setsub(0,0,R_inv);
    dest.setsub(0,S-1,t_inv);
    for (unsigned int i=0; i<S-1; ++i)
        dest(S-1,i)=0.0;
    dest(S-1,S-1)=1.0;

    return b;
}

typedef Mat<1,1,float> Mat1x1f;
typedef Mat<1,1,double> Mat1x1d;

typedef Mat<2,2,float> Mat2x2f;
typedef Mat<2,2,double> Mat2x2d;

typedef Mat<3,3,float> Mat3x3f;
typedef Mat<3,3,double> Mat3x3d;

typedef Mat<3,4,float> Mat3x4f;
typedef Mat<3,4,double> Mat3x4d;

typedef Mat<4,4,float> Mat4x4f;
typedef Mat<4,4,double> Mat4x4d;

#ifdef SOFA_FLOAT
typedef Mat2x2f Matrix2;
typedef Mat3x3f Matrix3;
typedef Mat4x4f Matrix4;
#else
typedef Mat2x2d Matrix2;
typedef Mat3x3d Matrix3;
typedef Mat4x4d Matrix4;
#endif


template <int L, int C, typename real>
std::ostream& operator<<(std::ostream& o, const Mat<L,C,real>& m)
{
    o << '[' << m[0];
    for (int i=1; i<L; i++)
        o << ',' << m[i];
    o << ']';
    return o;
}

template <int L, int C, typename real>
std::istream& operator>>(std::istream& in, sofa::defaulttype::Mat<L,C,real>& m)
{
    int c;
    c = in.peek();
    while (c==' ' || c=='\n' || c=='[')
    {
        in.get();
        if( c=='[' ) break;
        c = in.peek();
    }
    in >> m[0];
    for (int i=1; i<L; i++)
    {
        c = in.peek();
        while (c==' ' || c==',')
        {
            in.get();
            c = in.peek();
        }
        in >> m[i];
    }
    c = in.peek();
    while (c==' ' || c=='\n' || c==']')
    {
        in.get();
        if( c==']' ) break;
        c = in.peek();
    }
    return in;
}




/// printing in other software formats

template <int L, int C, typename real>
void printMatlab(std::ostream& o, const Mat<L,C,real>& m)
{
    o<<"[";
    for(int l=0; l<L; ++l)
    {
        for(int c=0; c<C; ++c)
        {
            o<<m[l][c];
            if( c!=C-1 ) o<<",\t";
        }
        if( l!=L-1 ) o<<";"<<std::endl;
    }
    o<<"]"<<std::endl;
}


template <int L, int C, typename real>
void printMaple(std::ostream& o, const Mat<L,C,real>& m)
{
    o<<"matrix("<<L<<","<<C<<", [";
    for(int l=0; l<L; ++l)
    {
        for(int c=0; c<C; ++c)
        {
            o<<m[l][c];
            o<<",\t";
        }
        if( l!=L-1 ) o<<std::endl;
    }
    o<<"])"<<std::endl;
}



/// Create a matrix as \f$ u v^T \f$
template <int L, int C, typename T>
inline Mat<L,C,T> dyad( const Vec<L,T>& u, const Vec<C,T>& v )
{
    Mat<L,C,T> res(NOINIT);
    for( int i=0; i<L; i++ )
        for( int j=0; j<C; j++ )
            res[i][j] = u[i]*v[j];
    return res;
}

/// Compute the scalar product of two matrix (sum of product of all terms)
template <int L, int C, typename real>
inline real scalarProduct(const Mat<L,C,real>& left,const Mat<L,C,real>& right)
{
    real product(0.);
    for(int i=0; i<L; i++)
        for(int j=0; j<C; j++)
            product += left(i,j) * right(i,j);
    return product;
}


/// skew-symmetric mapping
/// crossProductMatrix(v) * x = v.cross(x)
template<class Real>
inline defaulttype::Mat<3, 3, Real> crossProductMatrix(const defaulttype::Vec<3, Real>& v)
{
    defaulttype::Mat<3, 3, Real> res;
    res[0][0]=0;
    res[0][1]=-v[2];
    res[0][2]=v[1];
    res[1][0]=v[2];
    res[1][1]=0;
    res[1][2]=-v[0];
    res[2][0]=-v[1];
    res[2][1]=v[0];
    res[2][2]=0;
    return res;
}


/// return a * b^T
template<int L,class Real>
static Mat<L,L,Real> tensorProduct(const Vec<L,Real> a, const Vec<L,Real> b )
{
    typedef Mat<L,L,Real> Mat;
    Mat m;

    for( typename Mat::size_type i=0 ; i<L ; ++i )
    {
        m[i][i] = a[i]*b[i];
        for( typename Mat::size_type j=i+1 ; j<L ; ++j )
            m[i][j] = m[j][i] = a[i]*b[j];
    }

    return m;
}



////////////////////////////////////////////
// Specialization of the defaulttype::DataTypeInfo type traits template
////////////////////////////////////////////

template<int L, int C, typename real>
struct DataTypeInfo< sofa::defaulttype::Mat<L,C,real> > : public FixedArrayTypeInfo<sofa::defaulttype::Mat<L,C,real> >
{
    static std::string name() { std::ostringstream o; o << "Mat<" << L << "," << C << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName<defaulttype::Mat1x1f> { static const char* name() { return "Mat1x1f"; } };
template<> struct DataTypeName<defaulttype::Mat1x1d> { static const char* name() { return "Mat1x1d"; } };
template<> struct DataTypeName<defaulttype::Mat2x2f> { static const char* name() { return "Mat2x2f"; } };
template<> struct DataTypeName<defaulttype::Mat2x2d> { static const char* name() { return "Mat2x2d"; } };
template<> struct DataTypeName<defaulttype::Mat3x3f> { static const char* name() { return "Mat3x3f"; } };
template<> struct DataTypeName<defaulttype::Mat3x3d> { static const char* name() { return "Mat3x3d"; } };
template<> struct DataTypeName<defaulttype::Mat3x4f> { static const char* name() { return "Mat3x4f"; } };
template<> struct DataTypeName<defaulttype::Mat3x4d> { static const char* name() { return "Mat3x4d"; } };
template<> struct DataTypeName<defaulttype::Mat4x4f> { static const char* name() { return "Mat4x4f"; } };
template<> struct DataTypeName<defaulttype::Mat4x4d> { static const char* name() { return "Mat4x4d"; } };

/// \endcond

} // namespace defaulttype

} // namespace sofa

#endif
