/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_MAT_H
#define SOFA_DEFAULTTYPE_MAT_H

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/Vec.h>
#include <cassert>
#include <boost/static_assert.hpp>
#include <iostream>

namespace sofa
{

namespace defaulttype
{

using std::cerr;
using std::endl;
template <int L, int C, class real=float>
class Mat : public helper::fixed_array<VecNoInit<C,real>,L>
    //class Mat : public Vec<L,Vec<C,real> >
{
public:

    enum { N = L*C };

    typedef real Real;
    typedef Vec<C,real> Line;
    typedef VecNoInit<C,real> LineNoInit;
    typedef Vec<L,real> Col;

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
        BOOST_STATIC_ASSERT(L == 1);
        this->elems[0]=r1;
      }
    */

    /// Specific constructor with 2 lines.
    Mat(Line r1, Line r2)
    {
        BOOST_STATIC_ASSERT(L == 2);
        this->elems[0]=r1;
        this->elems[1]=r2;
    }

    /// Specific constructor with 3 lines.
    Mat(Line r1, Line r2, Line r3)
    {
        BOOST_STATIC_ASSERT(L == 3);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
    }

    /// Specific constructor with 4 lines.
    Mat(Line r1, Line r2, Line r3, Line r4)
    {
        BOOST_STATIC_ASSERT(L == 4);
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
    Line& x() { BOOST_STATIC_ASSERT(L >= 1); return this->elems[0]; }
    /// Special access to second line.
    Line& y() { BOOST_STATIC_ASSERT(L >= 2); return this->elems[1]; }
    /// Special access to third line.
    Line& z() { BOOST_STATIC_ASSERT(L >= 3); return this->elems[2]; }
    /// Special access to fourth line.
    Line& w() { BOOST_STATIC_ASSERT(L >= 4); return this->elems[3]; }

    /// Special access to first line (read-only).
    const Line& x() const { BOOST_STATIC_ASSERT(L >= 1); return this->elems[0]; }
    /// Special access to second line (read-only).
    const Line& y() const { BOOST_STATIC_ASSERT(L >= 2); return this->elems[1]; }
    /// Special access to thrid line (read-only).
    const Line& z() const { BOOST_STATIC_ASSERT(L >= 3); return this->elems[2]; }
    /// Special access to fourth line (read-only).
    const Line& w() const { BOOST_STATIC_ASSERT(L >= 4); return this->elems[3]; }

    /// Set matrix to identity.
    void identity()
    {
        BOOST_STATIC_ASSERT(L == C);
        clear();
        for (int i=0; i<L; i++)
            this->elems[i][i]=1;
    }

    /// Set matrix as the transpose of m.
    void transpose(const Mat<C,L,real> &m)
    {
        for (int i=0; i<L; i++)
            for (int j=0; j<C; j++)
                this->elems[i][j]=m[j][i];
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

    /// Transpose current matrix.
    void transpose()
    {
        BOOST_STATIC_ASSERT(L == C);
        for (int i=0; i<L; i++)
            for (int j=i+1; j<C; j++)
            {
                real t = this->elems[i][j];
                this->elems[i][j] = this->elems[j][i];
                this->elems[j][i] = t;
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


    bool isSymetric() const
    {
        for (int i=0; i<L; i++)
            for (int j=i+1; j<C; j++)
                if( fabs( this->elems[i][j] - this->elems[j][i] ) > EQUALITY_THRESHOLD ) return false;
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

    /// Addition of the transposed of m (possible for square matrices only)
    void addTransposed(const Mat<L,C,real>& m)
    {
        BOOST_STATIC_ASSERT(L==C);
        for(int i=0; i<L; i++)
            for(int j=0; j<L; j++)
                (*this)[i][j] += m[j][i];
    }

    /// Substraction of the transposed of m (possible for square matrices only)
    void subTransposed(const Mat<L,C,real>& m)
    {
        BOOST_STATIC_ASSERT(L==C);
        for(int i=0; i<L; i++)
            for(int j=0; j<L; j++)
                (*this)[i][j] += m[j][i];
    }

    /// Substraction assignment operator.
    void operator -=(const Mat<L,C,real>& m)
    {
        for(int i=0; i<L; i++)
            this->elems[i]-=m[i];
    }

    /// Invert matrix m
    bool invert(const Mat<L,C,real>& m)
    {
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




};

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
            cerr<<"Warning (Mat.h) : invertMatrix finds too small determinant, matrix = "<<from<<endl;
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
        cerr<<"Warning (Mat.h) : invertMatrix finds too small determinant, matrix = "<<from<<endl;
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
        cerr<<"Warning (Mat.h) : invertMatrix finds too small determinant, matrix = "<<from<<endl;
        return false;
    }

    dest(0,0)=  from(1,1)/det;
    dest(0,1)= -from(0,1)/det;
    dest(1,0)= -from(1,0)/det;
    dest(1,1)=  from(0,0)/det;

    return true;
}
#undef MIN_DETERMINANT

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
        if( l!=L-1 ) o<<";"<<endl;
    }
    o<<"]"<<endl;
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
        if( l!=L-1 ) o<<endl;
    }
    o<<"])"<<endl;
}



/// return the max of two values
template<class T1,class T2>
inline const T1 S_MAX(const T1 &a, const T2 &b)
{
    return b > a ? (b) : (a);
}

/// return the min of two values
template<class T1,class T2>
inline const T1 S_MIN(const T1 &a, const T2 &b)
{
    return b < a ? (b) : (a);
}

template<class T1,class T2>
inline const T1 S_SIGN(const T1 &a, const T2 &b)
{
    return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

template<class T>
inline const T S_SQR(const T a)
{
    return a*a;
}

///Computes sqrt(a� + b�) without destructive underflow or overflow.
template <class T1, class T2>
T1 pythag(const T1 a, const T2 b)
{
    T1 absa,absb;
    absa=fabs(a);
    absb=fabs(b);
    if (absa > absb) return absa*sqrt(1.0+SQR(absb/absa));
    else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+SQR(absa/absb)));
}
/// Compute the SVD decomposition of matrix a (from nr). a is replaced by its pivoted LU decomposition. indx stores pivoting indices.
/** SVD decomposition   a = u.w.vt
\pre a: original matrix, destroyed to become u
\pre w: diagonal vector
\pre v: matrix
 */
template< int m, int n, typename Real>
void svddcmp(Mat<m,n,Real> &a, Vec<n,Real> &w, Mat<n,m,Real> &v)
{
    bool flag;
    int i,its,j,jj,k,l,nm;
    Real anorm,c,f,g,h,s,scale,x,y,z;

    Vec<n,Real> rv1;
    g=scale=anorm=0.0;
    for (i=0; i<n; i++)
    {
        l=i+2;
        rv1[i]=scale*g;
        g=s=scale=0.0;
        if (i < m)
        {
            for (k=i; k<m; k++) scale += fabs(a[k][i]);
            if (scale != 0.0)
            {
                for (k=i; k<m; k++)
                {
                    a[k][i] /= scale;
                    s += a[k][i]*a[k][i];
                }
                f=a[i][i];
                g = -S_SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][i]=f-g;
                for (j=l-1; j<n; j++)
                {
                    for (s=0.0,k=i; k<m; k++) s += a[k][i]*a[k][j];
                    f=s/h;
                    for (k=i; k<m; k++) a[k][j] += f*a[k][i];
                }
                for (k=i; k<m; k++) a[k][i] *= scale;
            }
        }
        w[i]=scale *g;
        g=s=scale=0.0;
        if (i+1 <= m && i != n)
        {
            for (k=l-1; k<n; k++) scale += fabs(a[i][k]);
            if (scale != 0.0)
            {
                for (k=l-1; k<n; k++)
                {
                    a[i][k] /= scale;
                    s += a[i][k]*a[i][k];
                }
                f=a[i][l-1];
                g = -S_SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][l-1]=f-g;
                for (k=l-1; k<n; k++) rv1[k]=a[i][k]/h;
                for (j=l-1; j<m; j++)
                {
                    for (s=0.0,k=l-1; k<n; k++) s += a[j][k]*a[i][k];
                    for (k=l-1; k<n; k++) a[j][k] += s*rv1[k];
                }
                for (k=l-1; k<n; k++) a[i][k] *= scale;
            }
        }
        anorm=S_MAX(anorm,(fabs(w[i])+fabs(rv1[i])));
    }
    for (i=n-1; i>=0; i--)
    {
        if (i < n-1)
        {
            if (g != 0.0)
            {
                for (j=l; j<n; j++)
                    v[j][i]=(a[i][j]/a[i][l])/g;
                for (j=l; j<n; j++)
                {
                    for (s=0.0,k=l; k<n; k++) s += a[i][k]*v[k][j];
                    for (k=l; k<n; k++) v[k][j] += s*v[k][i];
                }
            }
            for (j=l; j<n; j++) v[i][j]=v[j][i]=0.0;
        }
        v[i][i]=1.0;
        g=rv1[i];
        l=i;
    }
    for (i=S_MIN(m,n)-1; i>=0; i--)
    {
        l=i+1;
        g=w[i];
        for (j=l; j<n; j++) a[i][j]=0.0;
        if (g != 0.0)
        {
            g=1.0/g;
            for (j=l; j<n; j++)
            {
                for (s=0.0,k=l; k<m; k++) s += a[k][i]*a[k][j];
                f=(s/a[i][i])*g;
                for (k=i; k<m; k++) a[k][j] += f*a[k][i];
            }
            for (j=i; j<m; j++) a[j][i] *= g;
        }
        else for (j=i; j<m; j++) a[j][i]=0.0;
        ++a[i][i];
    }
    for (k=n-1; k>=0; k--)
    {
        for (its=0; its<30; its++)
        {
            flag=true;
            for (l=k; l>=0; l--)
            {
                nm=l-1;
                if (fabs(rv1[l])+anorm == anorm)
                {
                    flag=false;
                    break;
                }
                if (fabs(w[nm])+anorm == anorm) break;
            }
            if (flag)
            {
                c=0.0;
                s=1.0;
                for (i=l-1; i<k+1; i++)
                {
                    f=s*rv1[i];
                    rv1[i]=c*rv1[i];
                    if (fabs(f)+anorm == anorm) break;
                    g=w[i];
                    h=pythag(f,g);
                    w[i]=h;
                    h=1.0/h;
                    c=g*h;
                    s = -f*h;
                    for (j=0; j<m; j++)
                    {
                        y=a[j][nm];
                        z=a[j][i];
                        a[j][nm]=y*c+z*s;
                        a[j][i]=z*c-y*s;
                    }
                }
            }
            z=w[k];
            if (l == k)
            {
                if (z < 0.0)
                {
                    w[k] = -z;
                    for (j=0; j<n; j++) v[j][k] = -v[j][k];
                }
                break;
            }
            if (its == 29)
            {
// 				std::cerr<<"Warning: Mat.h :: svddcmp: no convergence in 30 svdcmp iterations"<<std::endl;
                return;
            }
            x=w[l];
            nm=k-1;
            y=w[nm];
            g=rv1[nm];
            h=rv1[k];
            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
            g=pythag(f,1.0);
            f=((x-z)*(x+z)+h*((y/(f+S_SIGN(g,f)))-h))/x;
            c=s=1.0;
            for (j=l; j<=nm; j++)
            {
                i=j+1;
                g=rv1[i];
                y=w[i];
                h=s*g;
                g=c*g;
                z=pythag(f,h);
                rv1[j]=z;
                c=f/z;
                s=h/z;
                f=x*c+g*s;
                g=g*c-x*s;
                h=y*s;
                y *= c;
                for (jj=0; jj<n; jj++)
                {
                    x=v[jj][j];
                    z=v[jj][i];
                    v[jj][j]=x*c+z*s;
                    v[jj][i]=z*c-x*s;
                }
                z=pythag(f,h);
                w[j]=z;
                if (z)
                {
                    z=1.0/z;
                    c=f*z;
                    s=h*z;
                }
                f=c*g+s*y;
                x=c*y-s*g;
                for (jj=0; jj<m; jj++)
                {
                    y=a[jj][j];
                    z=a[jj][i];
                    a[jj][j]=y*c+z*s;
                    a[jj][i]=z*c-y*s;
                }
            }
            rv1[l]=0.0;
            rv1[k]=f;
            w[k]=x;
        }
    }
}



/// return the condition number of the matrix a following the euclidian norm (using the svd decomposition to find singular values)
template< int m, int n, typename Real>
Real cond(Mat<m,n,Real> &a)
{
    Vec<n,Real>w;
    Mat<n,m,Real> *v = new Mat<n,m,Real>();

    svddcmp( a, w, *v );

    delete v;

    return fabs(w[0]/w[n-1]);
}


/// Compute the LU decomposition of matrix a. a is replaced by its pivoted LU decomposition. indx stores pivoting indices.
template< int n, typename Real>
void ludcmp(Mat<n,n,Real> &a, Vec<n,int> &indx)
{
    const Real TINY=(Real)1.0e-20;
    int i,imax=0,j,k;
    Real big,dum,sum,temp;

    Vec<n,Real> vv;
    for (i=0; i<n; i++)
    {
        big=0.0;
        for (j=0; j<n; j++)
            if ((temp=fabs(a[i][j])) > big) big=temp;
        assert (big != 0.0);
        vv[i]=(Real)1.0/big;
    }
    for (j=0; j<n; j++)
    {
        for (i=0; i<j; i++)
        {
            sum=a[i][j];
            for (k=0; k<i; k++) sum -= a[i][k]*a[k][j];
            a[i][j]=sum;
        }
        big=0.0;
        for (i=j; i<n; i++)
        {
            sum=a[i][j];
            for (k=0; k<j; k++) sum -= a[i][k]*a[k][j];
            a[i][j]=sum;
            if ((dum=vv[i]*fabs(sum)) >= big)
            {
                big=dum;
                imax=i;
            }
        }
        if (j != imax)
        {
            for (k=0; k<n; k++)
            {
                dum=a[imax][k];
                a[imax][k]=a[j][k];
                a[j][k]=dum;
            }
            vv[imax]=vv[j];
        }
        indx[j]=imax;
        if (a[j][j] == 0.0) a[j][j]=TINY;
        if (j != n-1)
        {
            dum=(Real)1.0/(a[j][j]);
            for (i=j+1; i<n; i++) a[i][j] *= dum;
        }
    }
}

/// Compute the solution of Mx=b. b is replaced by x. a and indx together represent the LU decomposition of m, as given my method ludcmp.
template< int n, typename Real>
void lubksb(const Mat<n,n,Real> &a, const Vec<n,int> &indx, Vec<n,Real> &b)
{
    int i,ii=0,ip,j;
    Real sum;

    for (i=0; i<n; i++)
    {
        ip=indx[i];
        sum=b[ip];
        b[ip]=b[i];
        if (ii != 0)
            for (j=ii-1; j<i; j++) sum -= a[i][j]*b[j];
        else if (sum != 0.0)
            ii=i+1;
        b[i]=sum;
    }
    for (i=n-1; i>=0; i--)
    {
        sum=b[i];
        for (j=i+1; j<n; j++) sum -= a[i][j]*b[j];
        b[i]=sum/a[i][i];
    }
}

/** Compute the inverse of matrix m.
\warning Matrix m is replaced by its LU decomposition.
*/
template< int n, typename Real>
void luinv( Mat<n,n,Real> &inv, Mat<n,n,Real> &m )
{
    Vec<n,int> idx;
    Vec<n,Real> col;

    ludcmp(m,idx);

    for( int i=0; i<n; i++ )
    {
        for( int j=0; j<n; j++ )
            col[j] = 0;
        col[i] = 1;
        lubksb(m,idx,col);
        for( int j=0; j<n; j++ )
            inv[j][i] = col[j];
    }
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



} // namespace defaulttype

} // namespace sofa

// Specialization of the defaulttype::DataTypeInfo type traits template

namespace sofa
{

namespace defaulttype
{

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
