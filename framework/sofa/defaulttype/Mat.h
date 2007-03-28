/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
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
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_MAT_H
#define SOFA_DEFAULTTYPE_MAT_H

#include <sofa/defaulttype/Vec.h>
#include <assert.h>
#include <sofa/helper/static_assert.h>

namespace sofa
{

namespace defaulttype
{

template <int L, int C, class real=float>
class Mat : public helper::fixed_array<Vec<C,real>,L>
    //class Mat : public Vec<L,Vec<C,real> >
{
public:

    enum { N = L*C };

    typedef real Real;
    typedef Vec<C,real> Line;
    typedef Vec<L,real> Col;

    Mat()
    {
    }

    /*
      /// Specific constructor with a single line.
      Mat(Line r1)
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
    Mat(const real& v)
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
    Line& operator[](int i)
    {
        return this->elems[i];
    }

    /// Read-only access to line i.
    const Line& operator[](int i) const
    {
        return this->elems[i];
    }

    /// Write acess to line i.
    Line& operator()(int i)
    {
        return this->elems[i];
    }

    /// Read-only access to line i.
    const Line& operator()(int i) const
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
        Mat<C,L,real> m;
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

    // LINEAR ALGEBRA

    /// Matrix multiplication operator.
    template <int P>
    Mat<L,P,real> operator*(const Mat<C,P,real>& m) const
    {
        Mat<L,P,real> r;
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
        Mat r;
        for(int i = 0; i < L; i++)
            r[i] = (*this)[i] + m[i];
        return r;
    }

    /// Matrix subtraction operator.
    Mat<L,C,real> operator-(const Mat<L,C,real>& m) const
    {
        Mat r;
        for(int i = 0; i < L; i++)
            r[i] = (*this)[i] - m[i];
        return r;
    }

    /// Multiplication operator Matrix * Line.
    Col operator*(const Line& v) const
    {
        Col r;
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
        Line r;
        for(int i=0; i<C; i++)
        {
            r[i]=(*this)[0][i] * v[0];
            for(int j=1; j<L; j++)
                r[i] += (*this)[j][i] * v[j];
        }
        return r;
    }



    /// Scalar multiplication operator.
    Mat<L,C,real> operator*(real f) const
    {
        Mat<L,C,real> r;
        for(int i=0; i<L; i++)
            for(int j=0; j<C; j++)
                r[i][j] = (*this)[i][j] * f;
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

        if (pivot <= 1e-10)
        {
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

    if ( -1e-10<=det && det<=1e-10)
        return false;

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

    if ( -1e-10<=det && det<=1e-10)
        return false;

    dest(0,0)=  from(1,1)/det;
    dest(0,1)= -from(0,1)/det;
    dest(1,0)= -from(1,0)/det;
    dest(1,1)=  from(0,0)/det;

    return true;
}

typedef Mat<2,2,float> Mat2x2f;
typedef Mat<2,2,double> Mat2x2d;

typedef Mat<3,3,float> Mat3x3f;
typedef Mat<3,3,double> Mat3x3d;

typedef Mat<3,4,float> Mat3x4f;
typedef Mat<3,4,double> Mat3x4d;

typedef Mat<4,4,float> Mat4x4f;
typedef Mat<4,4,double> Mat4x4d;

typedef Mat2x2d Matrix2;
typedef Mat3x3d Matrix3;
typedef Mat4x4d Matrix4;



template <int L, int C, typename real>
std::ostream& operator<<(std::ostream& o, const Mat<L,C,real>& m)
{
    o << '<' << m[0];
    for (int i=1; i<L; i++)
        o << ',' << m[i];
    o << '>';
    return o;
}

template <int L, int C, typename real>
std::istream& operator>>(std::istream& in, sofa::defaulttype::Mat<L,C,real>& m)
{
    int c;
    c = in.peek();
    while (c==' ' || c=='\n' || c=='<')
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
    while (c==' ' || c=='\n' || c=='>')
    {
        in.get();
        c = in.peek();
    }
    return in;
}

/// Compute the LU decomposition of matrix a. a is replaced by its pivoted LU decomposition. indx stores pivoting indices.
template< int n, typename Real>
void ludcmp(Mat<n,n,Real> &a, Vec<n,int> &indx)
{
    const Real TINY=1.0e-20;
    int i,imax=0,j,k;
    Real big,dum,sum,temp;

    Vec<n,Real> vv;
    for (i=0; i<n; i++)
    {
        big=0.0;
        for (j=0; j<n; j++)
            if ((temp=fabs(a[i][j])) > big) big=temp;
        assert (big != 0.0);
        vv[i]=1.0/big;
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
            dum=1.0/(a[j][j]);
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
    Mat<L,C,T> res;
    for( int i=0; i<L; i++ )
        for( int j=0; j<C; j++ )
            res[i][j] = u[i]*v[j];
    return res;
}

} // namespace defaulttype

} // namespace sofa

// iostream



/// Scalar matrix multiplication operator.
template <int L, int C, typename real>
sofa::defaulttype::Mat<L,C,real> operator*(real r, const sofa::defaulttype::Mat<L,C,real>& m)
{
    return m*r;
}

#endif
