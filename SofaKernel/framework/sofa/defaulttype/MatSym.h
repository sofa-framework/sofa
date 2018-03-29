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
#ifndef SOFA_DEFAULTTYPE_MATSYM_H
#define SOFA_DEFAULTTYPE_MATSYM_H

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/Vec.h>
#include <cassert>
#include <iostream>
#include <sofa/defaulttype/Mat.h>



namespace sofa
{

namespace defaulttype
{
////class for 3*3 symmetric matrix only

template <int D,class real=float>
class MatSym : public VecNoInit<D*(D+1)/2,real>
//class Mat : public Vec<L,Vec<C,real> >
{
public:

    // enum { N = L*C };

    typedef real Real;
    typedef Vec<D,Real> Coord;


    MatSym()
    {
        clear();
    }

    explicit MatSym(NoInit)
    {
    }
    /// Constructor from 6 elements
    explicit MatSym(const real& v1,const real& v2,const real& v3,const real& v4,const real& v5,const real& v6)
    {
        this->elems[0] = v1;
        this->elems[1] = v2;
        this->elems[2] = v3;
        this->elems[3] = v4;
        this->elems[4] = v5;
        this->elems[5] = v6;
    }


    /// Constructor from an element
    explicit MatSym(const int sizeM,const real& v)
    {
        for( int i=0; i<sizeM*(sizeM+1)/2; i++ )
            this->elems[i] = v;
    }

    /// Constructor from another matrix
    template<typename real2>
    MatSym(const MatSym<D,real2>& m)
    {
        std::copy(m.begin(), m.begin()+D*(D+1)/2, this->begin());
    }


    /// Assignment from another matrix
    template<typename real2> void operator=(const MatSym<D,real2>& m)
    {
        std::copy(m.begin(), m.begin()+D*(D+1)/2, this->begin());
    }


    /// Sets each element to 0.
    void clear()
    {
        for (int i=0; i<D*(D+1)/2; i++)
            this->elems[i]=0;
    }

    /// Sets each element to r.
    void fill(real r)
    {
        for (int i=0; i<D*(D+1)/2; i++)
            this->elems[i].fill(r);
    }

    /// Write access to element (i,j).
    inline real& operator()(int i, int j)
    {
        if(i>=j)
        {  return this->elems[(i*(i+1))/2+j];}
        else
        {return this->elems[(j*(j+1))/2+i];}
    }

    /// Read-only access to element (i,j).
    inline const real& operator()(int i, int j) const
    {
        if(i>=j)
        {  return this->elems[(i*(i+1))/2+j];}
        else
        {return this->elems[(j*(j+1))/2+i];}
    }

    //convert matrix to sym
    //template<int D>
    void Mat2Sym( const Mat<D,D,real>& M, MatSym<D,real>& W)
    {
        for (int j=0; j<D; j++)
            for (int i=0; i <= j; i++)
                W(i,j) = (Real)((M(i,j) + M(j,i))/2.0);
    }

    // convert to Voigt notation

    inline Vec<D*(D+1)/2 ,real> getVoigt()
    {
        Vec<D*(D+1)/2 ,real> result;
        if (D==2)
        {
            result[0] = this->elems[0]; result[1] = this->elems[2]; result[2] = 2*this->elems[1];
        }
        else
        {
            result[0] = this->elems[0]; result[1] = this->elems[2]; result[2] = this->elems[5];
            result[3]=2*this->elems[4]; result[4]=2*this->elems[3]; result[5]=2*this->elems[1];
        }
        return result;

    }


    //convert into 3*3 matrix

    /*  Mat<D,D,real> convert() const
    {
      Mat<D,D,real> m;
      for(int k=0; k<D;k++){
    	for (int l=0;l<k;l++){
    		m[k][l]=m[l][k]=(*this)(k,l);
    	}
    }
      return m;
    }
     */
    /// Set matrix to identity.
    void identity()
    {
        for (int i=0; i<D; i++)
        {
            this->elems[i*(i+1)/2+i]=1;
            for (int j=i+1; j<D; j++)
            {
                this->elems[i*(i+1)/2+j]=0;
            }
        }
    }

    /// @name Tests operators
    /// @{

    bool operator==(const MatSym<D,real>& b) const
    {
        for (int i=0; i<D*(D+1)/2; i++)
            if (!(this->elems[i]==b[i])) return false;
        return true;
    }

    bool operator!=(const MatSym< D,real>& b) const
    {
        for (int i=0; i<D*(D+1)/2; i++)
            if (this->elems[i]!=b[i]) return true;
        return false;
    }


    /// @}

    // LINEAR ALGEBRA

    /// Matrix multiplication operator: product of two symmetric matrices
    //template <int D>
    Mat<D,D,real> SymSymMultiply(const MatSym<D,real>& m) const
    {
        Mat<D,D,real> r(NOINIT);

        for(int i=0; i<D; i++)
        {
            for(int j=0; j<D; j++)
            {
                r[i][j]=(*this)(i,0) * m(0,j);
                for(int k=1; k<D; k++) { r[i][j] += (*this)(i,k) * m(k,j);}
            }
        }
        return r;
    }

    //Multiplication by a non symmetric matrix on the right

    // template <int D>
    Mat<D,D,real> SymMatMultiply(const Mat<D,D,real>& m) const
    {
        Mat<D,D,real> r(NOINIT);

        for(int i=0; i<D; i++)
        {
            for(int j=0; j<D; j++)
            {
                r[i][j]=(*this)(i,0) * m[0][j];
                for(int k=1; k<D; k++)
                {
                    r[i][j] += (*this)(i,k) * m[k][j];
                }
            }
        }
        return r;
    }
    //Multiplication by a non symmetric matrix on the left

    // template <int D>
    Mat<D,D,real> MatSymMultiply(const Mat<D,D,real>& m) const
    {
        Mat<D,D,real> r(NOINIT);

        for(int i=0; i<D; i++)
        {
            for(int j=0; j<D; j++)
            {
                r[i][j]=m(i,0)* (*this)(0,j);
                for(int k=1; k<D; k++)
                {
                    r[i][j] += m(i,k) * (*this)(k,j);
                }
            }
        }
        return r;
    }


    /// Matrix addition operator with a symmetric matrix
    MatSym< D,real> operator+(const MatSym<D,real>& m) const
    {
        MatSym< D,real> r;
        for(int i = 0; i < D*(D+1)/2; i++)
            r[i] = (*this)[i] + m[i];
        return r;
    }

    /// Matrix addition operator with a non-symmetric matrix
    Mat<D,D,real> operator+(const Mat<D,D,real>& m) const
    {
        Mat<D,D,real> r(NOINIT);
        for(int i = 0; i < D; i++)
        {
            for(int j=0; j<D; j++)
            {
                r[i][j]=(*this)(i,j)+m[i][j];
            }
        }
        return r;

    }
    /// Matrix substractor operator with a symmetric matrix
    MatSym< D,real> operator-(const MatSym< D,real>& m) const
    {
        MatSym<D,real> r;
        for(int i = 0; i < D*(D+1)/2; i++)
            r[i] = (*this)[i] - m[i];
        return r;
    }

    /// Matrix substractor operator with a non-symmetric matrix
    Mat<D,D,real> operator-(const Mat<D,D,real>& m) const
    {
        Mat<D,D,real> r(NOINIT);
        for(int i = 0; i < D; i++)
        {
            for(int j=0; j<D; j++)
            {
                r[i][j]=(*this)(i,j)-m[i][j];
            }
        }
        return r;

    }


    /// Multiplication operator Matrix * Vector.
    Coord operator*(const Coord& v) const
    {


        Coord r(NOINIT);
        for(int i=0; i<D; i++)
        {
            r[i]=(*this)(i,0) * v[0];
            for(int j=1; j<D; j++)
                r[i] += (*this)(i,j) * v[j];
        }
        return r;
    }


    /// Scalar multiplication operator.
    MatSym<D,real> operator*(real f) const
    {
        MatSym<D,real> r(NOINIT);
        for(int i=0; i<D*(D+1)/2; i++)
            r[i] = (*this)[i] * f;
        return r;
    }

    /// Scalar matrix multiplication operator.
    friend MatSym<D,real> operator*(real r, const MatSym< D,real>& m)
    {
        return m*r;
    }

    /// Scalar division operator.
    MatSym< D,real> operator/(real f) const
    {
        MatSym< D,real> r(NOINIT);
        for(int i=0; i<D*(D+1)/2; i++)
            r[i] = (*this)[i] / f;
        return r;
    }

    /// Scalar multiplication assignment operator.
    void operator *=(real r)
    {
        for(int i=0; i<D*(D+1)/2; i++)
            this->elems[i]*=r;
    }

    /// Scalar division assignment operator.
    void operator /=(real r)
    {
        for(int i=0; i<D*(D+1)/2; i++)
            this->elems[i]/=r;
    }

    /// Addition assignment operator.
    void operator +=(const MatSym< D,real>& m)
    {
        for(int i=0; i<D*(D+1)/2; i++)
            this->elems[i]+=m[i];
    }



    /// Substraction assignment operator.
    void operator -=(const MatSym< D,real>& m)
    {
        for(int i=0; i<D*(D+1)/2; i++)
            this->elems[i]-=m[i];
    }

    /// Invert matrix m
    bool invert(const MatSym<D,real>& m)
    {

        return invertMatrix((*this), m);

    }



};



/// Determinant of a 3x3 matrix.
template<class real>
inline real determinant(const MatSym<3,real>& m)
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
inline real determinant(const MatSym<2,real>& m)
{
    //     m(0,0)*m(1,1) - m(1,0)*m(0,1);
    return m(0,0)*m(1,1) - m(0,1)*m(0,1);
}

#define MIN_DETERMINANT  1.0e-100


/// Trace of a 3x3 matrix.
template<class real>
inline real trace(const MatSym<3,real>& m)
{
    return m(0,0)+m(1,1)+m(2,2);

}
/// Trace of a 2x2 matrix.
template<class real>
inline real trace(const MatSym<2,real>& m)
{
    return m(0,0)+m(1,1);
}

/// Matrix inversion (general case).
template<int S, class real>
bool invertMatrix(MatSym<S,real>& dest, const MatSym<S,real>& from)
{
    int i, j, k;
    Vec<S,int> r, c, row, col;

    MatSym<S,real> m1 = from;
    MatSym<S,real> m2;
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
                real t = m1(i,j); if (t<0) t=-t;
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
            msg_error("MatSym") << "invertMatrix finds too small determinant, matrix = "<<from;
            return false;
        }

        row[r[k]] = col[c[k]] = 1;
        pivot = m1(r[k],c[k]);

        // Normalization
        for (j=0; j<S; j++)
        {
            m1[r[k]*(r[k]+1)/2+j] /= pivot;
            m1(r[k],c[k]) = 1;
            m2[r[k]*(r[k]+1)/2+j] /= pivot;
        }

        // Reduction
        for (i = 0; i < S; i++)
        {
            if (i != r[k])
            {
                for (j=0; j<S; j++)
                {

                    real f = m1(i,c[k]);
                    m1[i*(i+1)/2+j] -= m1[r[k]*(r[k]+1)/2+j]*f; m1(i,c[k]) = 0;
                    m2[i*(i+1)/2+j] -= m2[r[k]*(r[k]+1)/2+j]*f;
                }
            }
        }
    }

    for (i = 0; i < S; i++)
        for (j = 0; j < S; j++)
            if (c[j] == i)
                row[i] = r[j];

    for ( i = 0; i < S; i++ )
    {
        for (j=0; j<S; j++)
        {
            dest[i*(i+1)/2+j] = m2[row[i]*(row[i]+1)/2+j];
        }
    }

    return true;
}

/// Matrix inversion (special case 3x3).
template<class real>
bool invertMatrix(MatSym<3,real>& dest, const MatSym<3,real>& from)
{
    real det=determinant(from);

    if ( -(real) MIN_DETERMINANT<=det && det<=(real) MIN_DETERMINANT)
    {
        msg_error("MatSym") << "invertMatrix finds too small determinant, matrix = "<<from;
        return false;
    }

    dest(0,0)= (from(1,1)*from(2,2) - from(2,1)*from(1,2))/det;
    dest(1,0)= (from(1,2)*from(2,0) - from(2,2)*from(1,0))/det;
    dest(2,0)= (from(1,0)*from(2,1) - from(2,0)*from(1,1))/det;
    dest(1,1)= (from(2,2)*from(0,0) - from(0,2)*from(2,0))/det;
    dest(2,1)= (from(2,0)*from(0,1) - from(0,0)*from(2,1))/det;
    dest(2,2)= (from(0,0)*from(1,1) - from(1,0)*from(0,1))/det;

    return true;
}

/// Matrix inversion (special case 2x2).
template<class real>
bool invertMatrix(MatSym<2,real>& dest, const MatSym<2,real>& from)
{
    real det=determinant(from);

    if ( -(real) MIN_DETERMINANT<=det && det<=(real) MIN_DETERMINANT)
    {
        msg_error("MatSym") << "invertMatrix finds too small determinant, matrix = "<<from;
        return false;
    }

    dest(0,0)=  from(1,1)/det;
    dest(0,1)= -from(0,1)/det;
    //dest(1,0)= -from(1,0)/det;
    dest(1,1)=  from(0,0)/det;

    return true;
}
#undef MIN_DETERMINANT
/*
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
//////////////////////////////////////////////////////////
*/
template<int D,class real>
std::ostream& operator<<(std::ostream& o, const MatSym<D,real>& m)
{
    o << '[' ;
    for(int i=0; i<D; i++)
    {
        for(int j=0; j<D; j++)
        {
            o<<" "<<m(i,j);
        }
        o<<" ,";
    }
    o << ']';
    return o;
}

template<int D,class real>
std::istream& operator>>(std::istream& in, MatSym<D,real>& m)
{
    int c;
    c = in.peek();
    while (c==' ' || c=='\n' || c=='[')
    {
        in.get();
        if( c=='[' ) break;
        c = in.peek();
    }
    ///////////////////////////////////////////////
    for(int i=0; i<D; i++)
    {
        c = in.peek();
        while (c==' ' || c==',')
        {
            in.get(); c = in.peek();
        }

        for(int j=0; j<D; j++)
        {
            in >> m(i,j);
        }

    }

    ////////////////////////////////////////////////
    c = in.peek();
    while (c==' ' || c=='\n' || c==']')
    {
        in.get();
        if( c==']' ) break;
        c = in.peek();
    }
    return in;
}

/// Compute the scalar product of two matrix (sum of product of all terms)
template <int D, typename real>
inline real scalarProduct(const MatSym<D,real>& left, const MatSym<D,real>& right)
{
    real sympart(0.),dialpart(0.);
    for(int i=0; i<D; i++)
        for(int j=i+1; j<D; j++)
            sympart += left(i,j) * right(i,j);

    for(int d=0; d<D; d++)
        dialpart += left(d,d) * right(d,d);


    return 2. * sympart  + dialpart ;
}

template <int D, typename real>
inline real scalarProduct(const MatSym<D,real>& left, const Mat<D,D,real>& right)
{
    real product(0.);
    for(int i=0; i<D; i++)
        for(int j=0; j<D; j++)
            product += left(i,j) * right(i,j);
    return product;
}

template <int D, typename real>
inline real scalarProduct(const Mat<D,D,real>& left, const MatSym<D,real>& right)
{
    return scalarProduct(right, left);
}

} // namespace defaulttype

} // namespace sofa

// iostream

#endif
