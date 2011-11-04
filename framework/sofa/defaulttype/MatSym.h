/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_MATSYM_H
#define SOFA_DEFAULTTYPE_MATSYM_H

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/Vec.h>
#include <cassert>
#include <boost/static_assert.hpp>
#include <iostream>
#include <sofa/defaulttype/Mat.h>



namespace sofa
{

namespace defaulttype
{
using std::cerr;
using std::endl;
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
            cerr<<"Warning: invertMatrix finds too small determinant, matrix = "<<from<<endl;
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
        cerr<<"Warning: invertMatrix finds too small determinant, matrix = "<<from<<endl;
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
        cerr<<"Warning: invertMatrix finds too small determinant, matrix = "<<from<<endl;
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
        c = in.peek();
    }
    return in;
}


/*
//////////////////////////////////////////////////////////////////////////////////
/// printing in other software formats

template <int L, int C, typename real>
void printMatlab(std::ostream& o, const Mat<L,C,real>& m)
{
	o<<"[";
	for(int l=0;l<L;++l)
	{
		for(int c=0;c<C;++c)
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
	for(int l=0;l<L;++l)
	{
		for(int c=0;c<C;++c)
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
// SVD decomposition   a = u.w.vt
\pre a: original matrix, destroyed to become u
\pre w: diagonal vector
\pre v: matrix

template< int m, int n, typename Real>
void svddcmp(Mat<m,n,Real> &a, Vec<n,Real> &w, Mat<n,m,Real> &v)
{
	bool flag;
	int i,its,j,jj,k,l,nm;
	Real anorm,c,f,g,h,s,scale,x,y,z;

	Vec<n,Real> rv1;
	g=scale=anorm=0.0;
	for (i=0;i<n;i++) {
		l=i+2;
		rv1[i]=scale*g;
		g=s=scale=0.0;
		if (i < m) {
			for (k=i;k<m;k++) scale += fabs(a[k][i]);
			if (scale != 0.0) {
				for (k=i;k<m;k++) {
					a[k][i] /= scale;
					s += a[k][i]*a[k][i];
				}
				f=a[i][i];
				g = -S_SIGN(sqrt(s),f);
				h=f*g-s;
				a[i][i]=f-g;
				for (j=l-1;j<n;j++) {
					for (s=0.0,k=i;k<m;k++) s += a[k][i]*a[k][j];
					f=s/h;
					for (k=i;k<m;k++) a[k][j] += f*a[k][i];
				}
				for (k=i;k<m;k++) a[k][i] *= scale;
			}
		}
		w[i]=scale *g;
		g=s=scale=0.0;
		if (i+1 <= m && i != n) {
			for (k=l-1;k<n;k++) scale += fabs(a[i][k]);
			if (scale != 0.0) {
				for (k=l-1;k<n;k++) {
					a[i][k] /= scale;
					s += a[i][k]*a[i][k];
				}
				f=a[i][l-1];
				g = -S_SIGN(sqrt(s),f);
				h=f*g-s;
				a[i][l-1]=f-g;
				for (k=l-1;k<n;k++) rv1[k]=a[i][k]/h;
				for (j=l-1;j<m;j++) {
					for (s=0.0,k=l-1;k<n;k++) s += a[j][k]*a[i][k];
					for (k=l-1;k<n;k++) a[j][k] += s*rv1[k];
				}
				for (k=l-1;k<n;k++) a[i][k] *= scale;
			}
		}
		anorm=S_MAX(anorm,(fabs(w[i])+fabs(rv1[i])));
	}
	for (i=n-1;i>=0;i--) {
		if (i < n-1) {
			if (g != 0.0) {
				for (j=l;j<n;j++)
					v[j][i]=(a[i][j]/a[i][l])/g;
				for (j=l;j<n;j++) {
					for (s=0.0,k=l;k<n;k++) s += a[i][k]*v[k][j];
					for (k=l;k<n;k++) v[k][j] += s*v[k][i];
				}
			}
			for (j=l;j<n;j++) v[i][j]=v[j][i]=0.0;
		}
		v[i][i]=1.0;
		g=rv1[i];
		l=i;
	}
	for (i=S_MIN(m,n)-1;i>=0;i--) {
		l=i+1;
		g=w[i];
		for (j=l;j<n;j++) a[i][j]=0.0;
		if (g != 0.0) {
			g=1.0/g;
			for (j=l;j<n;j++) {
				for (s=0.0,k=l;k<m;k++) s += a[k][i]*a[k][j];
				f=(s/a[i][i])*g;
				for (k=i;k<m;k++) a[k][j] += f*a[k][i];
			}
			for (j=i;j<m;j++) a[j][i] *= g;
		} else for (j=i;j<m;j++) a[j][i]=0.0;
		++a[i][i];
	}
	for (k=n-1;k>=0;k--) {
		for (its=0;its<30;its++) {
			flag=true;
			for (l=k;l>=0;l--) {
				nm=l-1;
				if (fabs(rv1[l])+anorm == anorm) {
					flag=false;
					break;
				}
				if (fabs(w[nm])+anorm == anorm) break;
			}
			if (flag) {
				c=0.0;
				s=1.0;
				for (i=l-1;i<k+1;i++) {
					f=s*rv1[i];
					rv1[i]=c*rv1[i];
					if (fabs(f)+anorm == anorm) break;
					g=w[i];
					h=pythag(f,g);
					w[i]=h;
					h=1.0/h;
					c=g*h;
					s = -f*h;
					for (j=0;j<m;j++) {
						y=a[j][nm];
						z=a[j][i];
						a[j][nm]=y*c+z*s;
						a[j][i]=z*c-y*s;
					}
				}
			}
			z=w[k];
			if (l == k) {
				if (z < 0.0) {
					w[k] = -z;
					for (j=0;j<n;j++) v[j][k] = -v[j][k];
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
			for (j=l;j<=nm;j++) {
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
				for (jj=0;jj<n;jj++) {
					x=v[jj][j];
					z=v[jj][i];
					v[jj][j]=x*c+z*s;
					v[jj][i]=z*c-x*s;
				}
				z=pythag(f,h);
				w[j]=z;
				if (z) {
					z=1.0/z;
					c=f*z;
					s=h*z;
				}
				f=c*g+s*y;
				x=c*y-s*g;
				for (jj=0;jj<m;jj++) {
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
    for (i=0;i<n;i++) {
        big=0.0;
        for (j=0;j<n;j++)
            if ((temp=fabs(a[i][j])) > big) big=temp;
        assert (big != 0.0);
        vv[i]=(Real)1.0/big;
    }
    for (j=0;j<n;j++) {
        for (i=0;i<j;i++) {
            sum=a[i][j];
            for (k=0;k<i;k++) sum -= a[i][k]*a[k][j];
            a[i][j]=sum;
        }
        big=0.0;
        for (i=j;i<n;i++) {
            sum=a[i][j];
            for (k=0;k<j;k++) sum -= a[i][k]*a[k][j];
            a[i][j]=sum;
            if ((dum=vv[i]*fabs(sum)) >= big) {
                big=dum;
                imax=i;
            }
        }
        if (j != imax) {
            for (k=0;k<n;k++) {
                dum=a[imax][k];
                a[imax][k]=a[j][k];
                a[j][k]=dum;
            }
            vv[imax]=vv[j];
        }
        indx[j]=imax;
        if (a[j][j] == 0.0) a[j][j]=TINY;
        if (j != n-1) {
            dum=(Real)1.0/(a[j][j]);
            for (i=j+1;i<n;i++) a[i][j] *= dum;
        }
    }
}

/// Compute the solution of Mx=b. b is replaced by x. a and indx together represent the LU decomposition of m, as given my method ludcmp.
template< int n, typename Real>
        void lubksb(const Mat<n,n,Real> &a, const Vec<n,int> &indx, Vec<n,Real> &b)
{
    int i,ii=0,ip,j;
    Real sum;

    for (i=0;i<n;i++) {
        ip=indx[i];
        sum=b[ip];
        b[ip]=b[i];
        if (ii != 0)
            for (j=ii-1;j<i;j++) sum -= a[i][j]*b[j];
        else if (sum != 0.0)
            ii=i+1;
        b[i]=sum;
    }
    for (i=n-1;i>=0;i--) {
        sum=b[i];
        for (j=i+1;j<n;j++) sum -= a[i][j]*b[j];
        b[i]=sum/a[i][i];
    }
}

// Compute the inverse of matrix m.
\warning Matrix m is replaced by its LU decomposition.

template< int n, typename Real>
        void luinv( Mat<n,n,Real> &inv, Mat<n,n,Real> &m )
{
    Vec<n,int> idx;
    Vec<n,Real> col;

    ludcmp(m,idx);

    for( int i=0; i<n; i++ ){
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
}*/


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
