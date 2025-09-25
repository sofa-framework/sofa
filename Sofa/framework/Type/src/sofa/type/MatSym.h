/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>

#include <cassert>
#include <iostream>



namespace sofa::type
{

/**
 * Dense symmetric matrix of size DxD storing only D*(D+1)/2 values
 * \tparam D Size of the matrix
 * \tparam real Type of scalar
 */
template <sofa::Size D, class real = SReal>
class MatSym : public VecNoInit<D * (D + 1) / 2, real>
{
public:

    typedef real Real;
    typedef Vec<D,Real> Coord;
    static constexpr auto NumberStoredValues = D * (D + 1) / 2;

    constexpr MatSym() noexcept
    {
        clear();
    }

    constexpr explicit MatSym(NoInit) noexcept
    {
    }

    /// Constructor from 6 elements
    template<sofa::Size TD = D, typename = std::enable_if_t<TD == 3> >
    constexpr MatSym(
        const real& v1, const real& v2, const real& v3,
        const real& v4, const real& v5, const real& v6)
    {
        this->elems[0] = v1;
        this->elems[1] = v2;
        this->elems[2] = v3;
        this->elems[3] = v4;
        this->elems[4] = v5;
        this->elems[5] = v6;
    }

    /// Constructor from an element
    constexpr MatSym(const sofa::Size sizeM, const real& v)
    {
        assert(sizeM <= D);
        for (sofa::Size i = 0; i < sizeM * (sizeM + 1) / 2; ++i)
        {
            this->elems[i] = v;
        }
    }

    /// Constructor from another matrix
    template<typename real2>
    explicit MatSym(const MatSym<D, real2>& m)
    {
        std::copy(m.begin(), m.begin() + NumberStoredValues, this->begin());
    }


    /// Assignment from another matrix
    template<typename real2>
    void operator=(const MatSym<D, real2>& m)
    {
        std::copy(m.begin(), m.begin() + NumberStoredValues, this->begin());
    }

    /// Sets each element to 0.
    void clear()
    {
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            this->elems[i] = 0;
        }
    }

    /// Sets each element to r.
    void fill(real r)
    {
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            this->elems[i].fill(r);
        }
    }

    /// Write access to element (i,j).
    inline real& operator()(const int i, const int j)
    {
        if (i >= j)
        {
            return this->elems[(i * (i + 1)) / 2 + j];
        }
        return this->elems[(j * (j + 1)) / 2 + i];
    }

    /// Read-only access to element (i,j).
    inline const real& operator()(const int i, const int j) const
    {
        if (i >= j)
        {
            return this->elems[(i * (i + 1)) / 2 + j];
        }
        return this->elems[(j * (j + 1)) / 2 + i];
    }

    /// convert matrix to sym
    static void Mat2Sym( const Mat<D, D, real>& M, MatSym<D, real>& W)
    {
        for (sofa::Size j = 0; j < D; j++)
        {
            for (sofa::Size i = 0; i <= j; i++)
            {
                W(i, j) = (M(i, j) + M(j, i)) / 2;
            }
        }
    }

    /// convert to Voigt notation (supported only for D == 2 and D == 3)
    template<sofa::Size TD = D, typename = std::enable_if_t<TD == 3 || TD == 2> >
    inline Vec<NumberStoredValues, real> getVoigt() const
    {
        Vec<NumberStoredValues, real> result {NOINIT};
        if constexpr (D == 2)
        {
            result[0] = this->elems[0];
            result[1] = this->elems[2];
            result[2] = 2 * this->elems[1];
        }
        else if constexpr (D == 3)
        {
            result[0] = this->elems[0];
            result[1] = this->elems[2];
            result[2] = this->elems[5];
            result[3] = 2 * this->elems[4];
            result[4] = 2 * this->elems[3];
            result[5] = 2 * this->elems[1];
        }
        return result;
    }

    /// Set matrix to identity.
    constexpr void identity()
    {
        for (sofa::Size i = 0; i < D; i++)
        {
            this->elems[i * (i + 1) / 2 + i] = 1;
            for (sofa::Size j = i + 1; j < D; j++)
            {
                this->elems[i * (i + 1) / 2 + j] = 0;
            }
        }
    }

    /// @name Tests operators
    /// @{

    bool operator==(const MatSym<D, real>& b) const
    {
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            if (this->elems[i] != b[i])
            {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const MatSym<D, real>& b) const
    {
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            if (this->elems[i] != b[i])
            {
                return true;
            }
        }
        return false;
    }

    /// @}

    // LINEAR ALGEBRA

    /// Matrix multiplication operator: product of two symmetric matrices
    [[nodiscard]] Mat<D, D, real> SymSymMultiply(const MatSym<D, real>& m) const
    {
        Mat<D, D, real> r(NOINIT);

        for (sofa::Size i = 0; i < D; i++)
        {
            for (sofa::Size j = 0; j < D; j++)
            {
                r(i,j) = (*this)(i, 0) * m(0, j);
                for (sofa::Size k = 1; k < D; k++)
                {
                    r(i,j) += (*this)(i, k) * m(k, j);
                }
            }
        }
        return r;
    }

    Mat<D, D, real> operator*(const MatSym<D, real>& m) const
    {
        return SymSymMultiply(m);
    }

    //Multiplication by a non symmetric matrix on the right
    [[nodiscard]] Mat<D,D,real> SymMatMultiply(const Mat<D,D,real>& m) const
    {
        Mat<D,D,real> r(NOINIT);

        for (sofa::Size i = 0; i < D; i++)
        {
            for (sofa::Size j = 0; j < D; j++)
            {
                r(i,j) = (*this)(i, 0) * m(0,j);
                for (sofa::Size k = 1; k < D; k++)
                {
                    r(i,j) += (*this)(i, k) * m(k,j);
                }
            }
        }
        return r;
    }

    Mat<D,D,real> operator*(const Mat<D,D,real>& m) const
    {
        return SymMatMultiply(m);
    }

    //Multiplication by a non symmetric matrix on the left
    Mat<D, D, real> MatSymMultiply(const Mat<D, D, real>& m) const
    {
        Mat<D, D, real> r(NOINIT);

        for (sofa::Size i = 0; i < D; i++)
        {
            for (sofa::Size j = 0; j < D; j++)
            {
                r(i,j) = m(i, 0) * (*this)(0, j);
                for (sofa::Size k = 1; k < D; k++)
                {
                    r(i,j) += m(i, k) * (*this)(k, j);
                }
            }
        }
        return r;
    }


    /// Matrix addition operator with a symmetric matrix
    MatSym<D, real> operator+(const MatSym<D, real>& m) const
    {
        MatSym<D, real> r(NOINIT);
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            r[i] = (*this)[i] + m[i];
        }
        return r;
    }

    /// Matrix addition operator with a non-symmetric matrix
    Mat<D, D, real> operator+(const Mat<D, D, real>& m) const
    {
        Mat<D, D, real> r(NOINIT);
        for (sofa::Size i = 0; i < D; i++)
        {
            for (sofa::Size j = 0; j < D; j++)
            {
                r(i,j) = (*this)(i, j) + m(i,j);
            }
        }
        return r;
    }

    /// Matrix substractor operator with a symmetric matrix
    MatSym<D, real> operator-(const MatSym<D, real>& m) const
    {
        MatSym<D, real> r(NOINIT);
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            r[i] = (*this)[i] - m[i];
        }
        return r;
    }

    /// Matrix substractor operator with a non-symmetric matrix
    Mat<D, D, real> operator-(const Mat<D, D, real>& m) const
    {
        Mat<D, D, real> r(NOINIT);
        for (sofa::Size i = 0; i < D; i++)
        {
            for (sofa::Size j = 0; j < D; j++)
            {
                r(i,j) = (*this)(i, j) - m(i,j);
            }
        }
        return r;
    }

    /// Multiplication operator Matrix * Vector.
    Coord operator*(const Coord& v) const
    {
        Coord r(NOINIT);
        for (sofa::Size i = 0; i < D; i++)
        {
            r[i] = (*this)(i, 0) * v[0];
            for (sofa::Size j = 1; j < D; j++)
            {
                r[i] += (*this)(i, j) * v[j];
            }
        }
        return r;
    }

    /// Scalar multiplication operator.
    MatSym<D, real> operator*(real f) const
    {
        MatSym<D, real> r(NOINIT);
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            r[i] = (*this)[i] * f;
        }
        return r;
    }

    /// Scalar matrix multiplication operator.
    friend MatSym<D, real> operator*(real r, const MatSym<D, real>& m)
    {
        return m * r;
    }

    /// Scalar division operator.
    MatSym<D, real> operator/(real f) const
    {
        MatSym<D, real> r(NOINIT);
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            r[i] = (*this)[i] / f;
        }
        return r;
    }

    /// Scalar multiplication assignment operator.
    void operator *=(real r)
    {
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            this->elems[i] *= r;
        }
    }

    /// Scalar division assignment operator.
    void operator /=(real r)
    {
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            this->elems[i] /= r;
        }
    }

    /// Addition assignment operator.
    void operator +=(const MatSym< D,real>& m)
    {
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            this->elems[i] += m[i];
        }
    }

    /// Subtraction assignment operator.
    void operator -=(const MatSym< D,real>& m)
    {
        for (sofa::Size i = 0; i < NumberStoredValues; i++)
        {
            this->elems[i] -= m[i];
        }
    }

    /// Invert matrix m
    bool invert(const MatSym<D,real>& m)
    {
        return invertMatrix((*this), m);
    }
};

template <sofa::Size D, class real>
Mat<D, D, real> operator*(const Mat<D, D, real>& a, const MatSym<D, real>& b)
{
    return b.MatSymMultiply(a);
}

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
template <class real>
inline real determinant(const MatSym<2, real>& m)
{
    return m(0, 0) * m(1, 1) - m(0, 1) * m(0, 1);
}

#define MIN_DETERMINANT  1.0e-100


/// Trace of a 3x3 matrix.
template <class real>
inline real trace(const MatSym<3, real>& m)
{
    return m(0, 0) + m(1, 1) + m(2, 2);
}

/// Trace of a 2x2 matrix.
template <class real>
inline real trace(const MatSym<2, real>& m)
{
    return m(0, 0) + m(1, 1);
}

/// Matrix inversion (general case).
template<sofa::Size S, class real>
bool invertMatrix(MatSym<S,real>& dest, const MatSym<S,real>& from)
{
    sofa::Size i, j, k;
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
template <class real>
bool invertMatrix(MatSym<3, real>& dest, const MatSym<3, real>& from)
{
    real det=determinant(from);

    if ( -(real) MIN_DETERMINANT<=det && det<=(real) MIN_DETERMINANT)
    {
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
        return false;
    }

    dest(0,0)=  from(1,1)/det;
    dest(0,1)= -from(0,1)/det;
    //dest(1,0)= -from(1,0)/det;
    dest(1,1)=  from(0,0)/det;

    return true;
}
#undef MIN_DETERMINANT

template<sofa::Size D,class real>
std::ostream& operator<<(std::ostream& o, const MatSym<D,real>& m)
{
    o << '[' ;
    for(sofa::Size i=0; i<D; i++)
    {
        for(sofa::Size j=0; j<D; j++)
        {
            o<<" "<<m(i,j);
        }
        o<<" ,";
    }
    o << ']';
    return o;
}

template<sofa::Size D,class real>
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
    for(sofa::Size i=0; i<D; i++)
    {
        c = in.peek();
        while (c==' ' || c==',')
        {
            in.get(); c = in.peek();
        }

        for(sofa::Size j=0; j<D; j++)
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
template <sofa::Size D, typename real>
inline real scalarProduct(const MatSym<D,real>& left, const MatSym<D,real>& right)
{
    real sympart(0.),dialpart(0.);
    for(sofa::Size i=0; i<D; i++)
        for(sofa::Size j=i+1; j<D; j++)
            sympart += left(i,j) * right(i,j);

    for(sofa::Size d=0; d<D; d++)
        dialpart += left(d,d) * right(d,d);


    return 2. * sympart  + dialpart ;
}

template <sofa::Size D, typename real>
inline real scalarProduct(const MatSym<D,real>& left, const Mat<D,D,real>& right)
{
    real product(0.);
    for(sofa::Size i=0; i<D; i++)
        for(sofa::Size j=0; j<D; j++)
            product += left(i,j) * right(i,j);
    return product;
}

template <sofa::Size D, typename real>
inline real scalarProduct(const Mat<D,D,real>& left, const MatSym<D,real>& right)
{
    return scalarProduct(right, left);
}

} // namespace sofa::type
