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

#include <sofa/testing/config.h>

#include <sofa/testing/BaseTest.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/random.h>
#include <sofa/helper/cast.h>
#include <ctime>
#include <iostream>

namespace sofa::testing
{

/** @brief Helper functions to compare scalars, vectors, matrices, etc.
  */
template <typename _Real=SReal>
struct SOFA_TESTING_API NumericTest : public virtual BaseTest
{
    NumericTest() {}

    /** @name Scalars
     *  Type and functions to manipulate real numbers.
     */
    ///@{
    typedef _Real Real; ///< Scalar type

    /// the smallest real number
    static Real epsilon(){ return std::numeric_limits<Real>::epsilon(); }

    /// Infinity
    static Real infinity(){ return std::numeric_limits<Real>::infinity(); }

    /// true if the magnitude of r is less than ratio*epsilon
    static bool isSmall(Real r, Real ratio=1. ){
        return fabs(r) < ratio * std::numeric_limits<Real>::epsilon();
    }

    ///@}

    /** @name Vectors
     *  Functions to compare vectors
     */
    ///@{


    /// return the maximum difference between corresponding entries, or the infinity if the vectors have different sizes
    template< Size N, typename Real, typename Vector2>
    static Real vectorMaxDiff( const sofa::type::Vec<N,Real>& m1, const Vector2& m2 )
    {
        if( N !=m2.size() ) {
            ADD_FAILURE() << "Comparison between vectors of different sizes";
            return std::numeric_limits<Real>::infinity();
        }
        Real result = 0;
        for( unsigned i=0; i<N; i++ ){
            Real diff = (Real)fabs(m1[i] - m2.element(i));
            if( diff>result  ) result=diff;
        }
        return result;
    }


    /// return the maximum difference between corresponding entries
    template< Size N, typename Real>
    static Real vectorMaxDiff( const sofa::type::Vec<N,Real>& m1, const sofa::type::Vec<N,Real>& m2 )
    {
        Real result = 0;
        for( unsigned i=0; i<N; i++ ){
            Real diff = (Real)fabs(m1[i] - m2[i]);
            if( diff>result  ) result=diff;
        }
        return result;
    }

    /// Return the maximum difference between two containers. Issues a failure if sizes are different.
    template<class Container1, class Container2>
    Real vectorMaxDiff( const Container1& c1, const Container2& c2 )
    {
        if( c1.size()!=c2.size() ){
            ADD_FAILURE() << "containers have different sizes";
            return infinity();
        }

        Real maxdiff = 0.;
        for(unsigned i=0; i<(unsigned)c1.size(); i++ ){
//            cout<< c2[i]-c1[i] << " ";
            Real n = norm(c1[i]-c2[i]);
            if( n>maxdiff )
                maxdiff = n;
        }
        return maxdiff;
    }

    /// Return the maximum absolute value of a container
    template<class Container>
    Real vectorMaxAbs( const Container& c )
    {
        Real maxc = 0.;
        for(unsigned i=0; i<(unsigned)c.size(); i++ )
        {
            Real n = norm(c[i]);
            if( n>maxc )
                maxc = n;
        }
        return maxc;
    }

    ///@}

    /** @name Matrices
     *  Functions to compare matrices
     */
    ///@{

    /// return the maximum difference between corresponding entries, or the infinity if the matrices have different sizes
    template<typename Matrix1, typename Matrix2>
    static Real matrixMaxDiff( const Matrix1& m1, const Matrix2& m2 )
    {
        if (m1.rows() != m2.rows() || m2.cols() != m1.cols())
        {
            ADD_FAILURE() << "Comparison between matrices of different sizes";
            return infinity();
        }

        Real result = 0;
        for(typename Matrix1::Index i=0; i<m1.rows(); ++i)
        {
            for(typename Matrix1::Index j=0; j<m1.cols(); ++j)
            {
                const Real diff = std::fabs(m1(i,j) - m2(i,j));
                if(diff > result)
                    result = diff;
            }
        }
        return result;
    }

    /// Return the maximum difference between corresponding entries, or the infinity if the matrices have different sizes
    template<Size M, Size N, typename Real, typename Matrix2>
    static Real matrixMaxDiff( const sofa::type::Mat<M,N,Real>& m1, const Matrix2& m2 )
    {
        Real result = 0;
        if(M!=m2.rowSize() || m2.colSize()!=N){
            ADD_FAILURE() << "Comparison between matrices of different sizes";
            return std::numeric_limits<Real>::infinity();
        }
        for(unsigned i=0; i<M; i++)
            for(unsigned j=0; j<N; j++){
                Real diff = (Real)fabs(m1[i][j] - m2.element(i,j));
                if(diff > result)
                    result = diff;
            }
        return result;
    }

    ///@}

protected:
    // helpers
    static float norm(float a){ return std::abs(a); }
    static double norm(double a){ return std::abs(a); }

    template <typename T>
    static Real norm(T a){ return (Real)a.norm(); }


};

// Generic macro for comparing floating-point numbers
// Google Test provides EXPECT_FLOAT_EQ and EXPECT_DOUBLE_EQ, but it supposes that the type is known and constant. It
// cannot rely on a template mechanism for example.
#define EXPECT_FLOATINGPOINT_EQ(val1, val2) {\
    static_assert(std::is_same_v<std::decay_t<decltype(val1)>, std::decay_t<decltype(val2)> >,\
        "Different types for val1 and val2 are not supported");\
    static_assert(std::is_floating_point_v<std::decay_t<decltype(val1)> >,\
        "Non-floating-point types are not supported");\
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<std::decay_t<decltype(val1)> >, \
        val1, val2);}

// Generic macro for comparing floating-point numbers
// Google Test provides ASSERT_FLOAT_EQ and ASSERT_DOUBLE_EQ, but it supposes that the type is known and constant. It
// cannot rely on a template mechanism for example.
#define ASSERT_FLOATINGPOINT_EQ(val1, val2) {\
    static_assert(std::is_same_v<std::decay_t<decltype(val1)>, std::decay_t<decltype(val2)> >,\
        "Different types for val1 and val2 are not supported");\
    static_assert(std::is_floating_point_v<std::decay_t<decltype(val1)> >,\
        "Non-floating-point types are not supported");\
    ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<std::decay_t<decltype(val1)> >, \
        val1, val2);}


/// Resize the Vector and copy it from the Data
template<class Vector, class ReadData>
void copyFromData( Vector& v, const ReadData& d){
    v.resize(d.size());
    for(unsigned i=0; i<v.size(); i++)
        v[i] = d[i];
}

/// Copy the Vector to the Data. They must have the same size.
template<class WriteData, class Vector>
void copyToData( WriteData& d, const Vector& v){
    for( unsigned i=0; i<d.size(); i++)
        d[i] = v[i];
}

/** Helpers for DataTypes. Includes copies to and from vectors of scalars.
 *
 */
template<class _DataTypes>
struct data_traits
{
    typedef _DataTypes DataTypes;
    typedef sofa::Index Index;

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    /// Resize a vector of scalars, and copies a VecCoord in it
    template <class VectorOfScalars>
    inline static void VecCoord_to_Vector( VectorOfScalars& vec, const VecCoord& vcoord  )
    {
        vec.resize( vcoord.size() * coord_total_size );
        for( Index i=0; i<vcoord.size(); i++ ){
            for( Index j=0; j<coord_total_size; j++ )
                vec[j+coord_total_size*i] = vcoord[i][j];
        }
    }

    /// Resize a vector of scalars, and copies a VecDeriv in it
    template <class VectorOfScalars>
    inline static void VecDeriv_to_Vector( VectorOfScalars& vec, const VecDeriv vderiv  )
    {
        vec.resize( vderiv.size() * deriv_total_size );
        for( Index i=0; i<vderiv.size(); i++ ){
            for( Index j=0; j<deriv_total_size; j++ )
                vec[j+deriv_total_size*i] = vderiv[i][j];
        }
    }
};

// Do not use this class directly
template<class DataTypes, sofa::Size N, bool isVector>
struct setRotWrapper
{ static void setRot(typename DataTypes::Coord& coord, const sofa::type::Quat<SReal>& rot); };

template<class DataTypes, sofa::Size N>
struct setRotWrapper<DataTypes, N, true>
{ static void setRot(typename DataTypes::Coord& /*coord*/, const sofa::type::Quat<SReal>& /*rot*/) {} };

template<class DataTypes>
struct setRotWrapper<DataTypes, 2, false>
{ static void setRot(typename DataTypes::Coord& coord, const sofa::type::Quat<SReal>& rot)	{ coord.getOrientation() = rot.quatToRotationVector().z(); } };
// Use of quatToRotationVector instead of toEulerVector:
// this is done to keep the old behavior (before the
// correction of the toEulerVector  function). If the
// purpose was to obtain the Eulerian vector and not the
// rotation vector please use the following line instead
//{ static void setRot(typename DataTypes::Coord& coord, const sofa::type::Quat<SReal>& rot)	{ coord.getOrientation() = rot.toEulerVector().z(); } };

template<class DataTypes, sofa::Size N>
struct setRotWrapper<DataTypes, N, false>
{ static void setRot(typename DataTypes::Coord& coord, const sofa::type::Quat<SReal>& rot) 	{ DataTypes::setCRot(coord, rot); } };

template<class DataTypes>
void setRot(typename DataTypes::Coord& coord, const sofa::type::Quat<SReal>& rot)
{ setRotWrapper<DataTypes, DataTypes::Coord::spatial_dimensions, (unsigned)DataTypes::Coord::total_size == (unsigned)DataTypes::Coord::spatial_dimensions>::setRot(coord, rot); }

/// Create a coord of the specified type from a Vec3 and a Quater
template<class DataTypes>
typename DataTypes::Coord createCoord(const sofa::type::Vec3& pos, const sofa::type::Quat<SReal>& rot)
{
    typename DataTypes::Coord temp;
    DataTypes::set(temp, pos[0], pos[1], pos[2]);
    setRot<DataTypes>(temp, rot);
    return temp;
}

template <sofa::Size N, class real>
void EXPECT_VEC_DOUBLE_EQ(sofa::type::Vec<N, real> const& expected, sofa::type::Vec<N, real> const& actual) {
    for (sofa::Size i=0; i<expected.total_size; ++i)
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
}

template <sofa::Size L, sofa::Size C, class real>
void EXPECT_MAT_DOUBLE_EQ(sofa::type::Mat<L,C,real> const& expected, sofa::type::Mat<L,C,real> const& actual) {
    for (sofa::Size i=0; i<expected.nbLines; ++i)
        for (sofa::Size j=0; j<expected.nbCols; ++j)
            EXPECT_DOUBLE_EQ(expected(i,j), actual(i,j));
}

template <sofa::Size L, sofa::Size C, class real>
void EXPECT_MAT_NEAR(sofa::type::Mat<L,C,real> const& expected, sofa::type::Mat<L,C,real> const& actual, real abs_error) {
    for (sofa::Size i=0; i<expected.nbLines; ++i)
        for (sofa::Size j=0; j<expected.nbCols; ++j)
            EXPECT_NEAR(expected(i,j), actual(i,j), abs_error);
}

} // namespace sofa::testing


extern template struct SOFA_TESTING_API sofa::testing::NumericTest<float>;
extern template struct SOFA_TESTING_API sofa::testing::NumericTest<double>;
