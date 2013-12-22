/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_STANDARDTEST_Sofa_test_H
#define SOFA_STANDARDTEST_Sofa_test_H



#include <gtest/gtest.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/simulation/common/Node.h>
#include <time.h>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

namespace sofa {

/** Base class for all Sofa test fixtures, to provide helper functions to compare vectors, matrices, etc.
  */
template <typename _Real>
struct Sofa_test : public ::testing::Test
{
    typedef _Real Real; ///< Scalar type

    static Real epsilon(){ return std::numeric_limits<Real>::epsilon(); }
    static Real infinity(){ return std::numeric_limits<Real>::infinity(); }

    Sofa_test()
    {
        srand (time(NULL)); // comment out if you want to generate always the same sequence of pseudo-random numbers
    }

    /// true if the magnitude of r is less than ratio*numerical precision
    static bool isSmall(Real r, Real factor=1. ){
        return fabs(r) < factor * std::numeric_limits<Real>::epsilon();
    }

    /// return the maximum difference between corresponding entries, or the infinity if the matrices have different sizes
    template<typename Matrix1, typename Matrix2>
    static Real matrixCompare( const Matrix1& m1, const Matrix2& m2 )
    {
        Real result = 0;
        if(m1.rowSize()!=m2.rowSize() || m2.colSize()!=m1.colSize()){
            ADD_FAILURE() << "Comparison between matrices of different sizes";
            return infinity();
        }
        for(unsigned i=0; i<m1.rowSize(); i++)
            for(unsigned j=0; j<m1.colSize(); j++){
                Real diff = abs(m1.element(i,j)-m2.element(i,j));
                if(diff>result)
                    result = diff;
            }
        return result;
    }

    /// Return the maximum difference between corresponding entries, or the infinity if the matrices have different sizes
    template<int M, int N, typename Real, typename Matrix2>
    static Real matrixCompare( const sofa::defaulttype::Mat<M,N,Real>& m1, const Matrix2& m2 )
    {
        Real result = 0;
        if(M!=m2.rowSize() || m2.colSize()!=N){
            ADD_FAILURE() << "Comparison between matrices of different sizes";
            return std::numeric_limits<Real>::infinity();
        }
        for(unsigned i=0; i<M; i++)
            for(unsigned j=0; j<N; j++){
                Real diff = abs(m1.element(i,j)-m2.element(i,j));
                if(diff>result)
                    result = diff;
            }
        return result;
    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    template<typename Matrix1, typename Matrix2>
    static bool matricesAreEqual( const Matrix1& m1, const Matrix2& m2, double tolerance=std::numeric_limits<Real>::epsilon()*100)
    {
        bool result = true;
        if(m1.rows()!=m2.rows() || m2.cols()!=m1.cols()) result = false;
        for(unsigned i=0; i<m1.rows(); i++)
            for(unsigned j=0; j<m1.cols(); j++)
                if(abs(m1(i,j)-m2(i,j))>tolerance) {
                    cout<<"SofaTest::matricesAreEqual1 is false, difference = "<< abs(m1.element(i,j)-m2.element(i,j)) << " is larger than " << tolerance << endl;
                    result = false;
                }

        if( result == false ){
            cout<<"SofaTest::matricesAreEqual1 is false, matrix 1 = "<< m1 <<endl;
            cout<<"SofaTest::matricesAreEqual1, matrix 2 = "<< m2 <<endl;
        }
        return result;
    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    template<int M, int N, typename Real, typename Matrix2>
    static bool matricesAreEqual( const sofa::defaulttype::Mat<M,N,Real>& m1, const Matrix2& m2, double tolerance=std::numeric_limits<Real>::epsilon()*100 )
    {
        bool result = true;
        if(M!=m2.rows() || N!=m2.cols()) result= false;
        for( unsigned i=0; i<M; i++ )
            for( unsigned j=0; j<N; j++ )
                if( fabs(m1(i,j)-m2(i,j))>tolerance  ) {
                    cout<<"SofaTest::matricesAreEqual2 is false, difference = "<< fabs(m1(i,j)-m2(i,j)) << " is larger than " << tolerance << endl;
                    result= false;
                }

        if( result == false ){
            cout<<"SofaTest::matricesAreEqual2 is false, matrix 1 = "<< m1 <<endl;
            cout<<"SofaTest::matricesAreEqual2 is false, matrix 2 = "<< m2 <<endl;
        }
        return result;
    }


    /// return true if the vectors have same size and all their entries are equal within the given tolerance
    template< typename Vector1, typename Vector2>
    static bool vectorsAreEqual( const Vector1& m1, const Vector2& m2, double tolerance=std::numeric_limits<double>::epsilon() )
    {
        if( m1.size()!=m2.size() ) {
            ADD_FAILURE() << "Comparison between vectors of different sizes";
            return false;
        }
        for( unsigned i=0; i<m1.size(); i++ )
            if( fabs(m1.element(i)-m2.element(i))>tolerance  ) return false;
        return true;
    }


    /// return the maximum difference between corresponding entries, or the infinity if the vectors have different sizes
    template< typename Vector1, typename Vector2>
    static Real vectorCompare( const Vector1& m1, const Vector2& m2 )
    {
        if( m1.size()!=m2.size() ) {
            ADD_FAILURE() << "Comparison between vectors of different sizes";
            return std::numeric_limits<Real>::infinity();
        }
        Real result = 0;
        for( unsigned i=0; i<m1.size(); i++ ){
            Real diff = fabs(m1.element(i)-m2.element(i));
            if( diff>result  ) result=diff;
        }
        return result;
    }


    /// return true if the vectors have same size and all their entries are equal within the given tolerance
    template< int N, typename Real, typename Vector2>
    static bool vectorsAreEqual( const sofa::defaulttype::Vec<N,Real>& m1, const Vector2& m2, double tolerance=std::numeric_limits<double>::epsilon() )
    {
        if( N!=m2.size() ) return false;
        for( unsigned i=0; i<N; i++ )
            if( fabs(m1[i]-m2.element(i))>tolerance  ) return false;
        return true;
    }

    /// return the maximum difference between corresponding entries, or the infinity if the vectors have different sizes
    template< int N, typename Real, typename Vector2>
    static Real vectorCompare( const sofa::defaulttype::Vec<N,Real>& m1, const Vector2& m2 )
    {
        if( N !=m2.size() ) {
            ADD_FAILURE() << "Comparison between vectors of different sizes";
            return std::numeric_limits<Real>::infinity();
        }
        Real result = 0;
        for( unsigned i=0; i<N; i++ ){
            Real diff = fabs(m1.element(i)-m2.element(i));
            if( diff>result  ) result=diff;
        }
        return result;
    }


    /// return the maximum difference between corresponding entries, or the infinity if the vectors have different sizes
    template< int N, typename Real>
    static Real vectorCompare( const sofa::defaulttype::Vec<N,Real>& m1, const sofa::defaulttype::Vec<N,Real>& m2 )
    {
        Real result = 0;
        for( unsigned i=0; i<N; i++ ){
            Real diff = fabs(m1[i]-m2[i]);
            if( diff>result  ) result=diff;
        }
        return result;
    }

    /// Return the maximum difference between two containers. Issues a failure if sizes are different.
    template<class Container1, class Container2>
    Real maxDiff( const Container1& c1, const Container2& c2 )
    {
        if( c1.size()!=c2.size() ){
            ADD_FAILURE() << "containers have different sizes";
            return this->infinity();
        }

        Real maxdiff = 0.;
        for(unsigned i=0; i<c1.size(); i++ ){
//            cout<< c2[i]-c1[i] << " ";
            Real n = (c1[i]-c2[i]).norm();
            if( n>maxdiff )
                maxdiff = n;
        }
        return maxdiff;
    }



};

/// helper for more compact component creation
template<class Component>
typename Component::SPtr addNew( sofa::simulation::Node::SPtr parentNode, const char* name="component" )
{
    typename Component::SPtr component = sofa::core::objectmodel::New<Component>();
    parentNode->addObject(component);
    component->setName(parentNode->getName()+"_"+std::string(name));
    return component;
}

/// Resize the Vector and copy it from the Data
template<class Vector, class ReadData>
void copyFromData( Vector& v, const ReadData& d){
    v.resize(d.size());
    for( unsigned i=0; i<v.size(); i++)
        v[i] = d[i];
}

/// Copy the Vector to the Data. They must have the same size.
template<class WriteData, class Vector>
void copyToData( WriteData& d, const Vector& v){
    for( unsigned i=0; i<d.size(); i++)
        d[i] = v[i];
}



}

#endif




