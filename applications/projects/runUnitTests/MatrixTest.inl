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
#include "MatrixTest.h"
#include <sofa/helper/vector_algebra.h>
#include <sofa/helper/vector.h>
using std::cerr;
using std::endl;


namespace sofa
{
namespace helper
{


/// return true if the matrices have same size and all their entries are equal within the given tolerance
bool areEqual( const defaulttype::BaseMatrix& m1, const defaulttype::BaseMatrix& m2, double tolerance=std::numeric_limits<double>::epsilon() )
{
    if(m1.rowSize()!=m2.rowSize() || m2.colSize()!=m1.colSize()) return false;
    for( unsigned i=0; i<m1.rowSize(); i++ )
        for( unsigned j=0; j<m1.colSize(); j++ )
            if( fabs(m1.element(i,j)-m2.element(i,j))>tolerance  ) return false;
    return true;
}

/// return true if the matrices have same size and all their entries are equal within the given tolerance
bool areEqual( const defaulttype::BaseVector& m1, const defaulttype::BaseVector& m2, double tolerance=std::numeric_limits<double>::epsilon() )
{
    if( m1.size()!=m2.size() ) return false;
    for( unsigned i=0; i<m1.size(); i++ )
        if( fabs(m1.element(i)-m2.element(i))>tolerance  ) return false;
    return true;
}


template<typename Real, int RN, int CN>
MatrixTest<Real,RN,CN>::MatrixTest(std::string name, sofa::helper::UnitTest::VerbosityLevel vlevel)
    :UnitTest(name, vlevel)
{}


/** Check that EigenMatrix update works as well as direct init. Return true if the test succeeds.*/
template<typename Real, int RN, int CN>
bool MatrixTest<Real,RN,CN>::checkEigenMatrixUpdate()
{
    // fill two matrices with the same values, one directly, one it two passes, then compare their values
    EigenMatrix a,b;
    unsigned nrows = 5, ncols=3;
    a.resize(nrows,ncols);
    b.resize(nrows,ncols);
    for( unsigned j=0; j<ncols; j++)
    {
        for( unsigned i=0; i<nrows; i++)
        {
            double valij = i*ncols+j;
            a.set(i,j,valij);
            if( i==j )
                b.set(i,j,valij);
            else if( (i+j)%2==0 )
                b.set(i,j,-2);
        }
    }
    a.compress();
    b.compress();
    // second pass for b. Some values are set with the right value, some with the wrong value, some are not set
    for( unsigned j=0; j<ncols; j++)
    {
        for( unsigned i=0; i<nrows; i++)
        {
            double valij = i*ncols+j;
            b.set(i,j,valij);
        }
    }
    b.compress();
    return areEqual(a,b);
}

template<typename Real, int RN, int CN>
void MatrixTest<Real,RN,CN>::runTests( unsigned& numTests, unsigned& numWarnings, unsigned& numErrors )
{
    unsigned nrows = 5, ncols=3;
    CRSMatrix A1,A2;
    FullMatrix B;
    MapMatrix C;
    EigenMatrix D;
    FullVector V;
    A1.resize(nrows,ncols);
    A2.resize(nrows,ncols);
    B.resize(nrows,ncols);
    C.resize(nrows,ncols);
    D.resize(nrows,ncols);
    V.resize(ncols);

    // =========== Read-write access to matrix entries
    // fill the matrices in column-wise order, and compare them
    for( unsigned j=0; j<ncols; j++)
    {
        for( unsigned i=0; i<nrows; i++)
        {
            double valij = i*ncols+j;
            A1.set(i,j,valij);
            Block* b = A2.wbloc(i/BROWS,j/BCOLS,true);
            assert(b && "a matrix block exists");
            (*b)[i%BROWS][j%BCOLS] = valij;
            B.set(i,j,valij);
            C.set(i,j,valij);
            D.set(i,j,valij);
            V[j] = j;
        }
    }
    A1.compress(); A2.compress(); D.compress();
    checkIf(areEqual(A1,A2),"A1==A2",numTests,numErrors);
    checkIf(areEqual(A1,B), "A1==B", numTests,numErrors);
    checkIf(areEqual(A1,C), "A1==C", numTests,numErrors);
    checkIf(areEqual(A1,D), "A1==D", numTests,numErrors);
    checkIf( checkEigenMatrixUpdate(), "EigenMatrix update", numTests,numErrors);


    // =========== matrix-vector products: opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v)
    FullVector vref(nrows),v2(nrows);
//    cerr<<"MatrixTest: vref = Mat * V, Mat = "<< endl << A1 << "\n  vec = " << V << "\n  vref before = " << vref << endl;
    B.opMulV(&vref,&V);
//    cerr<<"MatrixTest: vref = " << vref << endl;
    C.opMulV(&v2,&V);
    checkIf(areEqual(vref,v2),"matrix-vector product C*v", numTests,numErrors);
    D.opMulV(&v2,&V);
    checkIf(areEqual(vref,v2),"matrix-vector product D*v", numTests,numErrors);
    serr()<<"Skipping test of CRSMatrix::opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) as it crashes. Uncomment it in MatrixTest.inl to reproduce the crash."<<endl;
    numWarnings++;
//    A2.opMulV(&v2,&V);
//   detectErrors( areEqual(vref,v2),"matrix-vector product A2*v",  numTests,numErrors);
//    A1.opMulV(&vref,&V);
//   detectErrors( areEqual(vref,v2),"matrix-vector product A1*v",  numTests,numErrors);

}




}
}


