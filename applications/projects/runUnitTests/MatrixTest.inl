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
MatrixTest<Real,RN,CN>::MatrixTest()
    :UnitTest("Operations on matrices")
{}


template<typename Real, int RN, int CN>
bool MatrixTest<Real,RN,CN>::succeeds()
{
    bool success = true;

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
    for( unsigned i=0; i<nrows; i++)
    {
        for( unsigned j=0; j<ncols; j++)
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
    if( !areEqual(A1,A2) ) success=false;
    if( !areEqual(A1,B) ) success=false;
    if( !areEqual(A1,C) ) success=false;
    if( !areEqual(A1,D) )
    {
        this->msg<< "MatrixTest fails, A1= " << endl << A1 << endl;
        this->msg<< "          D= " << endl << D << endl;
        success=false;
    }

    // =========== matrix-vector products: opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v)
    FullVector vref(nrows),v2(nrows);
//    cerr<<"MatrixTest: vref = Mat * V, Mat = "<< endl << A1 << "\n  vec = " << V << "\n  vref before = " << vref << endl;
    B.opMulV(&vref,&V);
//    cerr<<"MatrixTest: vref = " << vref << endl;
    C.opMulV(&v2,&V);
    if( !areEqual(vref,v2) )
    {
        this->msg<<"C.opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) failed"<< endl;
        this->msg<<" ref = " << vref << endl;
        this->msg<<" res = " << v2 << endl;
        success=false;
    }
    D.opMulV(&v2,&V);
    if( !areEqual(vref,v2) )
    {
        this->msg<<"D.opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) failed"<< endl;
        this->msg<<" ref = " << vref << endl;
        this->msg<<" res = " << v2 << endl;
        success=false;
    }
    log("Skipping test of CRSMatrix::opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) as it crashes. Uncomment it in MatrixTest.inl to reproduce the crash.");
//    cerr<<"now A2………"<<endl;
//    A2.opMulV(&v2,&V);
//    if( !areEqual(vref,v2) ){
//        this->msg<<"A2.opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) failed"<< endl;
//        this->msg<<" ref = " << vref << endl;
//        this->msg<<" res = " << v2 << endl;
//        success=false;
//    }
//    cerr<<"now A1………"<<endl;
//    A1.opMulV(&vref,&V);





    return success;
}




}
}


