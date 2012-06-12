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


template<typename Real, int RN, int CN>
MatrixTest<Real,RN,CN>::MatrixTest(std::string name, sofa::helper::UnitTest::VerbosityLevel vlevel, unsigned nrows, unsigned ncols)
    :UnitTest(name, vlevel)
    , nrows(nrows), ncols(ncols)
{}


/** Check that EigenMatrix update works as well as direct init. Return true if the test succeeds.*/
template<typename Real, int RN, int CN>
bool MatrixTest<Real,RN,CN>::checkEigenMatrixUpdate()
{
    // fill two matrices with the same values, one directly, one it two passes, then compare their values
    EigenBlockMatrix a,b;
    a.resize(nrows,ncols);
    b.resize(nrows,ncols);
    for( unsigned j=0; j<ncols; j++)
    {
        for( unsigned i=0; i<nrows; i++)
        {
            double valij = i*ncols+j;
            a.add(i,j,valij);
            if( i==j )
                b.add(i,j,valij);
        }
    }
    a.compress();
    b.compress();

//    cerr<<"MatrixTest<Real,RN,CN>::checkEigenMatrixUpdate, a = " << a << endl;
//    cerr<<"MatrixTest<Real,RN,CN>::checkEigenMatrixUpdate, b incomplete = " << b << endl;

    // second pass for b. Some values are set with the right value, some with the wrong value, some are not set
    for( unsigned j=0; j<ncols; j++)
    {
        for( unsigned i=0; i<nrows; i++)
        {
            double valij = i*ncols+j;
            if( i!=j )
                b.add(i,j,valij);
        }
    }
    b.compress();
//    cerr<<"MatrixTest<Real,RN,CN>::checkEigenMatrixUpdate, b complete = " << b << endl;
    return matricesAreEqual(a,b);
}

template<typename Real, int RN, int CN>
void MatrixTest<Real,RN,CN>::runTests( unsigned& numTests, unsigned& numWarnings, unsigned& numErrors )
{
    // Compares results of operations performed using different types of matrices.
    // The reference is the result obtained using a dense matrix

    CRSMatrix A1,A2;
    FullMatrix B;
    MapMatrix C;
    EigenBlockMatrix D1,D2;
    EigenBaseMatrix E;
    FullVector V;
    A1.resize(nrows,ncols);
    A2.resize(nrows,ncols);
    B.resize(nrows,ncols);
    C.resize(nrows,ncols);
    D1.resize(nrows,ncols);
    D2.resize(nrows,ncols);
    E.resize(nrows,ncols);
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
            D1.add(i,j,valij);
            E.add(i,j,valij);
            Block& bb = D2.wBlock(i/BROWS,j/BCOLS);
            bb[i%BROWS][j%BCOLS] = valij;
            V[j] = j;
        }
    }
    A1.compress(); A2.compress(); D1.compress(); D2.compress(); E.compress();
//    cerr<<"B = "<< B << endl;
//    cerr<<"D1 = "<< D1 << endl;
//    cerr<<"D2 = "<< D2 << endl;
//    cerr<<"E = "<< E << endl;
    checkIf(matricesAreEqual(B,A1),"A1==B",numTests,numErrors);
    checkIf(matricesAreEqual(B,A2),"A2==B", numTests,numErrors);
    checkIf(matricesAreEqual(B,C), "C==B", numTests,numErrors);
    checkIf(matricesAreEqual(B,D1), "D1==B", numTests,numErrors);
    checkIf(matricesAreEqual(B,D2), "D2==B", numTests,numErrors);
    checkIf(matricesAreEqual(B,E), "E==B", numTests,numErrors);
    checkIf( checkEigenMatrixUpdate(), "EigenMatrix update", numTests,numErrors);


    // =========== matrix-vector products: opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v)
    FullVector vref(nrows),v2(nrows);
    //    cerr<<"MatrixTest: vref = Mat * V, Mat = "<< endl << A1 << "\n  vec = " << V << "\n  vref before = " << vref << endl;
    B.opMulV(&vref,&V);
    //    cerr<<"MatrixTest: vref = " << vref << endl;
    C.opMulV(&v2,&V);
    checkIf(vectorsAreEqual(vref,v2),"matrix-vector product C*v", numTests,numErrors);
    D1.opMulV(&v2,&V);
    checkIf(vectorsAreEqual(vref,v2),"matrix-vector product D*v", numTests,numErrors);

    if( nrows%BROWS==0 && ncols%BCOLS==0)
    {
        A1.opMulV(&vref,&V);
        checkIf( vectorsAreEqual(vref,v2),"matrix-vector product A1*v",  numTests,numErrors);
        A2.opMulV(&v2,&V);
        checkIf( vectorsAreEqual(vref,v2),"matrix-vector product A2*v",  numTests,numErrors);
    }
    else
    {
        serr()<<"nrows = " << nrows << ", BROWS = " << BROWS <<  ", nrows%BROWS = " << nrows%BROWS << ", ncols = " << ncols << ", BCOLS = " << BCOLS << ", ncols%BCOLS = " << ncols%BCOLS << endl;
        serr()<<"Skipping test of CRSMatrix::opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v) when matrix size is not a multiple of block size, as it crashes. Edit MatrixTest.inl to reproduce the crash."<<endl;
        numWarnings++;
    }

    // =========== matrix-matrix products: opMulV(defaulttype::BaseVector* result, const defaulttype::BaseVector* v)




}




}
}


