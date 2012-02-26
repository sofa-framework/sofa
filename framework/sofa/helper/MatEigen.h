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
#ifdef SOFA_HAVE_EIGEN2
#ifndef SOFA_DEFAULTTYPE_MatEigen_H
#define SOFA_DEFAULTTYPE_MatEigen_H

/** Helpers to apply Eigen matrix methods to the Mat sofa type */

#include <sofa/defaulttype/Mat.h>
#include <Eigen/Dense>
#include <iostream>


namespace sofa
{

namespace helper
{
using std::cerr;
using std::endl;


template <int NumRows, int NumCols, class Real>
Eigen::Matrix<Real, NumRows, NumCols> eigenMat( const Mat< NumRows, NumCols, Real>& mat )
{
    Eigen::Matrix<Real, NumRows, NumCols> emat;
    for(int i=0; i<NumRows; i++)
        for(int j=0; j<NumCols; j++)
            emat(i,j) = mat[i][j];
    return emat;
}

template <int NumRows, int NumCols, class Real>
Mat<NumRows, NumCols, Real>  sofaMat( const Eigen::Matrix<Real, NumRows, NumCols>& emat )
{
    Mat<NumRows, NumCols, Real> mat;
    for(int i=0; i<NumRows; i++)
        for(int j=0; j<NumCols; j++)
            mat[i][j] = emat(i,j);
    return mat;
}

template <int NumRows, class Real>
Vec<NumRows, Real>  sofaVec( const Eigen::Matrix<Real, NumRows, 1>& evec )
{
    Vec<NumRows, Real> vec;
    for(int i=0; i<NumRows; i++)
        vec[i] = evec(i);
    return vec;
}

template <int NumRows, class Real>
Eigen::Matrix<Real, NumRows, 1>  eigenVec( const Vec<NumRows, Real>& vec )
{
    Eigen::Matrix<Real, NumRows, 1> evec;
    for(int i=0; i<NumRows; i++)
        evec(i)  = vec[i];
    return evec;
}



} // namespace defaulttype

} // namespace sofa

// iostream

#endif
#endif
