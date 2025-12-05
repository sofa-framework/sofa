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
#ifndef SOFA_DEFAULTTYPE_MatEigen_H
#define SOFA_DEFAULTTYPE_MatEigen_H

/** Helpers to apply Eigen matrix methods to the Mat sofa type */

#include <sofa/type/Mat.h>
#include <Eigen/Dense>
#include <iostream>


namespace sofa::helper
{

template <Size NumRows, Size NumCols, class Real>
Eigen::Matrix<Real, NumRows, NumCols> eigenMat( const type::Mat< NumRows, NumCols, Real>& mat )
{
    Eigen::Matrix<Real, NumRows, NumCols> emat;
    for(Size i=0; i<NumRows; i++)
        for(Size j=0; j<NumCols; j++)
            emat(i,j) = mat(i,j);
    return emat;
}

template <Size NumRows, Size NumCols, class Real>
type::Mat<NumRows, NumCols, Real>  sofaMat( const Eigen::Matrix<Real, NumRows, NumCols>& emat )
{
    type::Mat<NumRows, NumCols, Real> mat;
    for(Size i=0; i<NumRows; i++)
        for(Size j=0; j<NumCols; j++)
            mat(i,j) = emat(i,j);
    return mat;
}

template <Size NumRows, class Real>
type::Vec<NumRows, Real>  sofaVec( const Eigen::Matrix<Real, NumRows, 1>& evec )
{
    type::Vec<NumRows, Real> vec;
    for(Size i=0; i<NumRows; i++)
        vec[i] = evec(i);
    return vec;
}

template <Size NumRows, class Real>
Eigen::Matrix<Real, NumRows, 1>  eigenVec( const type::Vec<NumRows, Real>& vec )
{
    Eigen::Matrix<Real, NumRows, 1> evec;
    for(Size i=0; i<NumRows; i++)
        evec(i)  = vec[i];
    return evec;
}

}

#endif
