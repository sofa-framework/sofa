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

template <Size D, class Real>
Eigen::Matrix<Real, D, D> eigenMat( const type::MatSym<D, Real>& mat )
{
    Eigen::Matrix<Real, D, D> emat;
    for (Size i = 0; i < D; i++)
    {
        for (Size j = 0; j < D; j++)
        {
            emat(i, j) = mat(i, j);
        }
    }
    return emat;
}

template <Size NumRows, Size NumCols, class Real>
type::Mat<NumRows, NumCols, Real> sofaMat( const Eigen::Matrix<Real, NumRows, NumCols>& emat )
{
    type::Mat<NumRows, NumCols, Real> mat;
    for(Size i=0; i<NumRows; i++)
        for(Size j=0; j<NumCols; j++)
            mat(i,j) = emat(i,j);
    return mat;
}

template<class SofaMat>
void eigenToSofaMat(SofaMat& dest, const Eigen::Matrix<typename SofaMat::Real, SofaMat::nbLines, SofaMat::nbCols>& emat )
{
    for (Size i = 0; i < SofaMat::nbLines; i++)
    {
        for (Size j = 0; j < SofaMat::nbCols; j++)
        {
            dest(i, j) = emat(i, j);
        }
    }
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

enum class EigenValueFilter
{
    ClampNegative,
    Abs
};

template<class SofaMat>
void filterEigenValues(SofaMat& matrix, EigenValueFilter filter)
{
    using EigenMatrix = Eigen::Matrix<typename SofaMat::Real, SofaMat::nbLines, SofaMat::nbCols>;

    EigenMatrix H = sofa::helper::eigenMat(matrix);

    Eigen::SelfAdjointEigenSolver<EigenMatrix> eigen_solver(H);

    if (eigen_solver.info() != Eigen::Success)
    {
        return;
    }

    auto eigenvalues = eigen_solver.eigenvalues();

    if (filter == EigenValueFilter::ClampNegative)
    {
        eigenvalues = eigenvalues.cwiseMax(0);
    }
    else if (filter == EigenValueFilter::Abs)
    {
        eigenvalues = eigenvalues.cwiseAbs();
    }

    H = eigen_solver.eigenvectors() * eigenvalues.asDiagonal() * eigen_solver.eigenvectors().transpose();
    sofa::helper::eigenToSofaMat(matrix, H);
}

}

#endif
