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

#include <sofa/core/objectmodel/Data.h>
#include <Eigen/Dense>

/// Support of Eigen matrices in sofa::core::objectmodel::Data.
///
/// The string format is "rows cols c00 c01 ... c(rows-1)(cols-1)": the
/// dimensions, then the coefficients in row-major order, all separated by
/// whitespaces. Reading resizes dynamic matrices to the given dimensions, so
/// no pre-sized default value is needed; for fixed-size matrices the
/// dimensions must match the compile-time sizes, otherwise the read fails.
///
/// Both directions are explicit specializations of Data<T>::read() and
/// Data<T>::getValueString(), declared below and defined in Data[Eigen].cpp.
/// No operator<< or operator>> is added to namespace
///
/// Note: a matrix type that is not listed below cannot be stored in a Data:
/// the generic Data<T>::read() would be instantiated and fail to compile, as
/// Eigen matrices have no stream operators.
namespace sofa::core::objectmodel
{
template<> bool SOFA_CORE_API Data<Eigen::MatrixXf>::read( const std::string& str );
template<> std::string SOFA_CORE_API Data<Eigen::MatrixXf>::getValueString() const;

template<> bool SOFA_CORE_API Data<Eigen::MatrixXd>::read( const std::string& str );
template<> std::string SOFA_CORE_API Data<Eigen::MatrixXd>::getValueString() const;

template<> bool SOFA_CORE_API Data<Eigen::Matrix2f>::read( const std::string& str );
template<> std::string SOFA_CORE_API Data<Eigen::Matrix2f>::getValueString() const;

template<> bool SOFA_CORE_API Data<Eigen::Matrix2d>::read( const std::string& str );
template<> std::string SOFA_CORE_API Data<Eigen::Matrix2d>::getValueString() const;

template<> bool SOFA_CORE_API Data<Eigen::Matrix3f>::read( const std::string& str );
template<> std::string SOFA_CORE_API Data<Eigen::Matrix3f>::getValueString() const;

template<> bool SOFA_CORE_API Data<Eigen::Matrix3d>::read( const std::string& str );
template<> std::string SOFA_CORE_API Data<Eigen::Matrix3d>::getValueString() const;

template<> bool SOFA_CORE_API Data<Eigen::Matrix4f>::read( const std::string& str );
template<> std::string SOFA_CORE_API Data<Eigen::Matrix4f>::getValueString() const;

template<> bool SOFA_CORE_API Data<Eigen::Matrix4d>::read( const std::string& str );
template<> std::string SOFA_CORE_API Data<Eigen::Matrix4d>::getValueString() const;

template<> bool SOFA_CORE_API Data<Eigen::MatrixXi>::read( const std::string& str );
template<> std::string SOFA_CORE_API Data<Eigen::MatrixXi>::getValueString() const;
} // namespace sofa::core::objectmodel
