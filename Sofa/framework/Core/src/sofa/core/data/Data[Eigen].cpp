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
#include <sofa/core/data/Data[Eigen].h>
namespace sofa::core::objectmodel
{

/// Specialization for reading strings into an Eigen::MatrixXF
template<>
bool Data<Eigen::MatrixXf>::read( const std::string& str )
{
    std::stringstream tmp{str};
    auto matrix = *(this->m_value.beginEdit());

    for (int i = 0; i < matrix.rows(); ++i)
    {
        for (int j = 0; j < matrix.cols(); ++j)
        {
            tmp >> matrix(i, j);
        }
    }
    endEdit();
    return true;
}

template<>
std::string Data<Eigen::MatrixXf>::getValueString() const
{
    std::stringstream tmp;
    auto matrix = this->m_value.getValue();
    for (int i = 0; i < matrix.rows(); ++i)
    {
        for (int j = 0; j < matrix.cols(); ++j)
        {
            tmp >> matrix(i, j);
        }
    }
    return tmp.str();
}

}
