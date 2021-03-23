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

#include <sofa/testing/NumericTest.h>

//SOFA_DEPRECATED_HEADER(v21.12, "sofa/testing/NumericTest.h")

namespace sofa::helper
{
    namespace testing = sofa::testing;
}

//namespace sofa::helper::testing
//{
//    template <typename _Real = SReal>
//    using NumericTest = sofa::testing::NumericTest<_Real>;
//
//    template<class Vector, class ReadData>
//    void copyFromData(Vector& v, const ReadData& d) 
//    {
//        sofa::testing::copyFromData(v,d);
//    }
//
//    template<class WriteData, class Vector>
//    void copyToData(WriteData& d, const Vector& v) 
//    {
//        sofa::testing::copyToData(v, d);
//    }
//
//    template<class _DataTypes>
//    using data_traits = sofa::testing::data_traits<_DataTypes>;
//
//
//} // namespace sofa::helper::testing
