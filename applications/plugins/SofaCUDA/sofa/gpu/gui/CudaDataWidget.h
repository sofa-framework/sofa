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
#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/gui/qt/QModelViewTableDataContainer.h>

namespace sofa::gui::qt
{
////////////////////////////////////////////////////////////////
/// variable-sized vectors support
////////////////////////////////////////////////////////////////

template<class T>
class vector_data_trait < sofa::gpu::cuda::CudaVector<T> >
{
public:
    typedef sofa::gpu::cuda::CudaVector<T> data_type;
    typedef T value_type;
    enum { NDIM = 1 };
    static int size(const data_type& d) {
        return d.size();
    }
    static const char* header(const data_type& /*d*/, int /*i*/ = 0)
    {
        return nullptr;
    }
    static const value_type* get(const data_type& d, int i = 0)
    {
        return ((unsigned)i < (unsigned)size(d)) ? &(d[i]) : nullptr;
    }
    static void set(const value_type& v, data_type& d, int i = 0)
    {
        if ((unsigned)i < (unsigned)size(d))
            d[i] = v;
    }
    static void resize(int s, data_type& d)
    {
        d.resize(s);
    }
};


} // namespace sofa::gui::qt
