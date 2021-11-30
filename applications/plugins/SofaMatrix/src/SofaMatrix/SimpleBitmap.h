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
#include <SofaMatrix/config.h>
#include <image/CImgData.h>

namespace sofa::type
{

/// A simple proxy of an Image, compatible with a Data. It can be visualized with a SimpleImageViewerWidget
template<typename _T>
struct SimpleBitmap
{
    typedef _T T;
    typedef sofa::defaulttype::Image<T> ImageTypes;
    typedef SReal Real;

    SimpleBitmap()
        : m_img(nullptr)
    {}

    static const char* Name() { return "SimpleBitmap"; }

    void setInput(const ImageTypes& img)
    {
        m_img = &img;
    }

    friend std::istream& operator >> ( std::istream& in, SimpleBitmap& p )
    {
        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const SimpleBitmap& p )
    {
        return out;
    }

    [[nodiscard]] const ImageTypes* getImage() const
    {
        return m_img;
    }

protected:

    /// input image
    const ImageTypes* m_img { nullptr };

};

} //namespace sofa::type