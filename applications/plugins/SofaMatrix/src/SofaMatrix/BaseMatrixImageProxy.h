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
#include <sofa/linearalgebra/BaseMatrix.h>

namespace sofa::type
{

/// A simple proxy of a BaseMatrix, compatible with a Data that can be visualized in the GUI, as a bitmap, with a BaseMatrixImageViewerWidget
struct BaseMatrixImageProxy
{
    typedef SReal Real;

    BaseMatrixImageProxy()
    {}

    static const char* Name() { return "SimpleBitmap"; }

    friend std::istream& operator >> ( std::istream& in, BaseMatrixImageProxy& /*p*/ )
    {
        return in;
    }

    friend std::ostream& operator << ( std::ostream& out, const BaseMatrixImageProxy& p )
    {
        if (!p.m_matrix)
        {
            out << "invalid matrix";
        }
        else
        {
            out << std::to_string(p.m_matrix->rows()) << "x" << std::to_string(p.m_matrix->cols());
        }
        return out;
    }

    [[nodiscard]] linearalgebra::BaseMatrix* getMatrix() const
    {
        return m_matrix;
    }

    void setMatrix(linearalgebra::BaseMatrix* m_matrix)
    {
        this->m_matrix = m_matrix;
    }

protected:

    linearalgebra::BaseMatrix* m_matrix { nullptr };
};

} //namespace sofa::type