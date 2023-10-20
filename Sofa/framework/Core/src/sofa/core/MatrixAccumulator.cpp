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

#include <sofa/core/MatrixAccumulator.h>

namespace sofa::core
{
void MatrixAccumulatorInterface::add(sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<1, 1, float>& value)
{
    add(row, col, value[0][0]);
}

void MatrixAccumulatorInterface::add(sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<1, 1, double>& value)
{
    add(row, col, value[0][0]);
}

void MatrixAccumulatorInterface::add(sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<2, 2, float>& value)
{
    matAdd(row, col, value);
}

void MatrixAccumulatorInterface::add(sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<2, 2, double>& value)
{
    matAdd(row, col, value);
}

void MatrixAccumulatorInterface::add(sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<3, 3, float>& value)
{
    matAdd(row, col, value);
}

void MatrixAccumulatorInterface::add(sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<3, 3, double>& value)
{
    matAdd(row, col, value);
}

void MatrixAccumulatorInterface::add(sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<6, 6, float>& value)
{
    matAdd(row, col, value);
}

void MatrixAccumulatorInterface::add(sofa::SignedIndex row, sofa::SignedIndex col,
    const sofa::type::Mat<6, 6, double>& value)
{
    matAdd(row, col, value);
}

helper::logging::MessageDispatcher::LoggerStream matrixaccumulator::RangeVerification::
logger() const
{
    return m_messageComponent
               ? msg_error(m_messageComponent)
               : msg_error("RangeVerification");
}

void matrixaccumulator::RangeVerification::checkRowIndex(sofa::SignedIndex row)
{
    if (row < minRowIndex)
    {
        logger() << "Trying to accumulate a matrix entry out of the allowed submatrix: minimum "
            "row index is " << minRowIndex << " while " << row << " was provided";
    }
    if (row > maxRowIndex)
    {
        logger() << "Trying to accumulate a matrix entry out of the allowed submatrix: maximum "
            "row index is " << maxRowIndex << " while " << row << " was provided";
    }
}

void matrixaccumulator::RangeVerification::checkColIndex(sofa::SignedIndex col)
{
    if (col < minColIndex)
    {
        logger() << "Trying to accumulate a matrix entry out of the allowed submatrix: minimum "
            "column index is " << minColIndex << " while " << col << " was provided";
    }
    if (col > maxColIndex)
    {
        logger() << "Trying to accumulate a matrix entry out of the allowed submatrix: maximum "
            "column index is " << maxColIndex << " while " << col << " was provided";
    }
}
}
