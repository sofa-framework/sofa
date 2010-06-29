/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_EIGENMATRIXMANIPULATOR_H
#define SOFA_COMPONENT_LINEARSOLVER_EIGENMATRIXMANIPULATOR_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/vector.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
USING_PART_OF_NAMESPACE_EIGEN

namespace sofa
{

namespace component
{

namespace linearsolver
{

typedef Eigen::SparseMatrix<SReal,Eigen::RowMajor>    SparseMatrixEigen;
typedef Eigen::SparseVector<SReal,Eigen::RowMajor>    SparseVectorEigen;

struct LMatrixManipulator;

struct LLineManipulator
{
    friend struct LMatrixManipulator;
protected:
    typedef std::pair<unsigned int, SReal> LineCombination;
    typedef helper::vector< LineCombination > InternalData;
public:
    LLineManipulator& addCombination(unsigned int idxConstraint, SReal factor);


protected:
    void buildSparseLine(const helper::vector< SparseVectorEigen >& lines, SparseVectorEigen &vector) const;

    InternalData _data;
};

struct LMatrixManipulator
{
    void init(const SparseMatrixEigen& L);

    void buildLMatrix(const helper::vector<LLineManipulator> &lines, SparseMatrixEigen& matrix) const;

    helper::vector< SparseVectorEigen > LMatrix;
};

}
}
}

#endif
