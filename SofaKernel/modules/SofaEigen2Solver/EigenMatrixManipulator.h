/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_EIGENMATRIXMANIPULATOR_H
#define SOFA_COMPONENT_LINEARSOLVER_EIGENMATRIXMANIPULATOR_H
#include "config.h"

#include <sofa/helper/system/config.h>
#include <sofa/helper/vector.h>

#include <Eigen/Core>
#ifndef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#endif
#include <Eigen/Sparse>

namespace sofa
{

namespace component
{

namespace linearsolver
{

typedef Eigen::SparseMatrix<SReal>    SparseMatrixEigen;
typedef Eigen::SparseVector<SReal>    SparseVectorEigen;
typedef Eigen::Matrix<SReal, Eigen::Dynamic, 1>       VectorEigen;

struct LMatrixManipulator;

struct SOFA_EIGEN2_SOLVER_API LLineManipulator
{
    typedef std::pair<unsigned int, SReal> LineCombination;
    typedef helper::vector< LineCombination > InternalData;
public:
    LLineManipulator& addCombination(unsigned int idxConstraint, SReal factor=1.0);

    inline friend std::ostream& operator << ( std::ostream& out, const LLineManipulator& s )
    {
        for (InternalData::const_iterator it = s._data.begin(); it!=s._data.end(); ++it)
        {
            if (it->second == 1.0)
                out << "[" << it->first << "] ";
            else
                out << "[" << it->first << ", " << it->second <<"] ";
        }
        return out;
    }

    template <class Container, class Result>
    void buildCombination(const Container& lines, Result &output) const
    {
        //TODO: improve estimation of non zero coeff
        for (InternalData::const_iterator it=_data.begin(); it!=_data.end(); ++it)
        {
            const unsigned int indexConstraint=it->first;
            const SReal factor=it->second;
            output += lines[indexConstraint]*factor;
        }
    }
protected:
    InternalData _data;
};

struct SOFA_EIGEN2_SOLVER_API LMatrixManipulator
{
    void init(const SparseMatrixEigen& L);

    void buildLMatrix(const helper::vector<LLineManipulator> &lines, SparseMatrixEigen& matrix) const;

    helper::vector< SparseVectorEigen > LMatrix;
};

}
}
}

#endif
