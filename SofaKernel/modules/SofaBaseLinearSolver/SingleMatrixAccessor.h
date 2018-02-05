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
#ifndef SOFA_CORE_BEHAVIOR_SingleMatrixAccessor_H
#define SOFA_CORE_BEHAVIOR_SingleMatrixAccessor_H
#include "config.h"

#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/defaulttype/BaseMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/** Special case to access a single square matrix.
*/
class SOFA_BASE_LINEAR_SOLVER_API SingleMatrixAccessor : public core::behavior::MultiMatrixAccessor
{
public:
    typedef defaulttype::BaseMatrix BaseMatrix;

    SingleMatrixAccessor( BaseMatrix* m=0 ) { setMatrix(m); }
    virtual ~SingleMatrixAccessor();

    void setMatrix( BaseMatrix* m );
    BaseMatrix* getMatrix() { return matrix; }
    const BaseMatrix* getMatrix() const { return matrix; }


    virtual int getGlobalDimension() const { return matrix->rowSize(); }
    virtual int getGlobalOffset(const core::behavior::BaseMechanicalState*) const { return 0; }
    virtual MatrixRef getMatrix(const core::behavior::BaseMechanicalState*) const;


    virtual InteractionMatrixRef getMatrix(const core::behavior::BaseMechanicalState* mstate1, const core::behavior::BaseMechanicalState* mstate2) const;

protected:
    BaseMatrix* matrix;   ///< The single matrix
    MatrixRef matRef; ///< The accessor to the single matrix

};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
