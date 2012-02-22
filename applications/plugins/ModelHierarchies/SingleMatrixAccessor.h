/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_SingleMatrixAccessor_H
#define SOFA_CORE_BEHAVIOR_SingleMatrixAccessor_H

#include <sofa/core/behavior/MultiMatrixAccessor.h>
//#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/defaulttype/BaseMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/** Special case to access a single matrix.
*/
class SOFA_CORE_API SingleMatrixAccessor : public core::behavior::MultiMatrixAccessor
{
public:
    typedef defaulttype::BaseMatrix Matrix;

    virtual ~SingleMatrixAccessor();

    void setMatrix( Matrix* m );

//    /// Simple structure holding a reference to the submatrix related to one MechanicalState
//    class MatrixRef
//    {
//    public:
//        defaulttype::BaseMatrix* matrix;
//        unsigned int offset;
//        MatrixRef() : matrix(NULL), offset(0) {}
//        defaulttype::BaseMatrix* operator->() const { return matrix; }
//        bool operator!() const { return matrix == NULL; }
//        operator bool() const { return matrix != NULL; }
//        void operator =(const MatrixRef& b){offset = b.offset;matrix = b.matrix;}
//    };

//    /// Simple structure holding a reference to the submatrix related to the interactions between two MechanicalStates
//    class InteractionMatrixRef
//    {
//    public:
//        defaulttype::BaseMatrix* matrix;
//        unsigned int offRow, offCol;
//        InteractionMatrixRef() : matrix(NULL), offRow(0), offCol(0) {}
//        defaulttype::BaseMatrix* operator->() const { return matrix; }
//        bool operator!() const { return matrix == NULL; }
//        operator bool() const { return matrix != NULL; }
//        void operator =(const InteractionMatrixRef& b){offRow = b.offRow;offCol = b.offCol;matrix = b.matrix;}
//    };

    virtual int getGlobalDimension() const { return matrix->rowSize(); }
    virtual int getGlobalOffset(const core::behavior::BaseMechanicalState*) const { return 0; }
    virtual MatrixRef getMatrix(const core::behavior::BaseMechanicalState*) const;

    /// Should not be used. Returns a non-initialized MatrixRef
    virtual InteractionMatrixRef getMatrix(const core::behavior::BaseMechanicalState* mstate1, const core::behavior::BaseMechanicalState* mstate2) const;

protected:
    Matrix* matrix;  ///< The single matrix
    MatrixRef matRef;    ///< The accessor to the single matrix

};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
