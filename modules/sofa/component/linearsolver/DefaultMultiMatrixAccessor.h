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
#ifndef SOFA_COMPONENT_LINEARSOLVER_DEFAULTMULTIMATRIXACCESSOR_H
#define SOFA_COMPONENT_LINEARSOLVER_DEFAULTMULTIMATRIXACCESSOR_H

#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/component.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/GraphScatteredTypes.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

class SOFA_COMPONENT_LINEARSOLVER_API DefaultMultiMatrixAccessor : public sofa::core::behavior::MultiMatrixAccessor
{
public:
    DefaultMultiMatrixAccessor();
    virtual ~DefaultMultiMatrixAccessor();

    virtual void clear();
    virtual void setGlobalMatrix(defaulttype::BaseMatrix* matrix);

    virtual void addMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate);
    virtual void addMechanicalMapping(const sofa::core::BaseMapping* mapping);
    virtual void addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate);

    virtual void setupMatrices();

    virtual int getGlobalDimension() const;
    virtual int getGlobalOffset(const sofa::core::behavior::BaseMechanicalState* mstate) const;

    virtual MatrixRef getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const;
    virtual InteractionMatrixRef getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const;

    virtual void computeGlobalMatrix();

protected:

    defaulttype::BaseMatrix* globalMatrix;
    unsigned int globalDim;
    std::map< const sofa::core::behavior::BaseMechanicalState*, int > globalOffsets;
    std::set< defaulttype::BaseMatrix* > tempMatrices;
    std::map< const sofa::core::behavior::BaseMechanicalState*, MatrixRef > localMatrixMap;
    std::map< std::pair<const sofa::core::behavior::BaseMechanicalState*, const sofa::core::behavior::BaseMechanicalState*>, InteractionMatrixRef > interactionMatrixMap;
    std::map< const sofa::core::behavior::BaseMechanicalState*, std::map< const sofa::core::behavior::BaseMechanicalState*, InteractionMatrixRef > > mappingMatrixMap;

    virtual defaulttype::BaseMatrix* createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate);
    virtual defaulttype::BaseMatrix* createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2);

};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
