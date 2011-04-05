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
    virtual void addMechanicalMapping(sofa::core::BaseMapping* mapping);
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

    //in the list of mapping, only mapping where its getJ give a non-NULL matrix is valid, this one is attached with true value
    //The one whose getJ not implemented will attache with the false value
    mutable std::map<sofa::core::BaseMapping*, bool> mappingsList;

    //mappingTree allows to find the mapping knowing its state of ToModel
    mutable std::map< const sofa::core::behavior::BaseMechanicalState*, sofa::core::BaseMapping* > mappingsContributionTree;

    mutable std::map< const sofa::core::behavior::BaseMechanicalState*, defaulttype::BaseMatrix* > mappedMatrices;

    mutable std::map< const sofa::core::behavior::BaseMechanicalState*, MatrixRef > localMatrixMap;
    mutable std::map< std::pair<const sofa::core::behavior::BaseMechanicalState*, const sofa::core::behavior::BaseMechanicalState*>, InteractionMatrixRef > interactionMatrixMap;

    virtual defaulttype::BaseMatrix* createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const;
    virtual defaulttype::BaseMatrix* createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const;

};



//TODO separating in other file
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SOFA_COMPONENT_LINEARSOLVER_API MappedMultiMatrixAccessor : public DefaultMultiMatrixAccessor
{
public:
    MappedMultiMatrixAccessor() : DefaultMultiMatrixAccessor() {}
    virtual ~MappedMultiMatrixAccessor() { this->clear();}

    virtual defaulttype::BaseMatrix* createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const;
    virtual defaulttype::BaseMatrix* createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const;
    virtual void computeGlobalMatrix();

};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
