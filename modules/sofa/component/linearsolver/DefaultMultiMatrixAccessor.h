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
#include <vector>


namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::core;
using namespace sofa::core::behavior;


/* see all scenes in example/SanBox/MatrixContribution*  for diversity of test
 *
 *
 * */




class SOFA_COMPONENT_LINEARSOLVER_API DefaultMultiMatrixAccessor : public sofa::core::behavior::MultiMatrixAccessor
{
public:
    DefaultMultiMatrixAccessor();
    virtual ~DefaultMultiMatrixAccessor();

    virtual void clear();

    // setting the global matrix for the system. Its size must have the sum of all real Mechanical state
    virtual void setGlobalMatrix(defaulttype::BaseMatrix* matrix);

    // When a real MS is visited by the visitor, it must be registed in a local data here (realStateOffsets)
    // the global size of the system must be ajusted (adding the size of this MS)
    virtual void addMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate);

    // When a mapping is visited by the visitor, satisfying that is a mechanical mapping
    // and having implemented getJ, this mapping must be registed in a local data here (mappingList)
    virtual void addMechanicalMapping(sofa::core::BaseMapping* mapping);

    virtual void addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate);

    virtual void setupMatrices();

    virtual int getGlobalDimension() const;
    virtual int getGlobalOffset(const sofa::core::behavior::BaseMechanicalState* mstate) const;

    virtual MatrixRef getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const;
    virtual InteractionMatrixRef getMatrix(const BaseMechanicalState* mstate1, const BaseMechanicalState* mstate2) const;

    virtual void computeGlobalMatrix();

protected:

    defaulttype::BaseMatrix* globalMatrix;
    unsigned int globalDim;

    //           case1                                           case2
    //      |               |                                  |       |
    //     MS1             MS2                                MS1     MS2
    //      |               |                                 /      /
    //     mapA            mapB                             map   Inter
    //      |               |                                 \   /
    //     MS3 ---Inter--  MS4                                MS3/
    //
    //    K11 += JAt * K33 * JA                         K11 += Jt * K33 * J
    //    K22 += JBt * K44 * JB                         K12 += Jt * I32
    //    K12 += JAt * I34 * JB                         K21 +=      I32 * J
    //    K21 += JBt * I43 * JA
    //
    // using matrix buffer in case of interaction between mapped model

    //map used for real mechanical state (non-mapped only)
    std::map< const sofa::core::behavior::BaseMechanicalState*, int > realStateOffsets;

    //map used only for mapped mechanical state
    //a mapped state is added here if and only if its stiffness matrix is guessed by other component (forcefield)
    //by method "getMatrix" in order to fill its matrix
    mutable std::map< const BaseMechanicalState*, defaulttype::BaseMatrix* > mappedMatrices;

    //The data structure included mapped and on mapped state, the diagonal stiffness bloc and interaction stiffnessbloc
    mutable std::map< const BaseMechanicalState*, MatrixRef > diagonalStiffnessBloc;//todo remove
    mutable std::map< std::pair<const BaseMechanicalState*, const BaseMechanicalState*>, InteractionMatrixRef > interactionStiffnessBloc;//todo remove

    //The list of validated mapping in the order of visitor, to be read in the inverted direction for propagation contribution
    std::vector<sofa::core::BaseMapping*> mappingList;



    //Matrix creating is only call when there are a mapped state,
    //the stiffness and interaction stiffness of this state could'nt described on the principal matrix
    //then it demande to create a new matrix
    virtual defaulttype::BaseMatrix* createMatrix(const BaseMechanicalState* mstate) const;
    virtual defaulttype::BaseMatrix* createInteractionMatrix(const BaseMechanicalState* mstate1, const BaseMechanicalState* mstate2) const;

    bool MULTIMATRIX_VERBOSE;
};



//TODO separating in other file
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SOFA_COMPONENT_LINEARSOLVER_API MappedMultiMatrixAccessor : public DefaultMultiMatrixAccessor
{
public:
    MappedMultiMatrixAccessor() : DefaultMultiMatrixAccessor() {}
    ~MappedMultiMatrixAccessor() {	this->clear();}

    virtual defaulttype::BaseMatrix* createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const;
    virtual defaulttype::BaseMatrix* createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const;
    virtual void computeGlobalMatrix();

};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
