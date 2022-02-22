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
#include <sofa/core/config.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <vector>
#include <map>

namespace sofa::core::behavior
{

/* DefaultMultiMatrixAccessor is a simplest class managing the global matrix setup.
 * This class allow only cases where there are several Mechanical State on simulation scene
 * and interactions between them.
 *
 * CRSMultiMatrixAccessor is a more powerfull class managing the global system matrix.
 * This class allow the scene where there are mappings so mapped Mechanical State. It compute
 * The contribution of stiffness on mapped Mechanical State to the root State related by mapping.
 *
 * see Sofa/doc/modules/linearsolver/LinearSolver.tex
 * see all scenes in example/Component/linearsolver/MatrixContribution*  for tests
 *
 * */
class SOFA_CORE_API DefaultMultiMatrixAccessor : public sofa::core::behavior::MultiMatrixAccessor
{
public:
    DefaultMultiMatrixAccessor();
    ~DefaultMultiMatrixAccessor() override;

    virtual void clear();

    // setting the global matrix for the system. Its size must have the sum of all real Mechanical state
    virtual void setGlobalMatrix(linearalgebra::BaseMatrix* matrix);

    // When a real MS is visited by the visitor, it must be registed in a local data here (realStateOffsets)
    // the global size of the system must be ajusted.
    void addMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate) override;

    // When a mapping is visited by the visitor, satisfying that is a mechanical mapping
    // and having implemented getJ, this mapping must be registed in a local data here (mappingList)
    void addMechanicalMapping(sofa::core::BaseMapping* mapping) override;

    //do nothing for instance
    void addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate) override;

    //Read all Real Mechanical State
    virtual void setupMatrices();

    //give the sum of size of all Real Mechanical State in ordre to set the global matrix dimension
    Index getGlobalDimension() const override;

    //give position in global matrix of the blog related to a given Mechanical State
    int getGlobalOffset(const sofa::core::behavior::BaseMechanicalState* mstate) const override;

    //give the Matrix Reference (Matrix and Offset) related to a given Mechanical State
    MatrixRef getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const override;

    //give the Matrix Reference (Matrix and Offset) related to a interactionForceField (between 2 Mechanical State)
    InteractionMatrixRef getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const override;

    //Compute the global system matrix
    //If there are no mapping, do nothing
    //If there are mappings, compute the contribution
    virtual void computeGlobalMatrix();


    //Matrix creating is only call when there are mapped state,
    //the stiffness and interaction stiffness of this state couldn't directly described on the principal matrix
    //then it demande to create a new matrix
    static linearalgebra::BaseMatrix* createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2);
    static linearalgebra::BaseMatrix* createMatrixImpl(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2, bool doPrintInfo);

    //Activate/deactive the printing of extra information related to the numerical system that is being solved.
    void setDoPrintInfo(bool value){ m_doPrintInfo = value; }

protected:
    bool m_doPrintInfo {false} ;
    linearalgebra::BaseMatrix* globalMatrix {nullptr} ;
    Index globalDim {0} ;

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
    std::map< const sofa::core::behavior::BaseMechanicalState*, unsigned int > realStateOffsets;

    //map used only for mapped mechanical state
    //a mapped state is added here if and only if its stiffness matrix is guessed by other component (forcefield)
    //by method "getMatrix" in order to fill its matrix
    mutable std::map< const sofa::core::behavior::BaseMechanicalState*, linearalgebra::BaseMatrix* > mappedMatrices;

    //The data structure included mapped and on mapped state, the diagonal stiffness bloc and interaction stiffnessbloc
    mutable std::map< const sofa::core::behavior::BaseMechanicalState*, MatrixRef > diagonalStiffnessBloc;//todo remove
    mutable std::map< std::pair<const sofa::core::behavior::BaseMechanicalState*, const sofa::core::behavior::BaseMechanicalState*>, InteractionMatrixRef > interactionStiffnessBloc;//todo remove

    //The list of validated mapping in the order of visitor, to be read in the inverted direction for propagation contribution
    std::vector<sofa::core::BaseMapping*> mappingList;
};

} // namespace sofa::core::behavior
