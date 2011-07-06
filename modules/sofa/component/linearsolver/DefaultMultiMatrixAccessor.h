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
#include <sofa/core/objectmodel/Data.h>
#include <sofa/component/component.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
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




class SOFA_COMPONENT_LINEARSOLVER_API DefaultMultiMatrixAccessor : public sofa::core::behavior::MultiMatrixAccessor
{
public:
    DefaultMultiMatrixAccessor();
    virtual ~DefaultMultiMatrixAccessor();

    virtual void clear();

    // setting the global matrix for the system. Its size must have the sum of all real Mechanical state
    virtual void setGlobalMatrix(defaulttype::BaseMatrix* matrix);

    // When a real MS is visited by the visitor, it must be registed in a local data here (realStateOffsets)
    // the global size of the system must be ajusted.
    virtual void addMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate);

    // When a mapping is visited by the visitor, satisfying that is a mechanical mapping
    // and having implemented getJ, this mapping must be registed in a local data here (mappingList)
    virtual void addMechanicalMapping(sofa::core::BaseMapping* mapping);

    //do nothing for instance
    virtual void addMappedMechanicalState(const sofa::core::behavior::BaseMechanicalState* mstate);

    //Read all Real Mechanical State
    virtual void setupMatrices();

    //give the sum of size of all Real Mechanical State in ordre to set the global matrix dimension
    virtual int getGlobalDimension() const;

    //give position in global matrix of the blog related to a given Mechanical State
    virtual int getGlobalOffset(const sofa::core::behavior::BaseMechanicalState* mstate) const;

    //give the Matrix Reference (Matrix and Offset) related to a given Mechanical State
    virtual MatrixRef getMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const;

    //give the Matrix Reference (Matrix and Offset) related to a interactionForceField (between 2 Mechanical State)
    virtual InteractionMatrixRef getMatrix(const BaseMechanicalState* mstate1, const BaseMechanicalState* mstate2) const;

    //Compute the global system matrix
    //If there are no mapping, do nothing
    //If there are mappings, compute the contribution
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


    //Matrix creating is only call when there are mapped state,
    //the stiffness and interaction stiffness of this state couldn't directly described on the principal matrix
    //then it demande to create a new matrix
    virtual defaulttype::BaseMatrix* createMatrix(const BaseMechanicalState* mstate) const;
    virtual defaulttype::BaseMatrix* createInteractionMatrix(const BaseMechanicalState* mstate1, const BaseMechanicalState* mstate2) const;

    bool MULTIMATRIX_VERBOSE;
};



//TODO separating in other file
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* CRSMultiMatrixAccessor do the same thing as DefaultMultiMatrixAccessor but optimal.
 * The different is instead of creating a standard full matrix for mapped and mapping, CRSMultiMatrixAccessor works
 * with CompressedRowSparseMatrix.
 *
 * To be able to creat  CompressedRowSparseMatrix, it is needle to know about block format of relied to the
 * size of DOF of mapped Mechanical state and input-output Mechanical State of the mapping
 * */
class SOFA_COMPONENT_LINEARSOLVER_API CRSMultiMatrixAccessor : public DefaultMultiMatrixAccessor
{
public:
    CRSMultiMatrixAccessor() : DefaultMultiMatrixAccessor() {}
    ~CRSMultiMatrixAccessor() {	this->clear();}

    //Creating the stiffness matrix for a mapped Mechanical State
    virtual defaulttype::BaseMatrix* createMatrix(const sofa::core::behavior::BaseMechanicalState* mstate) const;

    //Creating the stiffness interaction matrix for pair of Mechanical State when they are not all real state
    virtual defaulttype::BaseMatrix* createInteractionMatrix(const sofa::core::behavior::BaseMechanicalState* mstate1, const sofa::core::behavior::BaseMechanicalState* mstate2) const;

    //Compute the contribution of all new created matrix to the the global system matrix
    virtual void computeGlobalMatrix();
};




/////////////////////////////////////////////////////////////////////////////////
///////////  SWITCH COMPILATION PROBLEM TO RUNTIME PROBLEM  /////////////////////

//template<int JblocRsize, int JblocCsize, int MblocCsize, class JelementType, class MelementType>
//bool opMulJTM_TFinal(defaulttype::BaseMatrix* out, const defaulttype::BaseMatrix* J, const defaulttype::BaseMatrix* M, int offsetRow, int offsetCol)
//{
//	typedef defaulttype::Mat<JblocRsize, JblocCsize, JelementType> JBloc;
//	typedef CompressedRowSparseMatrix<JBloc> JMatrix;
//	typedef defaulttype::Mat<JblocRsize, MblocCsize, MelementType> MBloc;
//	typedef CompressedRowSparseMatrix<MBloc> MMatrix;
//	typedef defaulttype::Mat<JblocCsize, MblocCsize, MelementType> OutBloc;
//	typedef CompressedRowSparseMatrix<OutBloc> OutMatrix;
//	JMatrix* Jmatrix = dynamic_cast<JMatrix*>(J);
//	MMatrix* Mmatrix = dynamic_cast<MMatrix*>(M);
//	OutMatrix* Outmatrix = dynamic_cast<OutMatrix*>(out);
//	if (!Jmatrix || !Mmatrix) return false;
//	// We can compute out += Jt M
//	if (!Outmatrix || (offsetRow % JblocCsize != 0) || (offsetCol % MblocCsize != 0))
//	{ // optimized multiplication, but generic output
//	}
//	else
//	{ // optimized multiplication and output
//		int offsetBRow = offsetRow / JblocCsize;
//		int offsetBCol = offsetCol / MblocCsize;
//	}
//	return true;
//}





template<int blocRsize, int blocCsize, class elementType>
inline defaulttype::BaseMatrix* createBlocSparseMatrixT(int nbRowBloc, int nbColBloc)
{
    typedef CompressedRowSparseMatrix< defaulttype::Mat<blocRsize, blocCsize, elementType> > BlocMatrix;
    BlocMatrix* m =	new BlocMatrix;
    m->resizeBloc(nbRowBloc,nbColBloc);
    return m;
}

template<int blocRsize, int blocCsize>
inline defaulttype::BaseMatrix* createBlocSparseMatrixTReal(int elementSize, int nbRowBloc, int nbColBloc)
{
    switch(elementSize)
    {
    case sizeof(float) : return createBlocSparseMatrixT<blocRsize,blocCsize, float>(nbRowBloc,nbColBloc);
    case sizeof(double): return createBlocSparseMatrixT<blocRsize,blocCsize,double>(nbRowBloc,nbColBloc);
    default            : return createBlocSparseMatrixT<blocRsize,blocCsize, SReal>(nbRowBloc,nbColBloc);
    }
}

template<int blocRsize>
inline defaulttype::BaseMatrix* createBlocSparseMatrixTRow(int blocCsize, int elementSize, int nbRowBloc, int nbColBloc)
{
    switch(blocCsize)
    {
    case 1: return createBlocSparseMatrixTReal<blocRsize,1>(elementSize  ,nbRowBloc,nbColBloc);
    case 2: return createBlocSparseMatrixTReal<blocRsize,2>(elementSize  ,nbRowBloc,nbColBloc);
    case 3: return createBlocSparseMatrixTReal<blocRsize,3>(elementSize  ,nbRowBloc,nbColBloc);
    case 4: return createBlocSparseMatrixTReal<blocRsize,4>(elementSize  ,nbRowBloc,nbColBloc);
    case 5: return createBlocSparseMatrixTReal<blocRsize,5>(elementSize  ,nbRowBloc,nbColBloc);
    case 6: return createBlocSparseMatrixTReal<blocRsize,6>(elementSize  ,nbRowBloc,nbColBloc);
    default: return createBlocSparseMatrixTReal<blocRsize,1>(elementSize  ,nbRowBloc,nbColBloc);
    }
}

inline defaulttype::BaseMatrix* createBlocSparseMatrix(int blocRsize, int blocCsize, int elementSize, int nbRowBloc, int nbColBloc)
{
    switch(blocRsize)
    {
    case 1: return createBlocSparseMatrixTRow<1>(blocCsize, elementSize  ,nbRowBloc,nbColBloc);
    case 2: return createBlocSparseMatrixTRow<2>(blocCsize, elementSize  ,nbRowBloc,nbColBloc);
    case 3: return createBlocSparseMatrixTRow<3>(blocCsize, elementSize  ,nbRowBloc,nbColBloc);
    case 4: return createBlocSparseMatrixTRow<4>(blocCsize, elementSize  ,nbRowBloc,nbColBloc);
    case 5: return createBlocSparseMatrixTRow<5>(blocCsize, elementSize  ,nbRowBloc,nbColBloc);
    case 6: return createBlocSparseMatrixTRow<6>(blocCsize, elementSize  ,nbRowBloc,nbColBloc);
    default: return createBlocSparseMatrixTRow<1>(blocCsize, elementSize  ,nbRowBloc,nbColBloc);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


















} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
