/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_PARTIAL_RIGIDIFICATION_FORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_PARTIAL_RIGIDIFICATION_FORCEFIELD_INL

//#include <sofa/core/behavior/ForceField.inl>
#include <SofaObjectInteraction/PartialRigidificationForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/system/glut.h>
#include <sofa/helper/rmath.h>
#include <assert.h>
#include <iostream>
#include <fstream>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes1, class DataTypes2>
void PartialRigidificationForceField<DataTypes1, DataTypes2>::init()
{
	Inherit1::init();
   /*
    vars.pos6D = this->mstate2->read(core::VecCoordId::position())->getValue()[object2_dof_index.getValue()];
    if(object2_invert.getValue())
        vars.pos6D = DataTypes2::inverse(vars.pos6D);
    initCalcF();
    */
}

template<class DataTypes1, class DataTypes2>
void PartialRigidificationForceField<DataTypes1, DataTypes2>::reinit()
{
	Inherit1::reinit();
    /*
    vars.pos6D = this->mstate2->read(core::VecCoordId::position())->getValue()[object2_dof_index.getValue()];
    if(object2_invert.getValue())
        vars.pos6D = DataTypes2::inverse(vars.pos6D);
    initCalcF();
    */
}



template<class DataTypes1, class DataTypes2>
void PartialRigidificationForceField<DataTypes1, DataTypes2>::addKToMatrix(const sofa::core::MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{

    if(this->f_printLog.getValue())
    {
        sout<<"entering addKToMatrix"<<sendl;
    }

    // mstate1 = mstate templated on Vec3 (free points)
    // mstate2 = mstate templated on Rigid (rigid degrees of freedom)

	std::cout << "Getting matrix for mstate1" << std::endl;
	sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat11 = matrix->getMatrix(this->mstate1);
	std::cout << "Resulting mstate1 matrix size " << mat11.matrix->rows() << std::endl;
	std::cout << "Getting matrix for mstate2" << std::endl;
	sofa::core::behavior::MultiMatrixAccessor::MatrixRef mat22 = matrix->getMatrix(this->mstate2);
	std::cout << "Resulting mstate2 matrix size " << mat22.matrix->rows() << std::endl;
	sofa::core::behavior::MultiMatrixAccessor::InteractionMatrixRef mat12 = matrix->getMatrix(this->mstate1, this->mstate2);
	sofa::core::behavior::MultiMatrixAccessor::InteractionMatrixRef mat21 = matrix->getMatrix(this->mstate2, this->mstate1);
	std::cout << "Resulting mstate1 mstate2 interaction matrix size " << mat12.matrix->rows() << std::endl;
	std::cout << "Resulting mstate2 mstate1 interaction matrix size " << mat21.matrix->rows() << std::endl;

//	matrix->addMechanicalState((const sofa::core::behavior::BaseMechanicalState*)NULL);
//	matrix->addMechanicalState(this->mstate2.get());


// get the 2 jacobian matrices from the subsetMultiMapping
    const helper::vector<sofa::defaulttype::BaseMatrix*>* J0J1 = m_subsetMultiMapping.get()->getJs();
    if (J0J1->size() != 2)
    {
        serr<<"subsetMultiMapping has not 2 output mechanical state. This is not handled in PartialRigidfication.AddKToMatrix not computed "<<sendl;
        return;
    }
    sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>* J0;
    sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>* J1;
    J0 = dynamic_cast< sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type> *> ( (*J0J1)[0] );
    J1 = dynamic_cast< sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type> *> ( (*J0J1)[1] );


// get the jacobian matrix from the rigidMapping
    const sofa::component::linearsolver::CompressedRowSparseMatrix<_3_6_Matrix_Type>* Jr;
    Jr = dynamic_cast<const sofa::component::linearsolver::CompressedRowSparseMatrix<_3_6_Matrix_Type> *> (m_rigidMapping.get()->getJ() );

    if(J0 == NULL)
    {
        serr<<"J0 null"<<sendl;
    }
    if(J1 == NULL)
    {
        serr<<"J1 null"<<sendl;
    }
    if(Jr == NULL)
    {
        serr<<"Jr null"<<sendl;
    }
    if((J0 == NULL) || (J1 == NULL)  || (Jr == NULL) )
    {
        serr<<"a jacobian matrix from Mapping is missing. AddKToMatrix not computed"<<sendl;
        return;
    }
    else
        serr<<" jacobian matrices have been found"<<sendl;


// get the stiffness matrix from the mapped ForceField
    sofa::component::linearsolver::CompressedRowSparseMatrix< _3_3_Matrix_Type > *mappedFFMatrix = new sofa::component::linearsolver::CompressedRowSparseMatrix< _3_3_Matrix_Type > ( );

	sofa::core::behavior::BaseMechanicalState* mstate =m_mappedForceField.get()->getContext()->getMechanicalState();
    mappedFFMatrix->resizeBloc( mstate->getSize() ,  mstate->getSize());

    sofa::component::linearsolver::DefaultMultiMatrixAccessor* mappedFFMatrixAccessor;
    mappedFFMatrixAccessor= new sofa::component::linearsolver::DefaultMultiMatrixAccessor;

	std::cout << "mapped forcefield state name = " << m_mappedForceField.get()->getContext()->getMechanicalState()->getName() << std::endl;
    std::cout<<"m_mappedForceField.get()->getContext()->getMechanicalState() size= "<<m_mappedForceField.get()->getContext()->getMechanicalState()->getMatrixSize()<<std::endl;

	sofa::core::behavior::BaseMechanicalState* thisstate = this->getContext()->getMechanicalState();

	std::cout << "global matrix dim: " << matrix->getGlobalDimension() << std::endl;
	std::cout << "mstate1 global offset: " << matrix->getGlobalOffset(this->mstate1) << std::endl;
	std::cout << "mstat21 global offset: " << matrix->getGlobalOffset(this->mstate2) << std::endl;

	if( thisstate != NULL )
	{
		std::cout << "this mstate size" << this->getContext()->getMechanicalState()->getMatrixSize() << std::endl;
	}
	else
	{
		std::cout << "No mstate in this context" << std::endl;
	}

    mappedFFMatrixAccessor->addMechanicalState(  m_mappedForceField.get()->getContext()->getMechanicalState() );
    mappedFFMatrixAccessor->setGlobalMatrix(mappedFFMatrix);
    mappedFFMatrixAccessor->setupMatrices();

    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = mappedFFMatrixAccessor->getMatrix(  m_mappedForceField.get()->getContext()->getMechanicalState()  );

    std::cout<<" aaa r.matrix size"<< r.matrix->bRowSize()<<" "<< r.matrix->bColSize()<<std::endl;
	std::cout<<" aaa r.matrix size"<< r.matrix->rows()<<" "<< r.matrix->cols()<<std::endl;



    m_mappedForceField.get()->addKToMatrix(mparams, mappedFFMatrixAccessor);
//	std:: cout << "result matrix" << std::endl;
//	std:: cout << *(r.matrix) << std::endl;

//    std::cout<<" dim mappedFFMatrix :"<<mappedFFMatrix->nBlocRow<<" "<<mappedFFMatrix->nBlocCol<<std::endl;
//     std::cout<<" dim J0 :"<<J0->nBlocRow<<" "<<J0->nBlocCol<<std::endl;
//	 std::cout<<" J0 :"<<*J0<<std::endl;
//     std::cout<<" dim J1 :"<<J1->nBlocRow<<" "<<J1->nBlocCol<<std::endl;
//	 std::cout<<" J1 :"<<*J1<<std::endl;
//     std::cout<<" dim Jr :"<<Jr->nBlocRow<<" "<<Jr->nBlocCol<<std::endl;
//	 std::cout<<" Jr :"<<*Jr<<std::endl;

	 std::cout << "Test transpose JO^t*J0" << std::endl;
//	 sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>* t1 = new sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>();
	 sofa::component::linearsolver::CompressedRowSparseMatrix<_3_6_Matrix_Type>* J1Jr = new sofa::component::linearsolver::CompressedRowSparseMatrix<_3_6_Matrix_Type>();
	 sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>* J0tK = new sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>();
	 sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>* J0tKJ0 = new sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>();
	 sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>* K;
	 sofa::component::linearsolver::CompressedRowSparseMatrix<_3_6_Matrix_Type>* J0tKJ1Jr = new sofa::component::linearsolver::CompressedRowSparseMatrix<_3_6_Matrix_Type>();
	 sofa::component::linearsolver::CompressedRowSparseMatrix<_6_6_Matrix_Type>* JrtJ1tKJ1Jr = new sofa::component::linearsolver::CompressedRowSparseMatrix<_6_6_Matrix_Type>();
	 sofa::component::linearsolver::CompressedRowSparseMatrix<_6_3_Matrix_Type>* JrtJ1tK = new sofa::component::linearsolver::CompressedRowSparseMatrix<_6_3_Matrix_Type>();
	 sofa::component::linearsolver::CompressedRowSparseMatrix<_6_3_Matrix_Type>* JrtJ1tKJ0 = new sofa::component::linearsolver::CompressedRowSparseMatrix<_6_3_Matrix_Type>();

	 K = dynamic_cast<sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>*>(r.matrix);

//	 multMatricesT<3,3,3>(*J0, *J0, *t1);
//	 std::cout << "*t1" << std::endl;
//	 std::cout << *t1 << std::endl;

	 multMatrices<3,3,6>(*J1, *Jr, *J1Jr);
	 std::cout << "*J1Jr" << std::endl;
//	 std::cout << *J1Jr << std::endl;


	 multMatricesT<3,3,3>(*J0, *K, *J0tK);
	 std::cout << "J0t * K" << std::endl;
//	 std::cout << *J0tK << std::endl;

	 multMatrices<3,3,3>(*J0tK, *J0, *J0tKJ0); //mat11
	 std::cout << "J0t * K * J0" << std::endl;
//	 std::cout << *J0tKJ0 << std::endl;

	 multMatrices<3, 3, 6>(*J0tK, *J1Jr, *J0tKJ1Jr); // mat12
	 std::cout << "J0t * K * J1 * Jr" << std::endl;
//	 std::cout << *J0tKJ1Jr << std::endl;

	 multMatricesT<3, 6, 3>(*J1Jr, *K, *JrtJ1tK);
	 std::cout << "Jrt * J1t * K" << std::endl;
//	 std::cout << *JrtJ1tK << std::endl;

	 multMatrices<6,3,6>(*JrtJ1tK, *J1Jr, *JrtJ1tKJ1Jr); // mat22
	 std::cout << "Jrt * J1t * K * J1 * Jr" << std::endl;
//	 std::cout << *JrtJ1tKJ1Jr << std::endl;

	 multMatrices<6,3,3>(*JrtJ1tK, *J0, *JrtJ1tKJ0); // mat21
	 std::cout << "Jrt * J1t * K * J0" << std::endl;
//	 std::cout << *JrtJ1tKJ0 << std::endl;


	 /**************************************** Add J0tKJ0 to global system **********************************************/

	 _3_3_Matrix_Type K11MatrixBuffer;
	 for(size_t k11RowIndex = 0 ; k11RowIndex < J0tKJ0->nBlocRow ; ++k11RowIndex)
	 {
		 for(_3_3_ColBlockConstIterator k11ColIter = J0tKJ0->bRowBegin(k11RowIndex); k11ColIter < J0tKJ0->bRowEnd(k11RowIndex) ; k11ColIter++)
		 {

			 _3_3_BlockConstAccessor k11Block = k11ColIter.bloc();

			 const _3_3_Matrix_Type& k11BlockData = *(const _3_3_Matrix_Type*) k11Block.elements(K11MatrixBuffer.ptr()); // get the block

			 int k11ColIndex = k11Block.getCol(); // get the block colum index

			 for (int i = 0; i < 3 ; i++)
				 for (int j = 0; j < 3; j++)
						 mat11.matrix->add(mat11.offset+3*k11RowIndex+i, mat11.offset+3*k11ColIndex+j, k11BlockData(i,j));
		 }
	 }

	 /**************************************** Add JrtJ1tKJ1Jr to global system **********************************************/

	 _6_6_Matrix_Type K22MatrixBuffer;
	 for(size_t k22RowIndex = 0 ; k22RowIndex < JrtJ1tKJ1Jr->nBlocRow ; ++k22RowIndex)
	 {
		 for(_6_6_ColBlockConstIterator k22ColIter = JrtJ1tKJ1Jr->bRowBegin(k22RowIndex); k22ColIter < JrtJ1tKJ1Jr->bRowEnd(k22RowIndex) ; k22ColIter++)
		 {

			 _6_6_BlockConstAccessor k22Block = k22ColIter.bloc();

			 const _6_6_Matrix_Type& k22BlockData = *(const _6_6_Matrix_Type*) k22Block.elements(K22MatrixBuffer.ptr()); // get the block

			 int k22ColIndex = k22Block.getCol(); // get the block colum index

			 for (int i = 0; i < 6 ; i++)
				 for (int j = 0; j < 6; j++)
					 mat22.matrix->add(mat22.offset+6*k22RowIndex+i, mat22.offset+6*k22ColIndex+j, k22BlockData(i,j));
		 }
	 }

	 _3_6_Matrix_Type K12MatrixBuffer;

	 for(size_t k12RowIndex = 0 ; k12RowIndex < J0tKJ1Jr->nBlocRow ; ++k12RowIndex)
	 {
		 for(_3_6_ColBlockConstIterator k12ColIter = J0tKJ1Jr->bRowBegin(k12RowIndex) ; k12ColIter < J0tKJ1Jr->bRowEnd(k12RowIndex) ; k12ColIter++ )
		 {
			 _3_6_BlockConstAccessor k12Block = k12ColIter.bloc();
			 const _3_6_Matrix_Type& k12BlockData = *(const _3_6_Matrix_Type*) k12Block.elements(K12MatrixBuffer.ptr());

			 int k12ColIndex = k12Block.getCol();

			 for(int i = 0 ; i < 3 ; ++i)
				 for(int j = 0 ; j < 6 ; ++j)
					 mat12.matrix->add(mat12.offRow+3*k12RowIndex+i, mat12.offCol+6*k12ColIndex+j, k12BlockData(i,j));
		 }
	 }

	 _6_3_Matrix_Type K21MatrixBuffer;

	 for(size_t k21RowIndex = 0 ; k21RowIndex < JrtJ1tKJ0->nBlocRow ; ++k21RowIndex)
	 {
		 for(_6_3_ColBlockConstIterator k21ColIter = JrtJ1tKJ0->bRowBegin(k21RowIndex) ; k21ColIter < JrtJ1tKJ0->bRowEnd(k21RowIndex) ; k21ColIter++)
		 {
			 _6_3_BlockConstAccessor k21Block = k21ColIter.bloc();
			 const _6_3_Matrix_Type k21BlockData = *(const _6_3_Matrix_Type*) k21Block.elements(K21MatrixBuffer.ptr());

			 int k21ColIndex = k21Block.getCol();

			 for(int i = 0 ; i < 6 ; ++i)
				 for(int j = 0 ; j < 3 ; ++j)
					 mat21.matrix->add(mat21.offRow+6*k21RowIndex+i, mat21.offCol+3*k21ColIndex+j, k21BlockData(i,j));
		 }
	 }


//	 std::ofstream outputFile("./complianceMat.txt");

	 std::cout << "Final system matrix" << std::endl;
//	 std::cout << *mat22.matrix << std::endl;

//	 outputFile.close();

	 delete J1Jr;
	 delete J0tK;
	 delete J0tKJ0;
	 delete K;
	 delete J0tKJ1Jr;
	 delete JrtJ1tKJ1Jr;
	 delete JrtJ1tK;
	 delete JrtJ1tKJ0;



    if(this->f_printLog.getValue())
    {
        sout<<"exit addKToMatrix"<<sendl;
    }

}



template<class DataTypes1, class DataTypes2>
void PartialRigidificationForceField<DataTypes1, DataTypes2>::draw(const core::visual::VisualParams* /*vparams*/)
{

}




} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
