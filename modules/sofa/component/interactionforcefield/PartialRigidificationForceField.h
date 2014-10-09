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
#ifndef SOFA_COMPONENT_FORCEFIELD_PARTIAL_RIGIDIFICATION_FORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_PARTIAL_RIGIDIFICATION_FORCEFIELD_H

#include <sofa/core/behavior/MixedInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/component/linearsolver/DefaultMultiMatrixAccessor.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{
using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::component::mapping;
using namespace sofa::core::objectmodel;


/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes1, class DataTypes2>
class PartialRigidificationForceFieldInternalData
{
public:
};

using sofa::component::linearsolver::CompressedRowSparseMatrix;
using defaulttype::Mat;

template<typename TDataTypes1, typename TDataTypes2>
class PartialRigidificationForceField : public core::behavior::MixedInteractionForceField<TDataTypes1, TDataTypes2>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(PartialRigidificationForceField, TDataTypes1, TDataTypes2), SOFA_TEMPLATE2(core::behavior::MixedInteractionForceField, TDataTypes1, TDataTypes2));

    typedef core::behavior::MixedInteractionForceField<TDataTypes1, TDataTypes2> Inherit;
    // Vec3
    typedef TDataTypes1 DataTypes1;
    typedef typename DataTypes1::VecCoord VecCoord1;
    typedef typename DataTypes1::VecDeriv VecDeriv1;
    typedef typename DataTypes1::Coord    Coord1;
    typedef typename DataTypes1::Deriv    Deriv1;
    typedef typename DataTypes1::Real     Real1;
    // Rigid
    typedef TDataTypes2 DataTypes2;
    typedef typename DataTypes2::VecCoord VecCoord2;
    typedef typename DataTypes2::VecDeriv VecDeriv2;
    typedef typename DataTypes2::Coord    Coord2;
    typedef typename DataTypes2::Deriv    Deriv2;
    typedef typename DataTypes2::Real     Real2;

    typedef core::objectmodel::Data<VecCoord1>    DataVecCoord1;
    typedef core::objectmodel::Data<VecDeriv1>    DataVecDeriv1;
    typedef core::objectmodel::Data<VecCoord2>    DataVecCoord2;
    typedef core::objectmodel::Data<VecDeriv2>    DataVecDeriv2;

  //  typedef typename defaulttype::Rigid3dTypes Rigid;
  //    typedef sofa::component::mapping::BarycentricMapping< DataTypes, Rigid3dTypes >   BarycentricMapping3_VtoR;

    typedef defaulttype::Mat<6,3,Real1> _6_3_Matrix_Type;
    typedef defaulttype::Mat<6,6,Real2> _6_6_Matrix_Type;
    typedef defaulttype::Mat<3,6,Real2> _3_6_Matrix_Type;
    typedef defaulttype::Mat<3,3,Real1> _3_3_Matrix_Type;

    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_6_6_Matrix_Type>::ColBlockConstIterator _6_6_ColBlockConstIterator;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_6_3_Matrix_Type>::ColBlockConstIterator _6_3_ColBlockConstIterator;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>::ColBlockConstIterator _3_3_ColBlockConstIterator;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_3_6_Matrix_Type>::ColBlockConstIterator _3_6_ColBlockConstIterator;

    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_6_6_Matrix_Type>::RowBlockConstIterator _6_6_RowBlockConstIterator;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_6_3_Matrix_Type>::RowBlockConstIterator _6_3_RowBlockConstIterator;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>::RowBlockConstIterator _3_3_RowBlockConstIterator;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_3_6_Matrix_Type>::RowBlockConstIterator _3_6_RowBlockConstIterator;

    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_6_6_Matrix_Type>::BlockConstAccessor _6_6_BlockConstAccessor;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_6_3_Matrix_Type>::BlockConstAccessor _6_3_BlockConstAccessor;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>::BlockConstAccessor _3_3_BlockConstAccessor;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_3_6_Matrix_Type>::BlockConstAccessor _3_6_BlockConstAccessor;

    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_6_6_Matrix_Type>::BlockAccessor _6_6_BlockAccessor;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_6_3_Matrix_Type>::BlockAccessor _6_3_BlockAccessor;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_3_3_Matrix_Type>::BlockAccessor _3_3_BlockAccessor;
    typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<_3_6_Matrix_Type>::BlockAccessor _3_6_BlockAccessor;


//    enum { N=DataTypes1::spatial_dimensions };
//    typedef defaulttype::Mat<N,N,Real1> Mat;
protected:
    SingleLink<PartialRigidificationForceField<DataTypes1,DataTypes2>, RigidMapping<DataTypes2, DataTypes1>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> m_rigidMapping;
    SingleLink<PartialRigidificationForceField<DataTypes1,DataTypes2>, SubsetMultiMapping<DataTypes1,DataTypes1>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> m_subsetMultiMapping;
	SingleLink<PartialRigidificationForceField<DataTypes1,DataTypes2>, core::behavior::BaseForceField, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> m_mappedForceField;


public:



protected:
    PartialRigidificationForceField():
        m_rigidMapping(initLink("rigidMapping", " link to the rigidMapping that does the rigidification")),
        m_subsetMultiMapping(initLink("subsetMultiMapping", " link to the subsetMultiMapping that unifies rigid and deformable parts")),
        m_mappedForceField(initLink("mappedForceField", "link to the forceField that is mapped under the subsetmultimapping"))
    {
    }

	// CSR Matrices product : R = A * B
//	template<unsigned int M, unsigned int N, unsigned int K>
//	static void multMatrices(const CompressedRowSparseMatrix<Mat<M,N, Real1> >& A, const CompressedRowSparseMatrix<Mat<N,K, Real1> >& B, CompressedRowSparseMatrix<Mat<M,K, Real1> >& R);

	template<unsigned int M, unsigned int N, unsigned int K>
	static void multMatrices(const CompressedRowSparseMatrix<Mat<M,N, Real1> >& A,
							 const CompressedRowSparseMatrix<Mat<N,K, Real1> >& B,
							 CompressedRowSparseMatrix<Mat<M,K, Real1> >& R)
	{
		size_t rBlockR = A.nBlocRow;
		size_t cBlockR = B.nBlocCol;

		R.resizeBloc(rBlockR, cBlockR);

		typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat<M,N, Real1> >::ColBlockConstIterator AColBlockIter;
		typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat<M,N, Real1> >::BlockConstAccessor ABlockConstAccessor;
		typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat<N,K, Real1> >::ColBlockConstIterator BColBlockIter;
		typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat<N,K, Real1> >::BlockConstAccessor BBlockConstAccessor;

		Mat<M,N, Real1> AMatrixBuffer;
		Mat<N,K, Real1> BMatrixBuffer;
		for (int ArBlockId = 0; ArBlockId < A.nBlocRow; ArBlockId++)    // for each line block of A
		{
			for (AColBlockIter AColIter = A.bRowBegin(ArBlockId); AColIter < A.bRowEnd(ArBlockId); AColIter++) // for each column  in row
			{
				ABlockConstAccessor ABlock = AColIter.bloc(); //take non zero blocks in row

				const Mat<M,N, Real1>& ABlockData = *(const Mat<M,N, Real1>*) ABlock.elements(AMatrixBuffer.ptr()); // get the block

				int AColIndex = ABlock.getCol(); // get the block colum index

				for (BColBlockIter BColIter = B.bRowBegin(AColIndex); BColIter < B.bRowEnd(AColIndex); BColIter++)
				{
					BBlockConstAccessor BBlock = BColIter.bloc();

					const Mat<N,K, Real1>& BBlockData = *(const Mat<N,K, Real1>*)BBlock.elements(BMatrixBuffer.ptr());

					int BColIndex = BBlock.getCol();

					Mat<M,K, Real1> RBlockData(0.0);
					//multiply the block, could be done more efficiently

                    for (unsigned int i = 0; i < M ; i++)
                        for (unsigned int j = 0; j < K; j++)
                            for (unsigned int k = 0; k < N; k++)
								RBlockData(i,j) += ABlockData(i,k)*BBlockData(k,j);

					R.blocAdd(ArBlockId, BColIndex, RBlockData.ptr());
				}
			}
		}
	}

	// CSR Matrices product (transposed) : R = A^t * B
	template<unsigned int M, unsigned int N, unsigned int K>
	static void multMatricesT(const CompressedRowSparseMatrix<Mat<M,N, Real1> >& At,
							  const CompressedRowSparseMatrix<Mat<M,K, Real1> >& B,
							  CompressedRowSparseMatrix<Mat<N,K, Real1> >& R)
	{
		size_t rBlockR = At.nBlocCol;
		size_t cBlockR = B.nBlocCol;

		R.resizeBloc(rBlockR, cBlockR);

		typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat<M,N, Real1> >::ColBlockConstIterator AColBlockIter;
		typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat<M,N, Real1> >::BlockConstAccessor ABlockConstAccessor;
		typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat<M,K, Real1> >::ColBlockConstIterator BColBlockIter;
		typedef typename sofa::component::linearsolver::CompressedRowSparseMatrix<Mat<M,K, Real1> >::BlockConstAccessor BBlockConstAccessor;

		Mat<M,N, Real1> AMatrixBuffer;
		Mat<M,K, Real1> BMatrixBuffer;

        for(size_t AtrBlockId = 0 ; AtrBlockId < (size_t)At.nBlocRow ; ++AtrBlockId)
		{
			for(AColBlockIter AtColIter = At.bRowBegin(AtrBlockId); AtColIter < At.bRowEnd(AtrBlockId) ; AtColIter++)
			{
				ABlockConstAccessor ABlock = AtColIter.bloc(); // is now a column block

				const Mat<M,N, Real1>& ABlockData = *(const Mat<M,N, Real1>*) ABlock.elements(AMatrixBuffer.ptr()); // get the block

				int AColIndex = ABlock.getCol(); // get the block colum index

//				for (BColBlockIter BColIter = B.bRowBegin(AColIndex); BColIter < B.bRowEnd(AColIndex); BColIter++) // modifier
				for (BColBlockIter BColIter = B.bRowBegin(AtrBlockId); BColIter < B.bRowEnd(AtrBlockId); BColIter++) // modifier
				{
					BBlockConstAccessor BBlock = BColIter.bloc();

					const Mat<M,K, Real1>& BBlockData = *(const Mat<M,K, Real1>*)BBlock.elements(BMatrixBuffer.ptr());

					int BColIndex = BBlock.getCol();

					Mat<N,K, Real1> RBlockData(0.0);
					//multiply the block, could be done more efficiently

                    for (unsigned int i = 0; i < N ; i++)
                        for (unsigned int j = 0; j < K; j++)
                            for (unsigned int k = 0; k < M; k++)
								RBlockData(i,j) += ABlockData(k,i)*BBlockData(k,j);

					R.blocAdd(AColIndex, BColIndex, RBlockData.ptr());
				}

//				for(size_t BBlockId = 0 ; BBlockId < B.nBlocRow ; ++BBlockId)
//				{
//					Mat<N,K, Real1> BBlock = B.blocGet(BBlockId, AColIndex);

//					Mat<N,K, Real1> RBlockData(0.0);
//					//multiply the block, could be done more efficiently

//					for (int i = 0; i < N ; i++)
//						for (int j = 0; j < K; j++)
//							for (int k = 0; k < M; k++)
//								RBlockData(i,j) += ABlockData(k,i)*BBlockData(k,j);

//					R->blocAdd(ArBlockId, BColIndex, RBlockData.ptr());
//				}
			}
		}
	}

public:

    virtual void addForce(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv1& /*f1*/, DataVecDeriv2& /*f2*/, const DataVecCoord1& /*x1*/, const DataVecCoord2& /*x2*/, const DataVecDeriv1& /*v1*/, const DataVecDeriv2& /*v2*/)
    {
//		std::cout << "x1 " << x1 << std::endl;
//		std::cout << "x2 " << x2 << std::endl;
//		std::cout << std::endl;
    }

    virtual void addDForce(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv1& /*df1*/, DataVecDeriv2& /*df2*/, const DataVecDeriv1& /*dx1*/, const DataVecDeriv2& /*dx2*/)
    {
//		std::cout << "dx1 " << dx1 << std::endl;
//		std::cout << "dx2 " << dx2 << std::endl;
//		std::cout << std::endl;
    }

    virtual void addKToMatrix(const sofa::core::MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix );

    virtual double getPotentialEnergy(const MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const DataVecCoord1& /*x1*/, const DataVecCoord2& /*x2*/) const
    {
        return 0.0;
    }

    void init();
    void reinit();

    void draw(const core::visual::VisualParams* /*vparams*/);

protected:

    Data< vector<unsigned> > indexPairs;                     ///< for each child, its parent and index in parent (two by two)



};

} // namespace interactionforcefield

} // namespace component

} // namespace sofa


#endif
