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
#ifndef SOFA_COMPONENT_FORCEFIELD_MappedMatrixForceFieldAndMass_INL
#define SOFA_COMPONENT_FORCEFIELD_MappedMatrixForceFieldAndMass_INL


#include "MappedMatrixForceFieldAndMass.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/rmath.h>
#include <assert.h>
#include <iostream>
#include <fstream>

// accumulate jacobian
#include <sofa/core/ExecParams.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>

// verify timing
#include <sofa/helper/system/thread/CTime.h>

//  Sparse Matrix
#include <Eigen/Sparse>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using sofa::component::linearsolver::DefaultMultiMatrixAccessor ;
using sofa::core::behavior::BaseMechanicalState ;

using sofa::core::objectmodel::ComponentState ;

template<class DataTypes1, class DataTypes2>
MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::MappedMatrixForceFieldAndMass()
    :
      l_mappedForceField(initLink("mappedForceField",
                                  "link to the forcefield that is mapped")),
      l_mappedForceField2(initLink("mappedForceField2",
                                   "link to a second forcefield that is mapped too (not mandatory)")),
      l_mappedMass(initLink("mappedMass",
                                   "link to a mass defined typically at the same node than mappedForceField"))
{
}

template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::init()
{
    m_componentstate = ComponentState::Invalid ;

    sofa::core::behavior::BaseInteractionForceField::init();

    if (mstate1.get() == NULL || mstate2.get() == NULL)
    {
        msg_error() << "Init of MappedMatrixForceFieldAndMass " << getContext()->getName() << " failed!" ;
        return;
    }

    m_childState = l_mappedForceField.get()->getContext()->getMechanicalState() ;
    if(m_childState==nullptr)
    {
        msg_error() << "There is no mechanical state in the context of the mappedForceField" ;
        return ;
    }

    m_componentstate = ComponentState::Valid ;
}

template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::buildIdentityBlocksInJacobian(core::behavior::BaseMechanicalState* mstate, sofa::core::MatrixDerivId Id)
{
    if(m_componentstate!=ComponentState::Valid)
        return ;

    msg_info() << "In buildIdentityBlocksInJacobianPasMOR, performECSW is false du coup " << mstate;
    sofa::helper::vector<unsigned int> list;
    for (unsigned int i=0; i<mstate->getSize(); i++)
        list.push_back(i);
    mstate->buildIdentityBlocksInJacobian(list, Id);
}

template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::accumulateJacobiansOptimized(const MechanicalParams* mparams)
{
    this->accumulateJacobians(mparams);
}

template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::accumulateJacobians(const MechanicalParams* mparams)
{
    if(m_componentstate!=ComponentState::Valid)
        return ;

    // STEP1 : accumulate Jacobians J1 and J2

    const core::ExecParams* eparams = dynamic_cast<const core::ExecParams *>( mparams );
    core::ConstraintParams cparams = core::ConstraintParams(*eparams);

    sofa::core::MatrixDerivId Id= sofa::core::MatrixDerivId::mappingJacobian();
    core::objectmodel::BaseContext* context = this->getContext();
    simulation::Node* gnode = dynamic_cast<simulation::Node*>(context);
    simulation::MechanicalResetConstraintVisitor(eparams).execute(gnode);

    buildIdentityBlocksInJacobian(m_childState, Id);

    MechanicalAccumulateJacobian(&cparams, core::MatrixDerivId::mappingJacobian()).execute(gnode);

}

template<class T>
void copyKToEigenFormat(CompressedRowSparseMatrix< T >* K, Eigen::SparseMatrix<double,Eigen::ColMajor>& Keig)
{
    Keig.resize(K->nRow,K->nRow);
    std::vector< Eigen::Triplet<double> > tripletList;
    tripletList.reserve(K->colsValue.size());

    int row;
    for (unsigned int it_rows_k=0; it_rows_k < K->rowIndex.size() ; it_rows_k ++)
    {
        row = K->rowIndex[it_rows_k] ;
        typename CompressedRowSparseMatrix<T>::Range rowRange( K->rowBegin[it_rows_k], K->rowBegin[it_rows_k+1] );
        for(sofa::defaulttype::BaseVector::Index xj = rowRange.begin() ; xj < rowRange.end() ; xj++ )  // for each non-null block
        {
            int col = K->colsIndex[xj];     // block column
            const T& k = K->colsValue[xj]; // non-null element of the matrix
            tripletList.push_back(Eigen::Triplet<double>(row,col,k));
        }
    }
    Keig.setFromTriplets(tripletList.begin(), tripletList.end());
}

template<class InputFormat>
static void copyMappingJacobianToEigenFormat(const typename InputFormat::MatrixDeriv& J, Eigen::SparseMatrix<double>& Jeig)
{
    typedef typename InputFormat::MatrixDeriv::RowConstIterator RowConstIterator;
    typedef typename InputFormat::MatrixDeriv::ColConstIterator ColConstIterator;
    typedef typename InputFormat::Deriv Deriv;
    int DerivSize = InputFormat::Deriv::total_size;
    int nbRowsJ = Jeig.rows();
    int maxRowIndex = 0, maxColIndex = 0;
    std::vector< Eigen::Triplet<double> > tripletListJ;

    for (RowConstIterator rowIt = J.begin(); rowIt !=  J.end(); ++rowIt)
    {
        int rowIndex = rowIt.index();
        if (rowIndex>maxRowIndex)
            maxRowIndex = rowIndex;
        for (ColConstIterator colIt = rowIt.begin(); colIt !=  rowIt.end(); ++colIt)
        {
            int colIndex = colIt.index();
            Deriv elemVal = colIt.val();
            for (int i=0;i<DerivSize;i++)
            {
                tripletListJ.push_back(Eigen::Triplet<double>(rowIndex,DerivSize*colIndex + i,elemVal[i]));
                if (colIndex>maxColIndex)
                        maxColIndex = colIndex;
            }
        }
    }
    Jeig.resize(nbRowsJ,DerivSize*(maxColIndex+1));
    Jeig.reserve(J.size());
    Jeig.setFromTriplets(tripletListJ.begin(), tripletListJ.end());
}

template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::optimizeAndCopyMappingJacobianToEigenFormat1(const typename DataTypes1::MatrixDeriv& J, Eigen::SparseMatrix<double>& Jeig)
{
    copyMappingJacobianToEigenFormat<DataTypes1>(J, Jeig);
}

template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::optimizeAndCopyMappingJacobianToEigenFormat2(const typename DataTypes2::MatrixDeriv& J, Eigen::SparseMatrix<double>& Jeig)
{
    copyMappingJacobianToEigenFormat<DataTypes2>(J, Jeig);
}

template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::addMassToSystem(const MechanicalParams* mparams, const DefaultMultiMatrixAccessor* KAccessor)
{
    if (l_mappedMass != NULL)
    {
        l_mappedMass.get()->addMToMatrix(mparams, KAccessor);
    }
    else
    {
        msg_info() << "There is no d_mappedMass";
    }
}


template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::addPrecomputedMassToSystem(const MechanicalParams* mparams, const unsigned int mstateSize,const Eigen::SparseMatrix<double> &Jeig, Eigen::SparseMatrix<double> &JtKJeig)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(mstateSize);
    SOFA_UNUSED(Jeig);
    SOFA_UNUSED(JtKJeig);
}




template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::addKToMatrix(const MechanicalParams* mparams,
                                                                         const MultiMatrixAccessor* matrix)
{
    if(m_componentstate!=ComponentState::Valid)
        return ;

    sofa::helper::system::thread::CTime *timer;
    double timeScale, time, totime ;
    timeScale = 1000.0 / (double)sofa::helper::system::thread::CTime::getTicksPerSec();

    time = (double)timer->getTime();
    totime = (double)timer->getTime();

    sofa::core::behavior::MechanicalState<DataTypes1>* ms1 = this->getMState1();
    sofa::core::behavior::MechanicalState<DataTypes2>* ms2 = this->getMState2();


    sofa::core::behavior::BaseMechanicalState*  bms1 = this->getMechModel1();
    sofa::core::behavior::BaseMechanicalState*  bms2 = this->getMechModel2();


    MultiMatrixAccessor::MatrixRef mat11 = matrix->getMatrix(mstate1);
    MultiMatrixAccessor::MatrixRef mat22 = matrix->getMatrix(mstate2);
    MultiMatrixAccessor::InteractionMatrixRef mat12 = matrix->getMatrix(mstate1, mstate2);
    MultiMatrixAccessor::InteractionMatrixRef mat21 = matrix->getMatrix(mstate2, mstate1);


    ///////////////////////////     STEP 1      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*              compute jacobians using generic implementation                */
    /* -------------------------------------------------------------------------- */
    time= (double)timer->getTime();
    accumulateJacobiansOptimized(mparams);
    msg_info() <<" accumulate J : "<<( (double)timer->getTime() - time)*timeScale<<" ms";



    ///////////////////////////     STEP 2      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*  compute the stiffness K of the forcefield and put it in a rowsparseMatrix */
    /*          get the stiffness matrix from the mapped ForceField               */
    /* TODO: use the template of the FF for Real                                  */
    /* -------------------------------------------------------------------------- */


    ///////////////////////     GET K       ////////////////////////////////////////
    CompressedRowSparseMatrix< Real1 >* K = new CompressedRowSparseMatrix< Real1 > ( );

    K->resizeBloc( 3*m_childState->getSize() ,  3*m_childState->getSize());
    K->clear();
    DefaultMultiMatrixAccessor* KAccessor;
    KAccessor = new DefaultMultiMatrixAccessor;
    KAccessor->addMechanicalState(  m_childState );
    KAccessor->setGlobalMatrix(K);
    KAccessor->setupMatrices();


    //------------------------------------------------------------------------------


    msg_info()<<" time get K : "<<( (double)timer->getTime() - time)*timeScale<<" ms";
    time= (double)timer->getTime();


    l_mappedForceField.get()->addKToMatrix(mparams, KAccessor);
    if (l_mappedForceField2 != NULL)
    {
        l_mappedForceField2.get()->addKToMatrix(mparams, KAccessor);
    }
    addMassToSystem(mparams,KAccessor);
    msg_info()<<" time addKtoMatrix K : "<<( (double)timer->getTime() - time)*timeScale<<" ms";
    time= (double)timer->getTime();

    if (!K)
    {
        msg_error(this) << "matrix of the force-field system not found";
        return;
    }


    ///////////////////////     COMPRESS K       ///////////////////////////////////
    K->compress();
    //------------------------------------------------------------------------------


    msg_info() << " time compress K : "<<( (double)timer->getTime() - time)*timeScale<<" ms";

    ///////////////////////////     STEP 3      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*                  we now get the matrices J1 and J2                         */
    /* -------------------------------------------------------------------------- */


    msg_info()<<" nRow: "<< K->nRow << " nCol: " << K->nCol;


    ///////////////////////////     STEP 4      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*          perform the multiplication with [J1t J2t] * K * [J1 J2]           */
    /* -------------------------------------------------------------------------- */

    double startTime= (double)timer->getTime();
    Eigen::SparseMatrix<double,Eigen::ColMajor> Keig;
    copyKToEigenFormat(K,Keig);
    msg_info() << "Keig size:" << Keig.size();
    msg_info() << "Keig rows:" << Keig.rows();
    msg_info() << "Keig cols:" << Keig.cols();

    //--------------------------------------------------------------------------------------------------------------------

    msg_info()<<" time set Keig : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";
    startTime= (double)timer->getTime();

    ///////////////////////    COPY J1 AND J2 IN EIGEN FORMAT //////////////////////////////////////
    sofa::core::MultiMatrixDerivId c = sofa::core::MatrixDerivId::mappingJacobian();
    const MatrixDeriv1 &J1 = c[ms1].read()->getValue();
    const MatrixDeriv2 &J2 = c[ms2].read()->getValue();

    Eigen::SparseMatrix<double> J1eig;
    Eigen::SparseMatrix<double> J2eig;
    J1eig.resize(K->nRow, J1.begin().row().size()*DerivSize1);
    unsigned int nbColsJ1 = 0, nbColsJ2 = 0;

    optimizeAndCopyMappingJacobianToEigenFormat1(J1, J1eig);
    if (bms1 != bms2)
    {
        double startTime2= (double)timer->getTime();
        J2eig.resize(K->nRow, J2.begin().row().size()*DerivSize2);
        optimizeAndCopyMappingJacobianToEigenFormat2(J2, J2eig);
        msg_info()<<" time set J2eig alone : "<<( (double)timer->getTime() - startTime2)*timeScale<<" ms";
    }

    msg_info()<<" time getJ + set J1eig (and potentially J2eig) : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";
    startTime= (double)timer->getTime();

    ///////////////////////     J1t * K * J1    //////////////////////////////////////////////////////////////////////////
    nbColsJ1 = J1eig.cols();
    if (bms1 != bms2)
    {
        nbColsJ2 = J2eig.cols();
    }
    Eigen::SparseMatrix<double>  J1tKJ1eigen(nbColsJ1,nbColsJ1);

    J1tKJ1eigen = J1eig.transpose()*Keig*J1eig;


    msg_info()<<" time compute J1tKJ1eigen alone : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";

    Eigen::SparseMatrix<double>  J2tKJ2eigen(nbColsJ2,nbColsJ2);
    Eigen::SparseMatrix<double>  J1tKJ2eigen(nbColsJ1,nbColsJ2);
    Eigen::SparseMatrix<double>  J2tKJ1eigen(nbColsJ2,nbColsJ1);

    if (bms1 != bms2)
    {
        double startTime2= (double)timer->getTime();
        J2tKJ2eigen = J2eig.transpose()*Keig*J2eig;
        J1tKJ2eigen = J1eig.transpose()*Keig*J2eig;
        J2tKJ1eigen = J2eig.transpose()*Keig*J1eig;
        msg_info()<<" time compute J1tKJ2eigen J2TKJ2 and J2tKJ1 : "<<( (double)timer->getTime() - startTime2)*timeScale<<" ms";
    }

    //--------------------------------------------------------------------------------------------------------------------

    msg_info()<<" time compute all JtKJeigen with J1eig and J2eig : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";
    unsigned int m_childStateSize = m_childState->getSize();
    addPrecomputedMassToSystem(mparams,m_childStateSize,J1eig,J1tKJ1eigen);
    int offset,offrow, offcol;
    startTime= (double)timer->getTime();
    offset = mat11.offset;
    for (int k=0; k<J1tKJ1eigen.outerSize(); ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(J1tKJ1eigen,k); it; ++it)
      {
              mat11.matrix->add(offset + it.row(),offset + it.col(), it.value());
      }
    msg_info()<<" time copy J1tKJ1eigen back to J1tKJ1 in CompressedRowSparse : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";

    if (bms1 != bms2)
    {
        startTime= (double)timer->getTime();
        offset = mat22.offset;
        for (int k=0; k<J2tKJ2eigen.outerSize(); ++k)
          for (Eigen::SparseMatrix<double>::InnerIterator it(J2tKJ2eigen,k); it; ++it)
          {
                  mat22.matrix->add(offset + it.row(),offset + it.col(), it.value());
          }
        msg_info()<<" time copy J2tKJ2eigen back to J2tKJ2 in CompressedRowSparse : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";
        startTime= (double)timer->getTime();
        offrow = mat12.offRow;
        offcol = mat12.offCol;
        for (int k=0; k<J1tKJ2eigen.outerSize(); ++k)
          for (Eigen::SparseMatrix<double>::InnerIterator it(J1tKJ2eigen,k); it; ++it)
          {
                  mat22.matrix->add(offrow + it.row(),offcol + it.col(), it.value());
          }
        msg_info()<<" time copy J1tKJ2eigen back to J1tKJ2 in CompressedRowSparse : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";
        startTime= (double)timer->getTime();
        offrow = mat21.offRow;
        offcol = mat21.offCol;
        for (int k=0; k<J2tKJ1eigen.outerSize(); ++k)
          for (Eigen::SparseMatrix<double>::InnerIterator it(J2tKJ1eigen,k); it; ++it)
          {
                  mat21.matrix->add(offrow + it.row(),offcol + it.col(), it.value());
          }
        msg_info()<<" time copy J2tKJ1eigen back to J2tKJ1 in CompressedRowSparse : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";

    }

    msg_info()<<" total time compute J() * K * J: "<<( (double)timer->getTime() - totime)*timeScale<<" ms";
    delete KAccessor;
    delete K;


    msg_info() << "EXIT addKToMatrix" ;


    const core::ExecParams* eparams = dynamic_cast<const core::ExecParams *>( mparams );
    core::ConstraintParams cparams = core::ConstraintParams(*eparams);

    core::objectmodel::BaseContext* context = this->getContext();
    simulation::Node* gnode = dynamic_cast<simulation::Node*>(context);
    simulation::MechanicalResetConstraintVisitor(eparams).execute(gnode);

}



// Even though it does nothing, this method has to be implemented
// since it's a pure virtual in parent class
template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::addForce(const MechanicalParams* mparams,
                                                              DataVecDeriv1& f1,
                                                              DataVecDeriv2& f2,
                                                              const DataVecCoord1& x1,
                                                              const DataVecCoord2& x2,
                                                              const DataVecDeriv1& v1,
                                                              const DataVecDeriv2& v2)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(f1);
    SOFA_UNUSED(f2);
    SOFA_UNUSED(x1);
    SOFA_UNUSED(x2);
    SOFA_UNUSED(v1);
    SOFA_UNUSED(v2);
}

// Even though it does nothing, this method has to be implemented
// since it's a pure virtual in parent class
template<class DataTypes1, class DataTypes2>
void MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::addDForce(const MechanicalParams* mparams,
                                                               DataVecDeriv1& df1,
                                                               DataVecDeriv2& df2,
                                                               const DataVecDeriv1& dx1,
                                                               const DataVecDeriv2& dx2)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(df1);
    SOFA_UNUSED(df2);
    SOFA_UNUSED(dx1);
    SOFA_UNUSED(dx2);
}

// Even though it does nothing, this method has to be implemented
// since it's a pure virtual in parent class
template<class DataTypes1, class DataTypes2>
double MappedMatrixForceFieldAndMass<DataTypes1, DataTypes2>::getPotentialEnergy(const MechanicalParams* mparams,
                                                                          const DataVecCoord1& x1,
                                                                          const DataVecCoord2& x2) const
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x1);
    SOFA_UNUSED(x2);

    return 0.0;
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
