/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2018 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ANIMATIONLOOP_MECHANICALMATRIXMAPPER_INL
#define SOFA_COMPONENT_ANIMATIONLOOP_MECHANICALMATRIXMAPPER_INL

#include "MechanicalMatrixMapper.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/rmath.h>

// accumulate jacobian
#include <sofa/core/ExecParams.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>

// verify timing
#include <sofa/helper/system/thread/CTime.h>

//  Eigen Sparse Matrix
#include <Eigen/Sparse>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes1, class DataTypes2>
MechanicalMatrixMapper<DataTypes1, DataTypes2>::MechanicalMatrixMapper()
    :
      d_forceFieldList(initData(&d_forceFieldList,"forceFieldList","List of ForceField Names to work on (by default will take all)")),
      l_nodeToParse(initLink("nodeToParse","link to the node on which the component will work, from this link the mechanicalState/mass/forceField links will be made")),
      l_mechanicalState(initLink("mechanicalState","The mechanicalState with which the component will work on (filled automatically during init)")),
      l_mappedMass(initLink("mass","mass with which the component will work on (filled automatically during init)")),
      l_forceField(initLink("forceField","The ForceField(s) attached to this node (filled automatically during init)"))
{
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::init()
{
    if(m_componentstate==ComponentState::Valid){
        msg_warning() << "Calling an already fully initialized component. You should use reinit instead." ;
    }

    if(l_nodeToParse.get() == NULL)
    {
        msg_error() << " failed to initialized -> missing/wrong link " << l_nodeToParse.getName() << " : " << l_nodeToParse.getLinkedPath() << sendl;
        m_componentstate = ComponentState::Invalid ;
        return;
    }

    sofa::core::behavior::BaseInteractionForceField::init();

    if (mstate1.get() == NULL || mstate2.get() == NULL)
    {
        msg_error() << " failed to initialized -> missing/wrong link " << mstate1.getName() << " or " << mstate2.getName() << sendl;
        m_componentstate = ComponentState::Invalid ;
        return;
    }


    // Add link to mass & and get mass component name to rm it from forcefields
    std::string massName;
    if (l_nodeToParse.get()->mass)
    {
        l_mappedMass.add(l_nodeToParse.get()->mass,l_nodeToParse.get()->mass->getPathName());
        massName.append(l_nodeToParse.get()->mass->getName());
    }

    // Add link to  mechanical
    if (l_nodeToParse.get()->mechanicalState)
    {
        l_mechanicalState.add(l_nodeToParse.get()->mechanicalState,l_nodeToParse.get()->mechanicalState->getPathName());
    }
    else
    {
        msg_error() << ": no mechanical object to link to for this node path: " << l_nodeToParse.getPath();
        m_componentstate = ComponentState::Invalid ;
        return;
    }

    // Parse l_nodeToParse to find & link with the forcefields
    parseNode(l_nodeToParse.get(),massName);
    m_nbInteractionForceFields = l_nodeToParse.get()->interactionForceField.size();

    if (l_forceField.size() == 0)
    {
        msg_error() << ": no forcefield to link to for this node path: " << l_nodeToParse.getPath();
        m_componentstate = ComponentState::Invalid ;
        return;
    }

    m_componentstate = ComponentState::Valid ;
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::parseNode(sofa::simulation::Node *node,std::string massName)
{
//    for(unsigned int i=0; i<l_forceField.size(); i++)
//    {
//        l_forceField.remove(l_forceField[i]);
//    }
    bool empty = d_forceFieldList.getValue().empty();
    for(unsigned int i=0; i<node->forceField.size(); i++)
    {
        if (node->forceField[i]->name != massName)
        {
            bool found = true;
            if (!empty)
                found = (std::find(d_forceFieldList.getValue().begin(), d_forceFieldList.getValue().end(), node->forceField[i]->getName()) != d_forceFieldList.getValue().end());

            if(found)
            {
                l_forceField.add(node->forceField[i],node->forceField[i]->getPathName());
            }
        }
    }
    for(unsigned int i=0; i<node->interactionForceField.size(); i++)
    {

        bool found = true;
        if (!empty)
            found = (std::find(d_forceFieldList.getValue().begin(), d_forceFieldList.getValue().end(), node->interactionForceField[i]->getName()) != d_forceFieldList.getValue().end());

        if(found)
        {
            l_forceField.add(node->interactionForceField[i],node->interactionForceField[i]->getPathName());
        }

    }
//    for(unsigned int i=0; i<node->interactionForceField.size(); i++)
//    {

//        l_forceField.add(node->interactionForceField[i],node->interactionForceField[i]->getPathName());
//    }
    for (sofa::simulation::Node::ChildIterator it = node->child.begin(), itend = node->child.end(); it != itend; ++it)
    {
        parseNode(it->get(),massName);
    }
    return;
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::buildIdentityBlocksInJacobian(core::behavior::BaseMechanicalState* mstate, sofa::core::MatrixDerivId Id)
{
    sofa::helper::vector<unsigned int> list;
    for (unsigned int i=0; i<mstate->getSize(); i++)
        list.push_back(i);
    mstate->buildIdentityBlocksInJacobian(list, Id);
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::accumulateJacobiansOptimized(const MechanicalParams* mparams)
{
    this->accumulateJacobians(mparams);
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::accumulateJacobians(const MechanicalParams* mparams)
{
    const core::ExecParams* eparams = dynamic_cast<const core::ExecParams *>( mparams );
    core::ConstraintParams cparams = core::ConstraintParams(*eparams);

    sofa::core::MatrixDerivId Id= sofa::core::MatrixDerivId::mappingJacobian();
    core::objectmodel::BaseContext* context = this->getContext();
    simulation::Node* gnode = dynamic_cast<simulation::Node*>(context);
    simulation::MechanicalResetConstraintVisitor(&cparams).execute(context);
    buildIdentityBlocksInJacobian(l_mechanicalState,Id);

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
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::optimizeAndCopyMappingJacobianToEigenFormat1(const typename DataTypes1::MatrixDeriv& J, Eigen::SparseMatrix<double>& Jeig)
{
    copyMappingJacobianToEigenFormat<DataTypes1>(J, Jeig);
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::optimizeAndCopyMappingJacobianToEigenFormat2(const typename DataTypes2::MatrixDeriv& J, Eigen::SparseMatrix<double>& Jeig)
{
    copyMappingJacobianToEigenFormat<DataTypes2>(J, Jeig);
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::addMassToSystem(const MechanicalParams* mparams, const DefaultMultiMatrixAccessor* KAccessor)
{
    if (l_mappedMass != NULL)
    {
        l_mappedMass->addMToMatrix(mparams, KAccessor);
    }
    else
    {
        msg_info() << "There is no mappedMass";
    }
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::addPrecomputedMassToSystem(const MechanicalParams* mparams, const unsigned int mstateSize,const Eigen::SparseMatrix<double> &Jeig, Eigen::SparseMatrix<double> &JtKJeig)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(mstateSize);
    SOFA_UNUSED(Jeig);
    SOFA_UNUSED(JtKJeig);
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::addKToMatrix(const MechanicalParams* mparams,
                                                                        const MultiMatrixAccessor* matrix)
{
    if(m_componentstate != ComponentState::Valid)
        return ;

    sofa::helper::system::thread::CTime *timer;
    double timeScale, time, totime ;
    timeScale = 1000.0 / (double)sofa::helper::system::thread::CTime::getTicksPerSec();

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
    msg_info(this) <<" accumulate J : "<<( (double)timer->getTime() - time)*timeScale<<" ms";

    ///////////////////////////     STEP 2      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*  compute the stiffness K of the forcefield and put it in a rowsparseMatrix */
    /*          get the stiffness matrix from the mapped ForceField               */
    /* TODO: use the template of the FF for Real                                  */
    /* -------------------------------------------------------------------------- */


    ///////////////////////     GET K       ////////////////////////////////////////
    CompressedRowSparseMatrix< Real1 >* K = new CompressedRowSparseMatrix< Real1 > ( );

    K->resizeBloc( 3*l_mechanicalState->getSize() ,  3*l_mechanicalState->getSize());
    K->clear();
    DefaultMultiMatrixAccessor* KAccessor;
    KAccessor = new DefaultMultiMatrixAccessor;
    KAccessor->addMechanicalState( l_mechanicalState );
    KAccessor->setGlobalMatrix(K);
    KAccessor->setupMatrices();

    time= (double)timer->getTime();

    sofa::simulation::Node *node = l_nodeToParse.get();
    unsigned int currentNbInteractionFFs = node->interactionForceField.size();
    for (int i;i<node->interactionForceField.size();i++)
    {
        msg_warning()<< "the interactionFFs:" << node->interactionForceField[i]->getName();
    }
    for (int i;i<node->forceField.size();i++)
    {
        msg_warning()<< "the FFs:" << node->forceField[i]->getName();
    }
    msg_warning()<< "currentNbInteractionFFs:" << currentNbInteractionFFs << " m_nbInteractionForceFields:" << m_nbInteractionForceFields;
    for(unsigned int i=0; i<l_forceField.size(); i++)
    {
        msg_warning()<< "the registered FFs:" << l_forceField[i]->getName();
    }
    if (m_nbInteractionForceFields != currentNbInteractionFFs)
    {
        parseNode(l_nodeToParse.get(),l_nodeToParse.get()->mass->getName());
        m_nbInteractionForceFields = currentNbInteractionFFs;
    }
    msg_warning()<< "plop";

    for(unsigned int i=0; i<l_forceField.size(); i++)
    {
        msg_warning()<< "adding forceField:" << i;
        //msg_error() << "nb forceField" << i;
        l_forceField[i]->addKToMatrix(mparams, KAccessor);
    }
    msg_warning()<< "plap";

    addMassToSystem(mparams,KAccessor);
    msg_warning()<< "plip";

    if (!K)
    {
        msg_error(this) << "matrix of the force-field system not found";
        return;
    }

    msg_info(this)<<" time addKtoMatrix K : "<<( (double)timer->getTime() - time)*timeScale<<" ms";

    ///////////////////////     COMPRESS K       ///////////////////////////////////
    time= (double)timer->getTime();
    K->compress();
    msg_info(this) << " time compress K : "<<( (double)timer->getTime() - time)*timeScale<<" ms";

    //------------------------------------------------------------------------------

    time = (double)timer->getTime();
    Eigen::SparseMatrix<double,Eigen::ColMajor> Keig;
    copyKToEigenFormat(K,Keig);
    msg_info(this)<<" time set Keig : "<<( (double)timer->getTime() - time)*timeScale<<" ms";


    ///////////////////////    COPY J1 AND J2 IN EIGEN FORMAT //////////////////////////////////////
    double startTime= (double)timer->getTime();
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
        msg_info(this)<<" time set J2eig alone : "<<( (double)timer->getTime() - startTime2)*timeScale<<" ms";
    }

    msg_info(this)<<" time getJ + set J1eig (and potentially J2eig) : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";
    startTime= (double)timer->getTime();

    ///////////////////////////     STEP 4      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*          perform the multiplication with [J1t J2t] * K * [J1 J2]           */
    /* -------------------------------------------------------------------------- */
    nbColsJ1 = J1eig.cols();
    if (bms1 != bms2)
    {
        nbColsJ2 = J2eig.cols();
    }
    Eigen::SparseMatrix<double>  J1tKJ1eigen(nbColsJ1,nbColsJ1);

    J1tKJ1eigen = J1eig.transpose()*Keig*J1eig;

    msg_info(this)<<" time compute J1tKJ1eigen alone : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";

    Eigen::SparseMatrix<double>  J2tKJ2eigen(nbColsJ2,nbColsJ2);
    Eigen::SparseMatrix<double>  J1tKJ2eigen(nbColsJ1,nbColsJ2);
    Eigen::SparseMatrix<double>  J2tKJ1eigen(nbColsJ2,nbColsJ1);

    if (bms1 != bms2)
    {
        double startTime2= (double)timer->getTime();
        J2tKJ2eigen = J2eig.transpose()*Keig*J2eig;
        J1tKJ2eigen = J1eig.transpose()*Keig*J2eig;
        J2tKJ1eigen = J2eig.transpose()*Keig*J1eig;
        msg_info(this)<<" time compute J1tKJ2eigen J2TKJ2 and J2tKJ1 : "<<( (double)timer->getTime() - startTime2)*timeScale<<" ms";

    }

    //--------------------------------------------------------------------------------------------------------------------

    msg_info(this)<<" time compute all JtKJeigen with J1eig and J2eig : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";
    //int row;
    unsigned int mstateSize = l_mechanicalState->getSize();
    addPrecomputedMassToSystem(mparams,mstateSize,J1eig,J1tKJ1eigen);
    int offset,offrow, offcol;
    startTime= (double)timer->getTime();
    offset = mat11.offset;
    for (int k=0; k<J1tKJ1eigen.outerSize(); ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(J1tKJ1eigen,k); it; ++it)
      {
              mat11.matrix->add(offset + it.row(),offset + it.col(), it.value());
      }
    msg_info(this)<<" time copy J1tKJ1eigen back to J1tKJ1 in CompressedRowSparse : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";

    if (bms1 != bms2)
    {
        startTime= (double)timer->getTime();
        offset = mat22.offset;
        for (int k=0; k<J2tKJ2eigen.outerSize(); ++k)
          for (Eigen::SparseMatrix<double>::InnerIterator it(J2tKJ2eigen,k); it; ++it)
          {
                  mat22.matrix->add(offset + it.row(),offset + it.col(), it.value());
          }
        msg_info(this)<<" time copy J2tKJ2eigen back to J2tKJ2 in CompressedRowSparse : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";
        startTime= (double)timer->getTime();
        offrow = mat12.offRow;
        offcol = mat12.offCol;
        for (int k=0; k<J1tKJ2eigen.outerSize(); ++k)
          for (Eigen::SparseMatrix<double>::InnerIterator it(J1tKJ2eigen,k); it; ++it)
          {
                  mat22.matrix->add(offrow + it.row(),offcol + it.col(), it.value());
          }
        msg_info(this)<<" time copy J1tKJ2eigen back to J1tKJ2 in CompressedRowSparse : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";
        startTime= (double)timer->getTime();
        offrow = mat21.offRow;
        offcol = mat21.offCol;
        for (int k=0; k<J2tKJ1eigen.outerSize(); ++k)
          for (Eigen::SparseMatrix<double>::InnerIterator it(J2tKJ1eigen,k); it; ++it)
          {
                  mat21.matrix->add(offrow + it.row(),offcol + it.col(), it.value());
          }
        msg_info(this)<<" time copy J2tKJ1eigen back to J2tKJ1 in CompressedRowSparse : "<<( (double)timer->getTime() - startTime)*timeScale<<" ms";

    }

    msg_info(this)<<" total time compute J() * K * J: "<<( (double)timer->getTime() - totime)*timeScale<<" ms";
    delete KAccessor;
    delete K;


    if(f_printLog.getValue())
        sout << "EXIT addKToMatrix\n" << sendl;


    const core::ExecParams* eparams = dynamic_cast<const core::ExecParams *>( mparams );
    core::ConstraintParams cparams = core::ConstraintParams(*eparams);

    core::objectmodel::BaseContext* context = this->getContext();
    simulation::MechanicalResetConstraintVisitor(&cparams).execute(context);

}

// Even though it does nothing, this method has to be implemented
// since it's a pure virtual in parent class
template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::addForce(const MechanicalParams* mparams,
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
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::addDForce(const MechanicalParams* mparams,
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
double MechanicalMatrixMapper<DataTypes1, DataTypes2>::getPotentialEnergy(const MechanicalParams* mparams,
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
