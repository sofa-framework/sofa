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

#include <sofa/component/mapping/mappedmatrix/MechanicalMatrixMapper.h>
#ifndef SOFA_BUILD_SOFA_COMPONENT_MAPPING_MAPPEDMATRIX
SOFA_DEPRECATED_HEADER_NOT_REPLACED("v23.06", "v23.12")
#endif

#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/linearalgebra/SparseMatrixProduct[EigenSparseMatrix].h>

#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalResetConstraintVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalResetConstraintVisitor;
#include <sofa/simulation/mechanicalvisitor/MechanicalAccumulateJacobian.h>

#include <sofa/core/MechanicalParams.h>

#include <sofa/simulation/CpuTask.h>
#include <sofa/simulation/TaskScheduler.h>

// verify timing
#include <sofa/helper/system/thread/CTime.h>

//  Eigen Sparse Matrix
#include <Eigen/Sparse>

#include <sofa/simulation/Node.h>


namespace sofa::component::mapping::mappedmatrix
{
template <typename TDataTypes1, typename TDataTypes2>
void MechanicalMatrixMapper<TDataTypes1, TDataTypes2>::computeMatrixProduct(
    const bool fastProduct,
    JtKMatrixProduct& product_1,
    linearalgebra::SparseMatrixProduct<Eigen::SparseMatrix<double> >& product_2,
    const Eigen::SparseMatrix<double>* J1, const Eigen::SparseMatrix<double>* J2,
    const Eigen::SparseMatrix<double>* K,
    Eigen::SparseMatrix<double>*& output)
{
    if (fastProduct)
    {
        // product_1 is involved in multiple products. It can be reused if already computed
        if (!product_1.isComputed)
        {
            const Eigen::SparseMatrix<double> Jt = J1->transpose();
            product_1.product.matrixA = &Jt;
            product_1.product.matrixB = K;
            product_1.product.computeProduct();
            product_1.isComputed = true;
        }

        product_2.matrixA = &product_1.product.getProductResult();
        product_2.matrixB = J2;
        product_2.computeProduct();

        output = const_cast<Eigen::SparseMatrix<double>*>(&product_2.getProductResult());
    }
    else
    {
        if (!product_1.isComputed)
        {
            const Eigen::SparseMatrix<double> Jt = J1->transpose();
            product_1.matrix.resize(Jt.rows(), K->cols());
            product_1.matrix = Jt * (*K);
        }
        output->resize(J1->cols(), J2->cols());
        *output = product_1.matrix * (*J2);
    }
}

template<class DataTypes1, class DataTypes2>
MechanicalMatrixMapper<DataTypes1, DataTypes2>::MechanicalMatrixMapper()
    :
      d_yesIKnowMatrixMappingIsSupportedAutomatically(initData(&d_yesIKnowMatrixMappingIsSupportedAutomatically, false, "yesIKnowMatrixMappingIsSupportedAutomatically", "If true the component is activated, otherwise it is deactivated.\nThis Data is used to explicitly state that the component must be used even though matrix mapping is now supported automatically, without MechanicalMatrixMapper.")),
      d_forceFieldList(initData(&d_forceFieldList,"forceFieldList","List of ForceField Names to work on (by default will take all)")),
      l_nodeToParse(initLink("nodeToParse","link to the node on which the component will work, from this link the mechanicalState/mass/forceField links will be made")),
      d_stopAtNodeToParse(initData(&d_stopAtNodeToParse,false,"stopAtNodeToParse","Boolean to choose whether forceFields in children Nodes of NodeToParse should be considered.")),
      d_skipJ1tKJ1(initData(&d_skipJ1tKJ1,false,"skipJ1tKJ1","Boolean to choose whether to skip J1tKJ1 to avoid 2 contributions, in case 2 MechanicalMatrixMapper are used")),
      d_skipJ2tKJ2(initData(&d_skipJ2tKJ2,false,"skipJ2tKJ2","Boolean to choose whether to skip J2tKJ2 to avoid 2 contributions, in case 2 MechanicalMatrixMapper are used")),
      d_fastMatrixProduct(initData(&d_fastMatrixProduct, true, "fastMatrixProduct", "If true, an accelerated method to compute matrix products based on the pre-computation of the matrices intersection is used. Regular matrix product otherwise.")),
      d_parallelTasks(initData(&d_parallelTasks, true, "parallelTasks", "Execute some tasks in parallel for better performances")),
      d_forceFieldAndMass(initData(&d_forceFieldAndMass, false, "forceFieldAndMass", "If true, allows forceField and mass to be in the same component.")),
      l_mechanicalState(initLink("mechanicalState","The mechanicalState with which the component will work on (filled automatically during init)")),
      l_mappedMass(initLink("mass","mass with which the component will work on (filled automatically during init)")),
      l_forceField(initLink("forceField","The ForceField(s) attached to this node (filled automatically during init)"))
{
}

template <typename TDataTypes1, typename TDataTypes2>
void MechanicalMatrixMapper<TDataTypes1, TDataTypes2>::parse(
    core::objectmodel::BaseObjectDescription* arg)
{
    Inherit1::parse(arg);
    if (!arg->getAttribute("yesIKnowMatrixMappingIsSupportedAutomatically", nullptr))
    {
        msg_warning() << "Matrix mapping is now supported automatically. Therefore, "
            << this->getClassName() << " is no longer necessary. Remove it from your scene.";
    }
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::init()
{
    if (!d_yesIKnowMatrixMappingIsSupportedAutomatically.getValue())
    {
        msg_error() << "This component is deprecated and deactivated because matrix mapping is now supported automatically";
        this->d_componentState.setValue(ComponentState::Invalid);
        return;
    }

    if(this->d_componentState.getValue() == ComponentState::Valid){
        msg_warning() << "Calling an already fully initialized component. You should use reinit instead." ;
    }

    if(l_nodeToParse.get() == nullptr)
    {
        msg_error() << " failed to initialized -> missing/wrong link " << l_nodeToParse.getName() << " : " << l_nodeToParse.getLinkedPath();
        this->d_componentState.setValue(ComponentState::Invalid) ;
        return;
    }

    Inherit1::init();

    if (mstate1.get() == nullptr || mstate2.get() == nullptr)
    {
        msg_error() << " failed to initialized -> missing/wrong link " << mstate1.getName() << " or " << mstate2.getName();
        this->d_componentState.setValue(ComponentState::Invalid) ;
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
        this->d_componentState.setValue(ComponentState::Invalid) ;
        return;
    }

    // Parse l_nodeToParse to find & link with the forcefields
    parseNode(l_nodeToParse.get(),massName);
    m_nbInteractionForceFields = l_nodeToParse.get()->interactionForceField.size();

    if (l_forceField.size() == 0)
    {
        msg_warning() << ": no forcefield to link to for this node path: " << l_nodeToParse.getPath();
    }

    auto ms1 = this->getMState1();
    auto ms2 = this->getMState2();
    m_nbColsJ1 = ms1->getSize()*DerivSize1;
    m_nbColsJ2 = ms2->getSize()*DerivSize2;

    auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler != nullptr);
    if (d_parallelTasks.getValue())
    {
        if (taskScheduler->getThreadCount() < 1)
        {
            taskScheduler->init(0);
            msg_info() << "Task scheduler initialized on " << taskScheduler->getThreadCount() << " threads";
        }
        else
        {
            msg_info() << "Task scheduler already initialized on " << taskScheduler->getThreadCount() << " threads";
        }
    }

    this->d_componentState.setValue(ComponentState::Valid) ;
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::bwdInit()
{
    m_fullMatrixSize = l_mechanicalState ? l_mechanicalState->getMatrixSize() : 0;
    if (m_fullMatrixSize > 0)
    {
        m_J1eig.resize(m_fullMatrixSize, m_nbColsJ1);
        m_J2eig.resize(m_fullMatrixSize, m_nbColsJ2);
    }
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::parseNode(sofa::simulation::Node *node, std::string massName)
{
    bool empty = d_forceFieldList.getValue().empty();
    msg_info() << "parsing node:";
    for(BaseForceField* forcefield : node->forceField)
    {
        if (forcefield->name.getValue() != massName || d_forceFieldAndMass.getValue())
        {
            bool found = true;
            if (!empty)
                found = (std::find(d_forceFieldList.getValue().begin(),
                                   d_forceFieldList.getValue().end(), forcefield->getName()) != d_forceFieldList.getValue().end());

            if(found)
            {
                l_forceField.add(forcefield,forcefield->getPathName());
            }
        }
    }

    for(BaseForceField* iforcefield : node->interactionForceField)
    {
        bool found = true;
        if (!empty)
            found = (std::find(d_forceFieldList.getValue().begin(),
                               d_forceFieldList.getValue().end(),
                               iforcefield->getName()) != d_forceFieldList.getValue().end());

        if(found)
        {
            l_forceField.add(iforcefield,iforcefield->getPathName());
        }

    }
    if (!d_stopAtNodeToParse.getValue())
        for(auto& child : node->child){
            parseNode(child.get(), massName);
        }
    return;
}

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::buildIdentityBlocksInJacobian(core::behavior::BaseMechanicalState* mstate, sofa::core::MatrixDerivId Id)
{
    sofa::type::vector<unsigned int> list;
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
    MechanicalResetConstraintVisitor(&cparams).execute(context);
    buildIdentityBlocksInJacobian(l_mechanicalState,Id);

    sofa::simulation::mechanicalvisitor::MechanicalAccumulateJacobian(&cparams, core::MatrixDerivId::mappingJacobian()).execute(gnode);
}

template<class T>
void copyKToEigenFormat(CompressedRowSparseMatrix< T >* K, Eigen::SparseMatrix<double,Eigen::ColMajor>& Keig)
{
    // It is assumed that K is not compressed. All the data is contained in the temporary container K->btemp
    // This data is provided to Eigen to build a compressed sparse matrix in Eigen format.
    // The strategy would be different if K was compressed. However, compression is avoided as it is very expensive.

    /// Structure complying with the expected interface of SparseMatrix::setFromTriplets
    struct IndexedBlocProxy
    {
        explicit IndexedBlocProxy(const typename CompressedRowSparseMatrix<T>::VecIndexedBlock::const_iterator& it) : m_iterator(it) {}
        T value() const { return m_iterator->value; }
        typename CompressedRowSparseMatrix< T >::Index row() const { return m_iterator->l; }
        typename CompressedRowSparseMatrix< T >::Index col() const { return m_iterator->c; }

        typename CompressedRowSparseMatrix<T>::VecIndexedBlock::const_iterator m_iterator;
    };
    /// Iterator provided to SparseMatrix::setFromTriplets giving access to an interface similar to Eigen::Triplet
    struct IndexedBlocIterator
    {
        explicit IndexedBlocIterator(const typename CompressedRowSparseMatrix<T>::VecIndexedBlock::const_iterator& it)
            : m_proxy(it) {}
        bool operator!=(const IndexedBlocIterator& rhs) const { return m_proxy.m_iterator != rhs.m_proxy.m_iterator; }
        IndexedBlocIterator& operator++() { ++m_proxy.m_iterator; return *this; }
        IndexedBlocProxy* operator->() { return &m_proxy; }
    private:
        IndexedBlocProxy m_proxy;
    };

    sofa::helper::ScopedAdvancedTimer timer("KfromTriplets" );
    IndexedBlocIterator begin(K->btemp.begin());
    IndexedBlocIterator end(K->btemp.end());
    Keig.setFromTriplets(begin, end);
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
    if (l_mappedMass != nullptr)
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
class MechanicalMatrixMapper<DataTypes1, DataTypes2>::JacobianTask final : public sofa::simulation::CpuTask
{
public:
    JacobianTask(
        sofa::simulation::CpuTask::Status* status,
        const MechanicalParams* mparams,
        MechanicalMatrixMapper<DataTypes1, DataTypes2>* mapper,
        sofa::core::behavior::MechanicalState<DataTypes1>* ms1,
        sofa::core::behavior::MechanicalState<DataTypes2>* ms2,
        sofa::core::behavior::BaseMechanicalState* bms1,
        sofa::core::behavior::BaseMechanicalState* bms2)
    : CpuTask(status)
    , m_mparams(mparams)
    , m_ms1(ms1)
    , m_ms2(ms2)
    , m_bms1(bms1)
    , m_bms2(bms2)
    , self(mapper)
    {}

    sofa::simulation::Task::MemoryAlloc run() override
    {
        self->accumulateJacobiansOptimized(m_mparams);

        sofa::core::MultiMatrixDerivId c = sofa::core::MatrixDerivId::mappingJacobian();
        const MatrixDeriv1 &J1 = c[m_ms1].read()->getValue();
        const MatrixDeriv2 &J2 = c[m_ms2].read()->getValue();

        self->optimizeAndCopyMappingJacobianToEigenFormat1(J1, self->m_J1eig);
        if (m_bms1 != m_bms2)
        {
            self->optimizeAndCopyMappingJacobianToEigenFormat2(J2, self->m_J2eig);
        }

        auto eparams = dynamic_cast<const core::ExecParams *>( m_mparams );
        auto cparams = core::ConstraintParams(*eparams);
        MechanicalResetConstraintVisitor(&cparams).execute(self->getContext());

        return simulation::Task::Stack;
    }

private:
    const MechanicalParams* m_mparams { nullptr };

    sofa::core::behavior::MechanicalState<DataTypes1>* m_ms1 { nullptr };
    sofa::core::behavior::MechanicalState<DataTypes2>* m_ms2 { nullptr };

    sofa::core::behavior::BaseMechanicalState* m_bms1 { nullptr };
    sofa::core::behavior::BaseMechanicalState* m_bms2 { nullptr };

    MechanicalMatrixMapper<DataTypes1, DataTypes2>* self { nullptr };
};

template<class DataTypes1, class DataTypes2>
void MechanicalMatrixMapper<DataTypes1, DataTypes2>::addKToMatrix(const MechanicalParams* mparams,
                                                                        const MultiMatrixAccessor* matrix)
{
    if(this->d_componentState.getValue() != ComponentState::Valid)
        return ;

    sofa::helper::ScopedAdvancedTimer addKToMatrixTimer("MMM-addKToMatrix");

    sofa::core::behavior::MechanicalState<DataTypes1>* ms1 = this->getMState1();
    sofa::core::behavior::MechanicalState<DataTypes2>* ms2 = this->getMState2();

    sofa::core::behavior::BaseMechanicalState*  bms1 = this->getMechModel1();
    sofa::core::behavior::BaseMechanicalState*  bms2 = this->getMechModel2();

    MultiMatrixAccessor::MatrixRef mat11 = matrix->getMatrix(mstate1);
    MultiMatrixAccessor::MatrixRef mat22 = matrix->getMatrix(mstate2);
    MultiMatrixAccessor::InteractionMatrixRef mat12 = matrix->getMatrix(mstate1, mstate2);
    MultiMatrixAccessor::InteractionMatrixRef mat21 = matrix->getMatrix(mstate2, mstate1);

    auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler != nullptr);

    ///////////////////////////     STEP 1      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*              compute jacobians using generic implementation                */
    /* -------------------------------------------------------------------------- */
    sofa::simulation::CpuTask::Status jacobianTaskStatus;
    JacobianTask jacobianTask(&jacobianTaskStatus, mparams, this, ms1, ms2, bms1, bms2);
    if (!d_parallelTasks.getValue())
    {
        jacobianTask.run();
    }
    else
    {
        taskScheduler->addTask(&jacobianTask);
    }

    ///////////////////////////     STEP 2      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*  compute the stiffness K of the forcefield and put it in a rowsparseMatrix */
    /*          get the stiffness matrix from the mapped ForceField               */
    /* TODO: use the template of the FF for Real                                  */
    /* -------------------------------------------------------------------------- */
    sofa::helper::AdvancedTimer::stepBegin("stiffness" );

    ///////////////////////     GET K       ////////////////////////////////////////
    CompressedRowSparseMatrix< Real1 >* K = new CompressedRowSparseMatrix< Real1 > ( );
    K->resizeBloc( m_fullMatrixSize ,  m_fullMatrixSize );
    K->clear();
    DefaultMultiMatrixAccessor* KAccessor;
    KAccessor = new DefaultMultiMatrixAccessor;
    KAccessor->addMechanicalState( l_mechanicalState );
    KAccessor->setGlobalMatrix(K);
    KAccessor->setupMatrices();

    const sofa::simulation::Node *node = l_nodeToParse.get();

    if (node->meshTopology)
    {
        if (const auto currentRevision = node->meshTopology->getRevision(); currentRevision != m_topologyRevision)
        {
            //the topology has been modified: intersection is no longer valid
            m_product_J1tK.product.invalidateIntersection();
            m_product_J2tK.product.invalidateIntersection();
            m_product_J1tKJ1.invalidateIntersection();
            m_product_J2tKJ2.invalidateIntersection();
            m_product_J1tKJ2.invalidateIntersection();
            m_product_J2tKJ1.invalidateIntersection();
            m_topologyRevision = currentRevision;
        }
    }

    size_t currentNbInteractionFFs = node->interactionForceField.size();
    msg_info() << "nb m_nbInteractionForceFields :" << m_nbInteractionForceFields << msgendl << "nb currentNbInteractionFFs :" << currentNbInteractionFFs;
    if (m_nbInteractionForceFields != currentNbInteractionFFs)
    {
        if (!l_forceField.empty())
        {
            while(l_forceField.size()>0)
            {
                l_forceField.removeAt(0);
            }

        }
        std::string massName;
        if (l_nodeToParse.get()->mass)
            massName.append(l_nodeToParse.get()->mass->getName());
        parseNode(l_nodeToParse.get(),massName);
        m_nbInteractionForceFields = currentNbInteractionFFs;
    }

    for(unsigned int i=0; i<l_forceField.size(); i++)
    {
        l_forceField[i]->addKToMatrix(mparams, KAccessor);
    }

    addMassToSystem(mparams,KAccessor);

    sofa::helper::AdvancedTimer::stepEnd("stiffness" );

    if (!K)
    {
        msg_error(this) << "matrix of the force-field system not found";
        return;
    }

    //------------------------------------------------------------------------------

    sofa::helper::AdvancedTimer::stepBegin("copyKToEigen" );
    Eigen::SparseMatrix<double,Eigen::ColMajor> Keig;
    Keig.resize(m_fullMatrixSize,m_fullMatrixSize);
    copyKToEigenFormat(K,Keig);
    sofa::helper::AdvancedTimer::stepEnd("copyKToEigen" );


    if (d_parallelTasks.getValue())
    {
        helper::ScopedAdvancedTimer jacobianTimer("waitJacobian");
        taskScheduler->workUntilDone(&jacobianTaskStatus);
    }

    ///////////////////////////     STEP 4      ////////////////////////////////////
    /* -------------------------------------------------------------------------- */
    /*          perform the multiplication with [J1t J2t] * K * [J1 J2]           */
    /* -------------------------------------------------------------------------- */
    sofa::helper::AdvancedTimer::stepBegin("Multiplication" );
    const auto fastProduct = d_fastMatrixProduct.getValue();
    m_product_J1tK.isComputed = false;
    m_product_J2tK.isComputed = false;

    m_nbColsJ1 = m_J1eig.cols();
    if (bms1 != bms2)
    {
        m_nbColsJ2 = m_J2eig.cols();
    }
    Eigen::SparseMatrix<double>* J1tKJ1eigen{ &m_J1tKJ1eigen };

    if (!d_skipJ1tKJ1.getValue())
    {
        sofa::helper::ScopedAdvancedTimer J1tKJ1Timer("J1tKJ1" );
        computeMatrixProduct(fastProduct, m_product_J1tK, m_product_J1tKJ1, &m_J1eig, &m_J1eig, &Keig, J1tKJ1eigen);
    }

    Eigen::SparseMatrix<double>* J2tKJ2eigen{ &m_J2tKJ2eigen };
    Eigen::SparseMatrix<double>* J1tKJ2eigen{ &m_J1tKJ2eigen };
    Eigen::SparseMatrix<double>* J2tKJ1eigen{ &m_J2tKJ1eigen };

    if (bms1 != bms2)
    {
        if (!d_skipJ2tKJ2.getValue())
        {
            sofa::helper::ScopedAdvancedTimer J2tKJ2Timer("J2tKJ2" );
            computeMatrixProduct(fastProduct, m_product_J2tK, m_product_J2tKJ2, &m_J2eig, &m_J2eig, &Keig, J2tKJ2eigen);
        }
        {
            sofa::helper::ScopedAdvancedTimer J1tKJ2Timer("J1tKJ2" );
            computeMatrixProduct(fastProduct, m_product_J1tK, m_product_J1tKJ2, &m_J1eig, &m_J2eig, &Keig, J1tKJ2eigen);
        }
        {
            sofa::helper::ScopedAdvancedTimer J2tKJ1Timer("J2tKJ1" );
            computeMatrixProduct(fastProduct, m_product_J2tK, m_product_J2tKJ1, &m_J2eig, &m_J1eig, &Keig, J2tKJ1eigen);
        }

    }

    sofa::helper::AdvancedTimer::stepEnd("Multiplication" );
    //--------------------------------------------------------------------------------------------------------------------

    const unsigned int mstateSize = l_mechanicalState->getSize();
    addPrecomputedMassToSystem(mparams,mstateSize,m_J1eig,*J1tKJ1eigen);

    sofa::helper::AdvancedTimer::stepBegin("copy" );

    const auto copyMatrixProduct = [](
        Eigen::SparseMatrix<double>* src, BaseMatrix* dst,
        const BaseMatrix::Index offrow, const BaseMatrix::Index offcol,
        const std::string stepName)
    {
        if (src)
        {
            sofa::helper::ScopedAdvancedTimer copyTimer(stepName );
            for (Eigen::Index k = 0; k < src->outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(*src, k); it; ++it)
                {
                    dst->add(
                        offrow + static_cast<BaseMatrix::Index>(it.row()),
                        offcol + static_cast<BaseMatrix::Index>(it.col()),
                        it.value());
                }
            }
        }
    };

    copyMatrixProduct(J1tKJ1eigen, mat11.matrix, mat11.offset, mat11.offset, "J1tKJ1-copy");

    if (bms1 != bms2)
    {
        copyMatrixProduct(J2tKJ2eigen, mat22.matrix, mat22.offset, mat22.offset, "J2tKJ2-copy");
        copyMatrixProduct(J1tKJ2eigen, mat12.matrix, mat12.offRow, mat12.offCol, "J1tKJ2-copy");
        copyMatrixProduct(J2tKJ1eigen, mat21.matrix, mat21.offRow, mat21.offCol, "J2tKJ1-copy");
    }
    sofa::helper::AdvancedTimer::stepEnd("copy" );

    delete KAccessor;
    delete K;
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
SReal MechanicalMatrixMapper<DataTypes1, DataTypes2>::getPotentialEnergy(const MechanicalParams* mparams,
                                                                          const DataVecCoord1& x1,
                                                                          const DataVecCoord2& x2) const
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x1);
    SOFA_UNUSED(x2);

    return 0.0;
}

} // namespace sofa::component::mapping::mappedmatrix
