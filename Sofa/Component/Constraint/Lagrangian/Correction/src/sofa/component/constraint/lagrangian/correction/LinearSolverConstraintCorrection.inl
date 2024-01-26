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
#include <sofa/component/constraint/lagrangian/correction/LinearSolverConstraintCorrection.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ConstraintParams.h>

#include <sstream>
#include <list>

#include <sofa/component/linearsolver/iterative/GraphScatteredTypes.h>

namespace sofa::component::constraint::lagrangian::correction
{

using sofa::core::objectmodel::ComponentState ;

#define MAX_NUM_CONSTRAINT_PER_NODE 100
#define EPS_UNITARY_FORCE 0.01

template<class DataTypes>
LinearSolverConstraintCorrection<DataTypes>::LinearSolverConstraintCorrection(sofa::core::behavior::MechanicalState<DataTypes> *mm)
: Inherit(mm)
, wire_optimization(initData(&wire_optimization, false, "wire_optimization", "constraints are reordered along a wire-like topology (from tip to base)"))
, l_linearSolver(initLink("linearSolver", "Link towards the linear solver used to compute the compliance matrix, requiring the inverse of the linear system matrix"))
, l_ODESolver(initLink("ODESolver", "Link towards the ODE solver used to recover the integration factors"))
{
}

template<class DataTypes>
LinearSolverConstraintCorrection<DataTypes>::~LinearSolverConstraintCorrection()
{
}


//////////////////////////////////////////////////////////////////////////
//   Precomputation of the Constraint Correction for all type of data
//////////////////////////////////////////////////////////////////////////

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::init()
{
    Inherit::init();

    sofa::core::objectmodel::BaseContext* context = this->getContext();


    // Find linear solver
    if (l_linearSolver.empty())
    {
        msg_info() << "Link \"linearSolver\" to the desired linear solver should be set to ensure right behavior." << msgendl
                   << "First LinearSolver found in current context will be used.";
        l_linearSolver.set( context->get<sofa::core::behavior::LinearSolver>(sofa::core::objectmodel::BaseContext::Local) );
    }

    if (l_linearSolver.get() == nullptr)
    {
        msg_error() << "No LinearSolver component found at path: " << l_linearSolver.getLinkedPath() << ", nor in current context: " << context->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    else
    {
        if (l_linearSolver.get()->getTemplateName() == "GraphScattered")
        {
            msg_error() << "Can not use the solver " << l_linearSolver.get()->getName() << " because it is templated on GraphScatteredType";
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
        else
        {
            msg_info() << "LinearSolver path used: '" << l_linearSolver.getLinkedPath() << "'";
        }
    }

    // Find ODE solver
    if (l_ODESolver.empty())
    {
        msg_info() << "Link \"ODESolver\" to the desired ODE solver should be set to ensure right behavior." << msgendl
                   << "First ODESolver found in current context will be used.";
        l_ODESolver.set( context->get<sofa::core::behavior::OdeSolver>(sofa::core::objectmodel::BaseContext::Local) );
        if (l_ODESolver.get() == nullptr)
        {
            l_ODESolver.set( context->get<sofa::core::behavior::OdeSolver>(sofa::core::objectmodel::BaseContext::SearchRoot) );
        }
    }

    if (l_ODESolver.get() == nullptr)
    {
        msg_error() << "No ODESolver component found at path: " << l_ODESolver.getLinkedPath() << ", nor in current context: " << context->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    else
    {
        msg_info() << "ODESolver path used: '" << l_ODESolver.getLinkedPath() << "'";
    }


    if(mstate==nullptr)
    {
        d_componentState.setValue(ComponentState::Invalid) ;
        return;
    }

    d_componentState.setValue(ComponentState::Valid) ;
}

template<class TDataTypes>
void LinearSolverConstraintCorrection<TDataTypes>::computeJ(sofa::linearalgebra::BaseMatrix* W, const MatrixDeriv& c)
{
    if(d_componentState.getValue() != ComponentState::Valid)
        return ;

    const unsigned int numDOFs = mstate->getSize();
    const unsigned int N = Deriv::size();
    const unsigned int numDOFReals = numDOFs*N;
    const unsigned int totalNumConstraints = W->rowSize();

    J.resize(totalNumConstraints, numDOFReals);

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        const int cid = rowIt.index();

        MatrixDerivColConstIterator colItEnd = rowIt.end();

        for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int dof = colIt.index();
            const Deriv n = colIt.val();

            for (unsigned int r = 0; r < N; ++r)
            {
                J.add(cid, dof * N + r, n[r]);
            }
        }
    }
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::addComplianceInConstraintSpace(const sofa::core::ConstraintParams *cparams, sofa::linearalgebra::BaseMatrix* W)
{
    if(d_componentState.getValue() != ComponentState::Valid)
        return ;

    // use the OdeSolver to get the position integration factor
    SReal factor = 1.0_sreal;

    switch (cparams->constOrder())
    {
    case core::ConstraintOrder::POS_AND_VEL :
    case core::ConstraintOrder::POS :
        factor = l_ODESolver->getPositionIntegrationFactor();
        break;

    case core::ConstraintOrder::ACC :
    case core::ConstraintOrder::VEL :
        factor = l_ODESolver->getVelocityIntegrationFactor();
        break;

    default :
        break;
    }

    // Compute J
    this->computeJ(W, cparams->readJ(this->mstate)->getValue());

    // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
    l_linearSolver.get()->setSystemLHVector(sofa::core::MultiVecDerivId::null());
    l_linearSolver.get()->addJMInvJt(W, &J, factor);
}


template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::rebuildSystem(SReal massFactor, SReal forceFactor)
{
    l_linearSolver.get()->rebuildSystem(massFactor, forceFactor);
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::getComplianceMatrix(linearalgebra::BaseMatrix* Minv) const
{
    if(d_componentState.getValue() != ComponentState::Valid)
        return ;

    const SReal factor = l_ODESolver.get()->getPositionIntegrationFactor();

    const unsigned int numDOFs = mstate->getSize();
    const unsigned int N = Deriv::size();
    const unsigned int numDOFReals = numDOFs*N;
    static linearalgebra::SparseMatrix<SReal> J; //local J
    if (J.rowSize() != (linearalgebra::BaseMatrix::Index)numDOFReals)
    {
        J.resize(numDOFReals,numDOFReals);
        for (unsigned int i=0; i<numDOFReals; ++i)
            J.set(i,i,1);
    }

    Minv->resize(numDOFReals,numDOFReals);

    // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
    l_linearSolver.get()->addJMInvJt(Minv, &J, factor);
}

template< class DataTypes >
void LinearSolverConstraintCorrection< DataTypes >::computeMotionCorrection(const core::ConstraintParams* /*cparams*/, core::MultiVecDerivId dx, core::MultiVecDerivId f)
{
    if (mstate && l_linearSolver.get())
    {
        l_linearSolver.get()->setSystemRHVector(f);
        l_linearSolver.get()->setSystemLHVector(dx);
        l_linearSolver.get()->solveSystem();
    }
}

template< class DataTypes >
void LinearSolverConstraintCorrection< DataTypes >::applyMotionCorrection(const core::ConstraintParams * cparams, Data< VecCoord > &x_d, Data< VecDeriv > &v_d, Data< VecDeriv > &dx_d, const Data< VecDeriv > &correction_d)
{
    if (mstate)
    {
        const unsigned int numDOFs = mstate->getSize();

        auto x = sofa::helper::getWriteAccessor(x_d);
        auto v = sofa::helper::getWriteAccessor(v_d);
        auto dx = sofa::helper::getWriteAccessor(dx_d);

        const VecDeriv& correction = correction_d.getValue();
        const VecCoord& x_free = cparams->readX(mstate)->getValue();
        const VecDeriv& v_free = cparams->readV(mstate)->getValue();

        const SReal positionFactor = l_ODESolver.get()->getPositionIntegrationFactor();
        const SReal velocityFactor = l_ODESolver.get()->getVelocityIntegrationFactor();

        for (unsigned int i = 0; i < numDOFs; i++)
        {
            const Deriv dxi = correction[i] * positionFactor;
            const Deriv dvi = correction[i] * velocityFactor;
            x[i] = x_free[i] + dxi;
            v[i] = v_free[i] + dvi;
            dx[i] = dxi;
        }
    }
}


template< class DataTypes >
void LinearSolverConstraintCorrection< DataTypes >::applyPositionCorrection(const sofa::core::ConstraintParams *cparams, Data< VecCoord >& x_d, Data< VecDeriv>& dx_d, const Data< VecDeriv >& correction_d)
{
    if (mstate)
    {
        const unsigned int numDOFs = mstate->getSize();
        auto x  = sofa::helper::getWriteAccessor(x_d);
        auto dx = sofa::helper::getWriteAccessor(dx_d);

        const VecDeriv& correction = correction_d.getValue();
        const VecCoord& x_free = cparams->readX(mstate)->getValue();

        const SReal positionFactor = l_ODESolver.get()->getPositionIntegrationFactor();
        for (unsigned int i = 0; i < numDOFs; i++)
        {
            const Deriv dxi = correction[i] * positionFactor;
            x[i] = x_free[i] + dxi;
            dx[i] = dxi;
        }
    }
}


template< class DataTypes >
void LinearSolverConstraintCorrection< DataTypes >::applyVelocityCorrection(const sofa::core::ConstraintParams *cparams, Data< VecDeriv>& v_d, Data< VecDeriv>& dv_d, const Data< VecDeriv >& correction_d)
{
    if (mstate)
    {
        const auto numDOFs = mstate->getSize();

        auto v  = sofa::helper::getWriteAccessor(v_d);
        auto dv = sofa::helper::getWriteAccessor(dv_d);

        const VecDeriv& correction = correction_d.getValue();
        const VecDeriv& v_free = cparams->readV(mstate)->getValue();

        const SReal velocityFactor = l_ODESolver.get()->getVelocityIntegrationFactor();

        for (unsigned int i = 0; i < numDOFs; i++)
        {
            Deriv dvi = correction[i] * velocityFactor;
            v[i] = v_free[i] + dvi;
            dv[i] = dvi;
        }
    }
}


template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::applyContactForce(const linearalgebra::BaseVector *f)
{
    core::VecDerivId forceID(core::VecDerivId::V_FIRST_DYNAMIC_INDEX);
    core::VecDerivId dxID = core::VecDerivId::dx();

    const unsigned int numDOFs = mstate->getSize();

    Data<VecDeriv>& dataDx = *mstate->write(dxID);
    VecDeriv& dx = *dataDx.beginEdit();

    dx.clear();
    dx.resize(numDOFs);
    for (unsigned int i=0; i< numDOFs; i++)
        dx[i] = Deriv();

    Data<VecDeriv>& dataForce = *mstate->write(forceID);
    VecDeriv& force = *dataForce.beginEdit();

    force.clear();
    force.resize(numDOFs);
    for (unsigned int i=0; i< numDOFs; i++)
        force[i] = Deriv();

    const MatrixDeriv& c = mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        const SReal fC1 = f->element(rowIt.index());

        if (fC1 != 0.0)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                force[colIt.index()] += colIt.val() * fC1;
            }
        }
    }
    l_linearSolver.get()->setSystemRHVector(forceID);
    l_linearSolver.get()->setSystemLHVector(dxID);
    l_linearSolver.get()->solveSystem();

    //TODO: tell the solver not to recompute the matrix

    // use the OdeSolver to get the position integration factor
    const SReal positionFactor = l_ODESolver.get()->getPositionIntegrationFactor();

    // use the OdeSolver to get the position integration factor
    const SReal velocityFactor = l_ODESolver.get()->getVelocityIntegrationFactor();

    Data<VecCoord>& xData     = *mstate->write(core::VecCoordId::position());
    Data<VecDeriv>& vData     = *mstate->write(core::VecDerivId::velocity());
    const Data<VecCoord> & xfreeData = *mstate->read(core::ConstVecCoordId::freePosition());
    const Data<VecDeriv> & vfreeData = *mstate->read(core::ConstVecDerivId::freeVelocity());
    VecCoord& x = *xData.beginEdit();
    VecDeriv& v = *vData.beginEdit();
    const VecCoord& x_free = xfreeData.getValue();
    const VecDeriv& v_free = vfreeData.getValue();

    for (unsigned int i=0; i< numDOFs; i++)
    {
        Deriv dxi = dx[i]*positionFactor;
        Deriv dvi = dx[i]*velocityFactor;
        x[i] = x_free[i] + dxi;
        v[i] = v_free[i] + dvi;
        dx[i] = dxi;

        msg_info() << "dx[" << i << "] = " << dx[i] ;
    }

    dataDx.endEdit();
    dataForce.endEdit();
    xData.endEdit();
    vData.endEdit();

    /// @todo: freeing forceID here is incorrect as it was not allocated
    /// Maybe the call to vAlloc at the beginning of this method should be enabled...
    /// -- JeremieA, 2011-02-16
    mstate->vFree(core::execparams::defaultInstance(), forceID);
}


template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::resetContactForce()
{
    Data<VecDeriv>& forceData = *mstate->write(core::VecDerivId::force());
    VecDeriv& force = *forceData.beginEdit();
    for( unsigned i=0; i<force.size(); ++i )
        force[i] = Deriv();
    forceData.endEdit();
}


template<class DataTypes>
bool LinearSolverConstraintCorrection<DataTypes>::hasConstraintNumber(int index)
{
    const MatrixDeriv& c = mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();

    return c.readLine(index) != c.end();
}


template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::verify_constraints()
{
    // New design prevents duplicated constraints.
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::resetForUnbuiltResolution(SReal* f, std::list<unsigned int>& renumbering)
{
    verify_constraints();

    const MatrixDeriv& constraints = mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();

    constraint_force.clear();
    constraint_force.resize(mstate->getSize());

    constraint_dofs.clear();

    const unsigned int nbConstraints = constraints.size();
    std::vector<unsigned int> VecMinDof;
    VecMinDof.resize(nbConstraints);

    unsigned int c = 0;

    MatrixDerivRowConstIterator rowItEnd = constraints.end();

    for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
    {

        const int indexC = rowIt.index(); // id constraint


        // buf the value of force applied on concerned dof : constraint_force
        // buf a table of indice of involved dof : constraint_dofs
        SReal fC = f[indexC];

        if (fC != 0.0)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const unsigned int dof = colIt.index();
                constraint_force[dof] += colIt.val() * fC;
            }
        }

        //////////// for wire optimization ////////////
        // VecMinDof contains the smallest id of the dofs involved in each constraint [c]

        MatrixDerivColConstIterator colItEnd = rowIt.end();


        VecMinDof[c] = mstate->getSize()+1;

        for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int dof = colIt.index();
            constraint_dofs.push_back(dof);
            if (dof < VecMinDof[c])
                VecMinDof[c] = dof;
        }

        c++;
    }
    // constraint_dofs buff the DOF that are involved with the constraints
    constraint_dofs.unique();


    // in the following the list "renumbering" is modified so that the constraint, in the list appears from the smallest dof to the greatest
    // However some constraints are not concerned by the structure... in such case, their order should not be changed
    // (in practice they will be put at the beginning of the list)
    if (wire_optimization.getValue())
    {
        std::vector< std::vector<unsigned int> > ordering_per_dof;
        ordering_per_dof.resize(mstate->getSize());   // for each dof, provide the list of constraint for which this dof is the smallest involved

        rowItEnd = constraints.end();
        {
            unsigned int constraintId = 0;

            // we process each constraint of the Mechanical State to know the smallest dofs
            // the constraints that are concerns the object are removed
            for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
            {
                ordering_per_dof[VecMinDof[constraintId]].push_back(rowIt.index());
                constraintId++;
                renumbering.remove( rowIt.index() );
            }
        }


        // fill the end renumbering list with the new order
        for (size_t dof = 0; dof < mstate->getSize(); dof++)
        {
            for (size_t constraintId = 0; constraintId < ordering_per_dof[dof].size(); constraintId++)
            {
                renumbering.push_back(ordering_per_dof[dof][constraintId]); // push_back the list of constraint by starting from the smallest dof
            }
        }
    }

    /////////////// SET INFO FOR LINEAR SOLVER /////////////
    core::VecDerivId forceID(core::VecDerivId::V_FIRST_DYNAMIC_INDEX);
    core::VecDerivId dxID = core::VecDerivId::dx();

    l_linearSolver.get()->setSystemRHVector(forceID);
    l_linearSolver.get()->setSystemLHVector(dxID);


    systemMatrix_buf   = l_linearSolver.get()->getSystemBaseMatrix();
    systemRHVector_buf = l_linearSolver.get()->getSystemRHBaseVector();
    systemLHVector_buf = l_linearSolver.get()->getSystemLHBaseVector();
    systemLHVector_buf_fullvector = dynamic_cast<linearalgebra::FullVector<Real>*>(systemLHVector_buf); // Cast checking whether the LH vector is a FullVector to improve performances

    constexpr const auto derivDim = Deriv::total_size;
    const unsigned int systemSize = mstate->getSize() * derivDim;
    systemRHVector_buf->resize(systemSize) ;
    systemLHVector_buf->resize(systemSize) ;

    for ( size_t i=0; i<mstate->getSize(); i++)
    {
        for  (unsigned int j=0; j<derivDim; j++)
            systemRHVector_buf->set(i*derivDim+j, constraint_force[i][j]);
    }

    // Init the internal data of the solver for partial solving
    l_linearSolver.get()->init_partial_solve();


    ///////// new : precalcul des liste d'indice ///////
    Vec_I_list_dof.clear(); // clear = the list is filled during the block compliance computation

    // Resize
    unsigned int maxIdConstraint = 0;
    for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
    {
        const unsigned int indexC = rowIt.index(); // id constraint
        if (indexC > maxIdConstraint) // compute the max of the Id
            maxIdConstraint = indexC;
    }

    Vec_I_list_dof.resize(maxIdConstraint + 1);

    last_disp = 0;
    last_force = nbConstraints - 1;
    _new_force = false;
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::addConstraintDisplacement(SReal*d, int begin, int end)
{
    const MatrixDeriv& constraints = mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();

    last_disp = begin;

    l_linearSolver->partial_solve(Vec_I_list_dof[last_disp], Vec_I_list_dof[last_force], _new_force);

    _new_force = false;

    // Lambda function adding the constraint displacement using [] if a FullVector is detected or element() else
    constexpr auto addConstraintDisplacement_impl = [](SReal* d, unsigned int id, auto* systemLHVector_buf, SReal positionIntegrationFactor, unsigned int dof, const Deriv& val)
    {
        constexpr const auto derivDim = Deriv::total_size;
        Deriv disp(type::NOINIT);

        for (Size j = 0; j < derivDim; j++)
        {
            if constexpr (std::is_same_v<decltype(systemLHVector_buf), linearalgebra::FullVector<SReal>*>)
            {
                disp[j] = (*systemLHVector_buf)[dof * derivDim + j] * positionIntegrationFactor;
            }
            else
            {
                disp[j] = (Real)(systemLHVector_buf->element(dof * derivDim + j)) * positionIntegrationFactor;
            }
        }

        d[id] += val * disp;
    };

    const auto positionIntegrationFactor = l_ODESolver->getPositionIntegrationFactor();

    // TODO => optimisation => for each bloc store J[bloc,dof]
    for (int i = begin; i <= end; i++)
    {
        MatrixDerivRowConstIterator rowIt = constraints.readLine(i);

        if (rowIt != constraints.end()) // useful ??
        {
            MatrixDerivColConstIterator rowEnd = rowIt.end();

            if (systemLHVector_buf_fullvector)
            {
                for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != rowEnd; ++colIt)
                {
                    addConstraintDisplacement_impl(d, i, systemLHVector_buf_fullvector, positionIntegrationFactor, colIt.index(), colIt.val());
                }
            }
            else
            {
                for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != rowEnd; ++colIt)
                {
                    addConstraintDisplacement_impl(d, i, systemLHVector_buf, positionIntegrationFactor, colIt.index(), colIt.val());
                }
            }
        }
    }
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::setConstraintDForce(SReal* df, int begin, int end, bool update)
{
    last_force = begin;

    if (!update)
        return;

    _new_force = true;

    constexpr const auto derivDim = Deriv::total_size;
    const MatrixDeriv& constraints = mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();

    // TODO => optimisation !!!
    for (int i = begin; i <= end; i++)
    {
         MatrixDerivRowConstIterator rowIt = constraints.readLine(i);

        if (rowIt != constraints.end())
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const Deriv n = colIt.val();
                const unsigned int dof = colIt.index();

                constraint_force[dof] += n * df[i]; // sum of the constraint force in the DOF space

            }
        }
    }

    // course on indices of the dofs involved invoved in the bloc //
    auto it_dof(Vec_I_list_dof[last_force].cbegin()), it_end(Vec_I_list_dof[last_force].cend());
    for(; it_dof!=it_end; ++it_dof)
    {
        auto dof =(*it_dof) ;
        for (Size j=0; j<derivDim; j++)
            systemRHVector_buf->set(dof * derivDim + j, constraint_force[dof][j]);
    }

}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::getBlockDiagonalCompliance(linearalgebra::BaseMatrix* W, int begin, int end)
{
    if(d_componentState.getValue() != ComponentState::Valid)
        return ;

    // use the OdeSolver to get the position integration factor
    const SReal factor = l_ODESolver.get()->getPositionIntegrationFactor(); //*m_ODESolver->getPositionIntegrationFactor(); // dt*dt

    const unsigned int numDOFs = mstate->getSize();
    const unsigned int N = Deriv::size();
    const unsigned int numDOFReals = numDOFs*N;

    // Compute J
    const MatrixDeriv& constraints = mstate->read(core::ConstMatrixDerivId::constraintJacobian())->getValue();
    const unsigned int totalNumConstraints = W->rowSize();

    J.resize(totalNumConstraints, numDOFReals);

    for (int i = begin; i <= end; i++)
    {


        MatrixDerivRowConstIterator rowIt = constraints.readLine(i);

        if (rowIt != constraints.end())
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            unsigned int dof_buf = 0;
            const int debug = 0;

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const unsigned int dof = colIt.index();
                const Deriv n = colIt.val();

                for (unsigned int r = 0; r < N; ++r)
                    J.add(i, dof * N + r, n[r]);

                if (debug!=0)
                {
                    const int test = dof_buf - dof;
                    if (test>2 || test< -2)
                        dmsg_info() << "For constraint id1 dof1 = " << dof_buf << " dof2 = " << dof;
                }

                dof_buf = dof;
            }
        }
    }

    // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
    l_linearSolver.get()->addJMInvJt(W, &J, factor);

    // construction of  Vec_I_list_dof : vector containing, for each constraint block, the list of dof concerned

    ListIndex list_dof;

    for (int i = begin; i <= end; i++)
    {


        MatrixDerivRowConstIterator rowIt = constraints.readLine(i);

        if (rowIt != constraints.end())
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                list_dof.push_back(colIt.index());
            }
        }
    }

    list_dof.sort();
    list_dof.unique();

    for (int i = begin; i <= end; i++)
    {
        Vec_I_list_dof[i] = list_dof;
    }
}

} //namespace sofa::component::constraint::lagrangian::correction
