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

#include <sofa/component/haptics/LCPForceFeedback.h>

#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>

#include <sofa/simulation/AnimateEndEvent.h>

#include <algorithm>
#include <mutex>

namespace
{

template <typename DataTypes>
bool derivVectors(const typename DataTypes::VecCoord& x0, const typename DataTypes::VecCoord& x1, typename DataTypes::VecDeriv& d, bool /*derivRotation*/)
{
    size_t sz0 = x0.size();
    const size_t szmin = std::min(sz0,x1.size());

    d.resize(sz0);
    for(size_t i=0; i<szmin; ++i)
    {
        d[i]=x1[i]-x0[i];
    }
    for(size_t i=szmin; i<sz0; ++i) // not sure in what case this is applicable...
    {
        d[i]=-x0[i];
    }
    return true;
}


template <typename DataTypes>
bool derivRigid3Vectors(const typename DataTypes::VecCoord& x0, const typename DataTypes::VecCoord& x1, typename DataTypes::VecDeriv& d, bool derivRotation=false)
{
    size_t sz0 = x0.size();
    const size_t szmin = std::min(sz0,x1.size());

    d.resize(sz0);
    for(size_t i=0; i<szmin; ++i)
    {
        getVCenter(d[i]) = x1[i].getCenter() - x0[i].getCenter();
        if (derivRotation)
        {
            // rotations are taken into account to compute the violations
            sofa::type::Quat<SReal> q;
            getVOrientation(d[i]) = x0[i].rotate(q.angularDisplacement(x1[i].getOrientation(), x0[i].getOrientation() ) ); // angularDisplacement compute the rotation vector btw the two quaternions
        }
        else
            getVOrientation(d[i]) *= 0;
    }

    for(size_t i=szmin; i<sz0; ++i) // not sure in what case this is applicable..
    {
        getVCenter(d[i]) = - x0[i].getCenter();

        if (derivRotation)
        {
            // rotations are taken into account to compute the violations
            sofa::type::Quat<SReal> q= x0[i].getOrientation();
            getVOrientation(d[i]) = -x0[i].rotate( q.quatToRotationVector() );  // Use of quatToRotationVector instead of toEulerVector:
                                                                                // this is done to keep the old behavior (before the
                                                                                // correction of the toEulerVector  function). If the
                                                                                // purpose was to obtain the Eulerian vector and not the
                                                                                // rotation vector please use the following line instead
        }
        else
            getVOrientation(d[i]) *= 0;
    }

    return true;
}


template <typename DataTypes>
double computeDot(const typename DataTypes::Deriv& v0, const typename DataTypes::Deriv& v1)
{
    return dot(v0,v1);
}


template<>
bool derivVectors<sofa::defaulttype::Rigid3Types>(const sofa::defaulttype::Rigid3Types::VecCoord& x0, const sofa::defaulttype::Rigid3Types::VecCoord& x1, sofa::defaulttype::Rigid3Types::VecDeriv& d, bool derivRotation )
{
    return derivRigid3Vectors<sofa::defaulttype::Rigid3Types>(x0,x1,d, derivRotation);
}
template <>
double computeDot<sofa::defaulttype::Rigid3Types>(const sofa::defaulttype::Rigid3Types::Deriv& v0, const sofa::defaulttype::Rigid3Types::Deriv& v1)
{
    return dot(getVCenter(v0),getVCenter(v1)) + dot(getVOrientation(v0), getVOrientation(v1));
}



} // anonymous namespace

namespace sofa::component::haptics
{

template <class DataTypes>
LCPForceFeedback<DataTypes>::LCPForceFeedback()
    : forceCoef(initData(&forceCoef, 0.03, "forceCoef","multiply haptic force by this coef."))
    , solverTimeout(initData(&solverTimeout, 0.0008, "solverTimeout","max time to spend solving constraints."))
    , d_solverMaxIt(initData(&d_solverMaxIt, 100, "solverMaxIt", "max iteration to spend solving constraints"))
    , d_derivRotations(initData(&d_derivRotations, false, "derivRotations", "if true, deriv the rotations when updating the violations"))
    , d_localHapticConstraintAllFrames(initData(&d_localHapticConstraintAllFrames, false, "localHapticConstraintAllFrames", "Flag to enable/disable constraint haptic influence from all frames"))
    , mState(nullptr)
    , mNextBufferId(0)
    , mCurBufferId(0)
    , mIsCuBufferInUse(false)
    , constraintSolver(nullptr)
    , _timer(nullptr)
    , time_buf(0)
    , timer_iterations(0)
    , haptic_freq(0.0)
    , num_constraints(0)
{
    this->f_listening.setValue(true);
    mCP[0] = nullptr;
    mCP[1] = nullptr;
    mCP[2] = nullptr;
    _timer = new helper::system::thread::CTime();
    time_buf = _timer->getTime();
    timer_iterations = 0;
}


template <class DataTypes>
void LCPForceFeedback<DataTypes>::init()
{
    const core::objectmodel::BaseContext* c = this->getContext();

    this->ForceFeedback::init();
    if(!c)
    {
        msg_error() << "LCPForceFeedback has no current context. Initialisation failed.";
        return;
    }

    c->get(constraintSolver);

    if (!constraintSolver)
    {
        msg_error() << "LCPForceFeedback has no binding ConstraintSolver. Initialisation failed.";
        return;
    }

    mState = dynamic_cast<core::behavior::MechanicalState<DataTypes> *> (c->getMechanicalState());
    if (!mState)
    {
        msg_error() << "LCPForceFeedback has no binding MechanicalState. Initialisation failed.";
        return;
    }
}

template <class DataTypes>
void LCPForceFeedback<DataTypes>::setLock(bool value)
{
    value == true ? lockForce.lock() : lockForce.unlock();
}


static std::mutex s_mtx;

template <class DataTypes>
void LCPForceFeedback<DataTypes>::computeForce(const VecCoord& state,  VecDeriv& forces)
{
    if (!this->d_activate.getValue())
    {
        return;
    }
    updateStats();

    lockForce.lock(); // check if computation has not been locked using setLock method.
    updateConstraintProblem();
    doComputeForce(state, forces);
    lockForce.unlock();
}
template <class DataTypes>
void LCPForceFeedback<DataTypes>::updateStats()
{
    using namespace helper::system::thread;

    const ctime_t actualTime = _timer->getTime();
    ++timer_iterations;
    if (actualTime - time_buf >= sofa::helper::system::thread::CTime::getTicksPerSec())
    {
        haptic_freq = (double)(timer_iterations*sofa::helper::system::thread::CTime::getTicksPerSec())/ (double)( actualTime - time_buf) ;
        time_buf = actualTime;
        timer_iterations = 0;
    }
}

template <class DataTypes>
bool LCPForceFeedback<DataTypes>::updateConstraintProblem()
{
    const int prevId = mCurBufferId;

    //
    // Retrieve the last LCP and constraints computed by the Sofa thread.
    //
    mIsCuBufferInUse = true;

    {
        // TODO: Lock and/or memory barrier HERE
        mCurBufferId = mNextBufferId;
    }

    const bool changed = (prevId != mCurBufferId);

    const sofa::component::constraint::lagrangian::solver::ConstraintProblem* cp = mCP[mCurBufferId];

    if(!cp)
    {
        mIsCuBufferInUse = false;
    }

    return changed;
}

template <class DataTypes>
void LCPForceFeedback<DataTypes>::doComputeForce(const VecCoord& state,  VecDeriv& forces)
{
    const unsigned int stateSize = state.size();
    forces.resize(stateSize);
    for (unsigned int i = 0; i < forces.size(); ++i)
    {
        forces[i].clear();
    }

    if(!constraintSolver||!mState)
        return;

    const MatrixDeriv& constraints = mConstraints[mCurBufferId];
    VecCoord &val = mVal[mCurBufferId];
    sofa::component::constraint::lagrangian::solver::ConstraintProblem* cp = mCP[mCurBufferId];

    if(!cp)
    {
        return;
    }

    if(!constraints.empty())
    {
        VecDeriv dx;

        derivVectors< DataTypes >(val, state, dx, d_derivRotations.getValue());

        const bool localHapticConstraintAllFrames = d_localHapticConstraintAllFrames.getValue();

        // Modify Dfree
        MatrixDerivRowConstIterator rowItEnd = constraints.end();
        num_constraints = constraints.size();

        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                cp->getDfree()[rowIt.index()] += computeDot<DataTypes>(colIt.val(), dx[localHapticConstraintAllFrames ? 0 : colIt.index()]);
            }
        }

        s_mtx.lock();

        // Solving constraints
        cp->solveTimed(cp->tolerance * 0.001, d_solverMaxIt.getValue(), solverTimeout.getValue());	// tol, maxIt, timeout

        // Restore Dfree
        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                cp->getDfree()[rowIt.index()] -= computeDot<DataTypes>(colIt.val(), dx[localHapticConstraintAllFrames ? 0 : colIt.index()]);
            }
        }

        s_mtx.unlock();

        VecDeriv tempForces;
        tempForces.resize(val.size());

        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            if (cp->getF()[rowIt.index()] != 0.0)
            {
                MatrixDerivColConstIterator colItEnd = rowIt.end();

                for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
                {
                    tempForces[localHapticConstraintAllFrames ? 0 : colIt.index()] += colIt.val() * cp->getF()[rowIt.index()];
                }
            }
        }

        for(unsigned int i = 0; i < stateSize; ++i)
        {
            forces[i] = tempForces[i] * forceCoef.getValue();
        }
    }
}


template <typename DataTypes>
void LCPForceFeedback<DataTypes>::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (!sofa::simulation::AnimateEndEvent::checkEventType(event))
        return;

    if (!constraintSolver)
        return;

    if (!mState)
        return;

    sofa::component::constraint::lagrangian::solver::ConstraintProblem* new_cp = constraintSolver->getConstraintProblem();

    if (!new_cp)
        return;

    // Find available buffer

    unsigned char buf_index=0;
    const unsigned char cbuf_index=mCurBufferId;
    const unsigned char nbuf_index=mNextBufferId;

    if (buf_index == cbuf_index || buf_index == nbuf_index)
    {
        buf_index++;
        if (buf_index == cbuf_index || buf_index == nbuf_index)
            buf_index++;
    }

    // Compute constraints, id_buf lcp and val for the current lcp.

    MatrixDeriv& constraints = mConstraints[buf_index];

    //	std::vector<int>& id_buf = mId_buf[buf_index];
    VecCoord& val = mVal[buf_index];

    // Update LCP
    mCP[buf_index] = new_cp;

    // Update Val
    val = mState->read(sofa::core::VecCoordId::freePosition())->getValue();

    // Update constraints and id_buf
    constraints.clear();
    //	id_buf.clear();

    const MatrixDeriv& c = mState->read(core::ConstMatrixDerivId::constraintJacobian())->getValue()   ;

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        constraints.addLine(rowIt.index(), rowIt.row());
    }

    // make sure the MatrixDeriv has been compressed
    constraints.compress();

    // valid buffer

    {
        // TODO: Lock and/or memory barrier HERE
        mNextBufferId = buf_index;
    }

    // Lock lcp to prevent its use by the SOFA thread while it is used by haptic thread
    if(mIsCuBufferInUse)
        constraintSolver->lockConstraintProblem(this, mCP[mCurBufferId], mCP[mNextBufferId]);
    else
        constraintSolver->lockConstraintProblem(this, mCP[mNextBufferId]);
}


//
// Those functions are here for compatibility with the Forcefeedback scheme
//

template <typename DataTypes>
void LCPForceFeedback<DataTypes>::computeForce(SReal , SReal, SReal, SReal, SReal, SReal, SReal, SReal&, SReal&, SReal&)
{

}


template <typename DataTypes>
void LCPForceFeedback<DataTypes>::computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &,
        const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &,
        sofa::defaulttype::SolidTypes<SReal>::SpatialVector & )
{

}



template <>
void SOFA_COMPONENT_HAPTICS_API LCPForceFeedback< sofa::defaulttype::Rigid3Types >::computeForce(SReal x, SReal y, SReal z, SReal, SReal, SReal, SReal, SReal& fx, SReal& fy, SReal& fz);

template <>
void SOFA_COMPONENT_HAPTICS_API LCPForceFeedback< sofa::defaulttype::Rigid3Types >::computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &world_H_tool,
        const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &/*V_tool_world*/,
        sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world );




} // namespace sofa::component::haptics
