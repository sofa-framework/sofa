/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_INL
#define SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_INL

#include <SofaHaptics/LCPForceFeedback.h>

#include <SofaConstraint/ConstraintSolverImpl.h>

#include <sofa/simulation/AnimateEndEvent.h>

#include <algorithm>
#include <mutex>

namespace
{

template <typename DataTypes>
bool derivVectors(const typename DataTypes::VecCoord& x0, const typename DataTypes::VecCoord& x1, typename DataTypes::VecDeriv& d, bool /*derivRotation*/)
{
    unsigned int sz0 = x0.size();
    unsigned int szmin = std::min(sz0,(unsigned int)x1.size());

    d.resize(sz0);
    for(unsigned int i=0; i<szmin; ++i)
    {
        d[i]=x1[i]-x0[i];
    }
    for(unsigned int i=szmin; i<sz0; ++i) // not sure in what case this is applicable...
    {
        d[i]=-x0[i];
    }
    return true;
}


template <typename DataTypes>
bool derivRigid3Vectors(const typename DataTypes::VecCoord& x0, const typename DataTypes::VecCoord& x1, typename DataTypes::VecDeriv& d, bool derivRotation=false)
{
    unsigned int sz0 = x0.size();
    unsigned int szmin = std::min(sz0,(unsigned int)x1.size());

    d.resize(sz0);
    for(unsigned int i=0; i<szmin; ++i)
    {
        getVCenter(d[i]) = x1[i].getCenter() - x0[i].getCenter();
        if (derivRotation)
        {
            // rotations are taken into account to compute the violations
            sofa::defaulttype::Quat q;
            getVOrientation(d[i]) = x0[i].rotate(q.angularDisplacement(x1[i].getOrientation(), x0[i].getOrientation() ) ); // angularDisplacement compute the rotation vector btw the two quaternions
        }
        else
            getVOrientation(d[i]) *= 0; 
    }

    for(unsigned int i=szmin; i<sz0; ++i) // not sure in what case this is applicable.. 
    {
        getVCenter(d[i]) = - x0[i].getCenter();

        if (derivRotation)
        {
            // rotations are taken into account to compute the violations
            sofa::defaulttype::Quat q= x0[i].getOrientation();
            getVOrientation(d[i]) = -x0[i].rotate( q.quatToRotationVector() );  // Use of quatToRotationVector instead of toEulerVector:
                                                                                // this is done to keep the old behavior (before the
                                                                                // correction of the toEulerVector  function). If the
                                                                                // purpose was to obtain the Eulerian vector and not the
                                                                                // rotation vector please use the following line instead
//            getVOrientation(d[i]) = -x0[i].rotate( q.toEulerVector() );
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


#ifndef SOFA_FLOAT
template<>
bool derivVectors<sofa::defaulttype::Rigid3dTypes>(const sofa::defaulttype::Rigid3dTypes::VecCoord& x0, const sofa::defaulttype::Rigid3dTypes::VecCoord& x1, sofa::defaulttype::Rigid3dTypes::VecDeriv& d, bool derivRotation )
{
    return derivRigid3Vectors<sofa::defaulttype::Rigid3dTypes>(x0,x1,d, derivRotation);
}
template <>
double computeDot<sofa::defaulttype::Rigid3dTypes>(const sofa::defaulttype::Rigid3dTypes::Deriv& v0, const sofa::defaulttype::Rigid3dTypes::Deriv& v1)
{
    return dot(getVCenter(v0),getVCenter(v1)) + dot(getVOrientation(v0), getVOrientation(v1));
}

#endif
#ifndef SOFA_DOUBLE
template<>
bool derivVectors<sofa::defaulttype::Rigid3fTypes>(const sofa::defaulttype::Rigid3fTypes::VecCoord& x0, const sofa::defaulttype::Rigid3fTypes::VecCoord& x1, sofa::defaulttype::Rigid3fTypes::VecDeriv& d, bool derivRotation )
{
    return derivRigid3Vectors<sofa::defaulttype::Rigid3fTypes>(x0,x1,d, derivRotation);
}
template <>
double computeDot<sofa::defaulttype::Rigid3fTypes>(const sofa::defaulttype::Rigid3fTypes::Deriv& v0, const sofa::defaulttype::Rigid3fTypes::Deriv& v1)
{
    return dot(getVCenter(v0),getVCenter(v1)) + dot(getVOrientation(v0), getVOrientation(v1));
}

#endif

} // anonymous namespace

namespace sofa
{

namespace component
{

namespace controller
{

template <class DataTypes>
LCPForceFeedback<DataTypes>::LCPForceFeedback()
    : forceCoef(initData(&forceCoef, 0.03, "forceCoef","multiply haptic force by this coef."))
    , solverTimeout(initData(&solverTimeout, 0.0008, "solverTimeout","max time to spend solving constraints."))
    , d_derivRotations(initData(&d_derivRotations, false, "derivRotations", "if true, deriv the rotations when updating the violations"))
    , mState(NULL)
    , mNextBufferId(0)
    , mCurBufferId(0)
    , mIsCuBufferInUse(false)
    , constraintSolver(NULL)
    , _timer(NULL)
    , time_buf(0)
    , timer_iterations(0)
    , haptic_freq(0.0)
    , num_constraints(0)
{
    this->f_listening.setValue(true);
    mCP[0] = NULL;
    mCP[1] = NULL;
    mCP[2] = NULL;
    _timer = new helper::system::thread::CTime();
    time_buf = _timer->getTime();
    timer_iterations = 0;
}


template <class DataTypes>
void LCPForceFeedback<DataTypes>::init()
{
    core::objectmodel::BaseContext* c = this->getContext();

    this->ForceFeedback::init();
    if(!c)
    {
        serr << "LCPForceFeedback has no current context. Initialisation failed." << sendl;
        return;
    }

    c->get(constraintSolver);

    if (!constraintSolver)
    {
        serr << "LCPForceFeedback has no binding ConstraintSolver. Initialisation failed." << sendl;
        return;
    }

    mState = dynamic_cast<core::behavior::MechanicalState<DataTypes> *> (c->getMechanicalState());
    if (!mState)
    {
        serr << "LCPForceFeedback has no binding MechanicalState. Initialisation failed." << sendl;
        return;
    }
}

std::mutex s_mtx;

template <class DataTypes>
void LCPForceFeedback<DataTypes>::computeForce(const VecCoord& state,  VecDeriv& forces)
{    
    const unsigned int stateSize = state.size();
    // Resize du vecteur force. Initialization � 0 ?
    forces.resize(stateSize);
    // Init to 0
    for(unsigned int i = 0; i < stateSize; ++i)
    {
        forces[i] = Deriv();
    }

    sofa::helper::system::thread::ctime_t actualTime = _timer->getTime();
    ++timer_iterations;
    if (actualTime - time_buf >= sofa::helper::system::thread::CTime::getTicksPerSec())
    {
        haptic_freq = (double)(timer_iterations*sofa::helper::system::thread::CTime::getTicksPerSec())/ (double)( actualTime - time_buf) ;
        time_buf = actualTime;
        timer_iterations = 0;
    }

    if(!constraintSolver||!mState)
        return;


    if (!this->f_activate.getValue())
    {
        return;
    }


    //
    // Retrieve the last LCP and constraints computed by the Sofa thread.
    //
    mIsCuBufferInUse = true;


    mCurBufferId = mNextBufferId;

    const MatrixDeriv& constraints = mConstraints[mCurBufferId];
    //	std::vector<int> &id_buf = mId_buf[mCurBufferId];
    VecCoord &val = mVal[mCurBufferId];
    component::constraintset::ConstraintProblem* cp = mCP[mCurBufferId];

    if(!cp)
    {
        mIsCuBufferInUse = false;
        return;
    }

    if(!constraints.empty())
    {
        VecDeriv dx;

        derivVectors< DataTypes >(val, state, dx, d_derivRotations.getValue());

        // Modify Dfree
        MatrixDerivRowConstIterator rowItEnd = constraints.end();
        num_constraints = constraints.size();

        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                cp->getDfree()[rowIt.index()] += computeDot<DataTypes>(colIt.val(), dx[colIt.index()]);
            }
        }

        s_mtx.lock();

        // Solving constraints
        cp->solveTimed(cp->tolerance * 0.001, 100, solverTimeout.getValue());	// tol, maxIt, timeout

        s_mtx.unlock();

        // Restore Dfree
        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                cp->getDfree()[rowIt.index()] -= computeDot<DataTypes>(colIt.val(), dx[colIt.index()]);
            }
        }

        VecDeriv tempForces;
        tempForces.resize(val.size());

        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            if (cp->getF()[rowIt.index()] != 0.0)
            {
                MatrixDerivColConstIterator colItEnd = rowIt.end();

                for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
                {
                    tempForces[colIt.index()] += colIt.val() * cp->getF()[rowIt.index()];
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
    if (sofa::simulation::AnimateEndEvent::checkEventType(event))
        return;

    if (!constraintSolver)
        return;

    if (!mState)
        return;

    component::constraintset::ConstraintProblem* new_cp = constraintSolver->getConstraintProblem();
    
    if (!new_cp)
        return;

    // Find available buffer

    unsigned char buf_index=0;
    unsigned char cbuf_index=mCurBufferId;
    unsigned char nbuf_index=mNextBufferId;

    if (buf_index == cbuf_index || buf_index == nbuf_index)
        buf_index++;

    if (buf_index == cbuf_index || buf_index == nbuf_index)
        buf_index++;

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

    const MatrixDeriv& c = mState->read(core::ConstMatrixDerivId::holonomicC())->getValue()   ;

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        constraints.addLine(rowIt.index(), rowIt.row());
    }

    // valid buffer
    mNextBufferId = buf_index;

    // Lock lcp to prevent its use by the SOFA thread while it is used by haptic thread
    if(mIsCuBufferInUse)
        constraintSolver->lockConstraintProblem(mCP[mCurBufferId], mCP[mNextBufferId]);
    else
        constraintSolver->lockConstraintProblem(mCP[mNextBufferId]);
}


//
// Those functions are here for compatibility with the sofa::component::controller::Forcefeedback scheme
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


#ifndef SOFA_DOUBLE

template <>
void SOFA_HAPTICS_API LCPForceFeedback< sofa::defaulttype::Rigid3fTypes >::computeForce(SReal x, SReal y, SReal z, SReal, SReal, SReal, SReal, SReal& fx, SReal& fy, SReal& fz);

#endif // SOFA_DOUBLE

#ifndef SOFA_FLOAT

template <>
void SOFA_HAPTICS_API LCPForceFeedback< sofa::defaulttype::Rigid3dTypes >::computeForce(double x, double y, double z, double, double, double, double, double& fx, double& fy, double& fz);

template <>
void SOFA_HAPTICS_API LCPForceFeedback< sofa::defaulttype::Rigid3dTypes >::computeWrench(const sofa::defaulttype::SolidTypes<double>::Transform &world_H_tool,
        const sofa::defaulttype::SolidTypes<double>::SpatialVector &/*V_tool_world*/,
        sofa::defaulttype::SolidTypes<double>::SpatialVector &W_tool_world );

#endif // SOFA_FLOAT


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_INL
