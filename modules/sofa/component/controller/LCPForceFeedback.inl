
#ifndef SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_INL
#define SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_INL

#include <sofa/component/controller/LCPForceFeedback.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/component/constraintset/ConstraintSolverImpl.h>
#include <sofa/core/objectmodel/BaseContext.h>

#include <algorithm>

namespace
{
template <typename DataTypes>
bool derivVectors(const typename DataTypes::VecCoord& x0, const typename DataTypes::VecCoord& x1, typename DataTypes::VecDeriv& d)
{
    unsigned int sz0 = x0.size();
    unsigned int szmin = std::min(sz0,(unsigned int)x1.size());

    d.resize(sz0);
    for(unsigned int i=0; i<szmin; ++i)
    {
        d[i]=x1[i]-x0[i];
    }
    for(unsigned int i=szmin; i<sz0; ++i)
    {
        d[i]=-x0[i];
    }
    return true;
}

template <typename DataTypes>
bool derivRigid3Vectors(const typename DataTypes::VecCoord& x0, const typename DataTypes::VecCoord& x1, typename DataTypes::VecDeriv& d)
{
    unsigned int sz0 = x0.size();
    unsigned int szmin = std::min(sz0,(unsigned int)x1.size());

    d.resize(sz0);
    for(unsigned int i=0; i<szmin; ++i)
    {
        getVCenter(d[i]) = x1[i].getCenter() - x0[i].getCenter();
        // Pas de prise en charge des rotations
    }
    for(unsigned int i=szmin; i<sz0; ++i)
    {
        getVCenter(d[i]) = - x0[i].getCenter();
    }
    return true;
}


template <typename DataTypes>
double computeDot(const typename DataTypes::Deriv& v0, const typename DataTypes::Deriv& v1)
{
    return v0.x()*v1.x();
}


#ifndef SOFA_FLOAT
using sofa::defaulttype::Rigid3dTypes;
template<>
bool derivVectors<Rigid3dTypes>(const Rigid3dTypes::VecCoord& x0, const Rigid3dTypes::VecCoord& x1, Rigid3dTypes::VecDeriv& d)
{
    return derivRigid3Vectors<Rigid3dTypes>(x0,x1,d);
}
template <>
double computeDot<Rigid3dTypes>(const Rigid3dTypes::Deriv& v0, const Rigid3dTypes::Deriv& v1)
{
    return dot(getVCenter(v0),getVCenter(v1));
}

#endif
#ifndef SOFA_DOUBLE
using sofa::defaulttype::Rigid3fTypes;
template<>
bool derivVectors<Rigid3fTypes>(const Rigid3fTypes::VecCoord& x0, const Rigid3fTypes::VecCoord& x1, Rigid3fTypes::VecDeriv& d)
{
    return derivRigid3Vectors<Rigid3fTypes>(x0,x1,d);
}
template <>
double computeDot<Rigid3fTypes>(const Rigid3fTypes::Deriv& v0, const Rigid3fTypes::Deriv& v1)
{
    return dot(getVCenter(v0),getVCenter(v1));
}

#endif


}




namespace sofa
{
namespace component
{
namespace controller
{

template <class DataTypes>
LCPForceFeedback<DataTypes>::LCPForceFeedback()
    : f_activate(initData(&f_activate, false, "activate", "boolean to activate or deactivate the forcefeedback"))
    , forceCoef(initData(&forceCoef, 0.03, "forceCoef","multiply haptic force by this coef."))
    , haptic_freq(0.0)
{
    this->f_listening.setValue(true);
    mCP[0] = NULL;
    mCP[1] = NULL;
    mCP[2] = NULL;
    mCurBufferId = 0;
    mNextBufferId = 0;
    mIsCuBufferInUse = false;
    _timer = new CTime();

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
        serr << "LCPForceFeedback has no binding MasterContactSolver. Initialisation failed." << sendl;
        return;
    }

    mState = dynamic_cast<core::behavior::MechanicalState<DataTypes> *> (c->getMechanicalState());
    if (!mState)
    {
        serr << "LCPForceFeedback has no binding MechanicalState. Initialisation failed." << sendl;
        return;
    }

    //sout << "init LCPForceFeedback done " << sendl;
};



template <class DataTypes>
void LCPForceFeedback<DataTypes>::computeForce(const VecCoord& state,  VecDeriv& forces)
{
    const unsigned int stateSize = state.size();
    // Resize du vecteur force. Initialization � 0 ?
    forces.resize(stateSize);


    if(!constraintSolver||!mState)
        return;


    if (!f_activate.getValue())
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

        derivVectors< DataTypes >(val, state, dx);

        // Modify Dfree
        MatrixDerivRowConstIterator rowItEnd = constraints.end();

        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                cp->getDfree()[rowIt.index()] += computeDot<DataTypes>(colIt.val(), dx[colIt.index()]);
            }
        }

        // Solving constraints
        cp->solveTimed(cp->tolerance * 0.001, 100, 0.0008);	// tol, maxIt, timeout

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

    mIsCuBufferInUse = false;
}

template <typename DataTypes>
void LCPForceFeedback<DataTypes>::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (!dynamic_cast<sofa::simulation::AnimateEndEvent*>(event))
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

    const MatrixDeriv& c = *(mState->getC());

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
{}

#ifndef SOFA_DOUBLE
using sofa::defaulttype::Rigid3fTypes;
template <>
void LCPForceFeedback<Rigid3fTypes>::computeForce(SReal x, SReal y, SReal z, SReal, SReal, SReal, SReal, SReal& fx, SReal& fy, SReal& fz)
{
    Rigid3fTypes::VecCoord state;
    Rigid3fTypes::VecDeriv forces;
    state.resize(1);
    state[0].getCenter() = sofa::defaulttype::Vec3f((float)x,(float)y,(float)z);
    computeForce(state,forces);
    fx = getVCenter(forces[0]).x();
    fy = getVCenter(forces[0]).y();
    fz = getVCenter(forces[0]).z();
}
#endif

#ifndef SOFA_FLOAT
using sofa::defaulttype::Rigid3dTypes;
template <>
void LCPForceFeedback<Rigid3dTypes>::computeForce(double x, double y, double z, double, double, double, double, double& fx, double& fy, double& fz)
{
    Rigid3dTypes::VecCoord state;
    Rigid3dTypes::VecDeriv forces;
    state.resize(1);
    state[0].getCenter() = sofa::defaulttype::Vec3d(x,y,z);
    computeForce(state,forces);
    fx = getVCenter(forces[0]).x();
    fy = getVCenter(forces[0]).y();
    fz = getVCenter(forces[0]).z();
}
#endif

// 6D rendering of contacts
#ifndef SOFA_FLOAT
using sofa::defaulttype::Rigid3dTypes;
template <>
void LCPForceFeedback<Rigid3dTypes>::computeWrench(const SolidTypes<double>::Transform &world_H_tool,
        const SolidTypes<double>::SpatialVector &/*V_tool_world*/,
        SolidTypes<double>::SpatialVector &W_tool_world )
{
    //std::cerr<<"WARNING : LCPForceFeedback::computeWrench is not implemented"<<std::endl;

    if (!f_activate.getValue())
    {
        return;
    }


    Rigid3dTypes::VecCoord state;
    Rigid3dTypes::VecDeriv forces;
    state.resize(1);
    state[0].getCenter()	  = world_H_tool.getOrigin();
    state[0].getOrientation() = world_H_tool.getOrientation();


    computeForce(state,forces);

    W_tool_world.setForce(getVCenter(forces[0]));
    W_tool_world.setTorque(getVOrientation(forces[0]));



    //Vec3d Force(0.0,0.0,0.0);

    //this->computeForce(world_H_tool.getOrigin()[0], world_H_tool.getOrigin()[1],world_H_tool.getOrigin()[2],
    //				   world_H_tool.getOrientation()[0], world_H_tool.getOrientation()[1], world_H_tool.getOrientation()[2], world_H_tool.getOrientation()[3],
    //				   Force[0],  Force[1], Force[2]);

    //W_tool_world.setForce(Force);



};


#endif


template <typename DataTypes>
void LCPForceFeedback<DataTypes>::computeWrench(const SolidTypes<SReal>::Transform &,
        const SolidTypes<SReal>::SpatialVector &,
        SolidTypes<SReal>::SpatialVector & )
{}


} // namespace controller
} // namespace component
} // namespace sofa

#endif
