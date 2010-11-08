
#ifndef SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_INL
#define SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_INL

#include <sofa/component/controller/LCPForceFeedback.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/component/constraintset/LCPConstraintSolver.h>
#include <sofa/component/constraintset/GenericLCPConstraintSolver.h>

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
        d[i].getVCenter() = x1[i].getCenter() - x0[i].getCenter();
        // Pas de prise en charge des rotations
    }
    for(unsigned int i=szmin; i<sz0; ++i)
    {
        d[i].getVCenter() = - x0[i].getCenter();
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
    return dot(v0.getVCenter(),v1.getVCenter());
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
    return dot(v0.getVCenter(),v1.getVCenter());
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
    : forceCoef(initData(&forceCoef, 0.03, "forceCoef","multiply haptic force by this coef.")),
      haptic_freq(0.0)
{
    this->f_listening.setValue(true);
    mLcp[0] = 0;
    mLcp[1] = 0;
    mLcp[2] = 0;
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

    c->get(lcpconstraintSolver);

    if (!lcpconstraintSolver)
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
    // Resize du vecteur force. Initialization à 0 ?
    forces.resize(stateSize);


    if(!lcpconstraintSolver||!mState)
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
    component::constraintset::LCP* lcp = mLcp[mCurBufferId];

    if(!lcp)
    {
        mIsCuBufferInUse = false;
        return;
    }

    component::constraintset::GenericLCP* glcp = dynamic_cast<component::constraintset::GenericLCP*>(lcp);
    if((lcp->getMu() > 0.0 || glcp) && (!constraints.empty()))
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
                lcp->getDfree()[rowIt.index()] += computeDot<DataTypes>(colIt.val(), dx[colIt.index()]);
            }
        }

        double tol = lcp->getTolerance();
        int max = 100;

        tol *= 0.001;

        if(glcp != NULL)
        {
            glcp->setTolerance(tol);
            glcp->setMaxIter(max);
            glcp->gaussSeidel(0.0008);
        }
        else
            helper::nlcp_gaussseidelTimed(lcp->getNbConst(), lcp->getDfree(), lcp->getW(), lcp->getF(), lcp->getMu(), tol, max, true, 0.0008);

        // Restore Dfree
        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                lcp->getDfree()[rowIt.index()] -= computeDot<DataTypes>(colIt.val(), dx[colIt.index()]);
            }
        }

        VecDeriv tempForces;
        tempForces.resize(val.size());

        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            if (lcp->getF()[rowIt.index()] != 0.0)
            {
                MatrixDerivColConstIterator colItEnd = rowIt.end();

                for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
                {
                    tempForces[colIt.index()] += colIt.val() * lcp->getF()[rowIt.index()];
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

    if (!lcpconstraintSolver)
        return;

    if (!mState)
        return;

    component::constraintset::LCP* new_lcp = lcpconstraintSolver->getLCP();
    if (!new_lcp)
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
    mLcp[buf_index] = new_lcp;

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

    // Lock lcp to prevent its use by the SOfa thread while it is used by haptic thread
    if(mIsCuBufferInUse)
        lcpconstraintSolver->lockLCP(mLcp[mCurBufferId], mLcp[mNextBufferId]);
    else
        lcpconstraintSolver->lockLCP(mLcp[mNextBufferId]);
}




//
// Those functions are here for compatibility with the sofa::component::controller::Forcefeedback scheme
//

template <typename DataTypes>
void LCPForceFeedback<DataTypes>::computeForce(double , double, double, double, double, double, double, double&, double&, double&)
{}

#ifndef SOFA_DOUBLE
using sofa::defaulttype::Rigid3fTypes;
template <>
void LCPForceFeedback<Rigid3fTypes>::computeForce(double x, double y, double z, double, double, double, double, double& fx, double& fy, double& fz)
{
    Rigid3fTypes::VecCoord state;
    Rigid3fTypes::VecDeriv forces;
    state.resize(1);
    state[0].getCenter() = sofa::defaulttype::Vec3f((float)x,(float)y,(float)z);
    computeForce(state,forces);
    fx = forces[0].getVCenter().x();
    fy = forces[0].getVCenter().y();
    fz = forces[0].getVCenter().z();
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
    fx = forces[0].getVCenter().x();
    fy = forces[0].getVCenter().y();
    fz = forces[0].getVCenter().z();
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





    Vec3d Force(0.0,0.0,0.0);

    this->computeForce(world_H_tool.getOrigin()[0], world_H_tool.getOrigin()[1],world_H_tool.getOrigin()[2],
            world_H_tool.getOrientation()[0], world_H_tool.getOrientation()[1], world_H_tool.getOrientation()[2], world_H_tool.getOrientation()[3],
            Force[0],  Force[1], Force[2]);

    W_tool_world.setForce(Force);



};


#endif


template <typename DataTypes>
void LCPForceFeedback<DataTypes>::computeWrench(const SolidTypes<double>::Transform &,
        const SolidTypes<double>::SpatialVector &,
        SolidTypes<double>::SpatialVector & )
{}


} // namespace controller
} // namespace component
} // namespace sofa

#endif
