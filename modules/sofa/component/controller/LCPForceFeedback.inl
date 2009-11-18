
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/component/mastersolver/MasterContactSolver.h>

#include <sofa/core/objectmodel/BaseContext.h>

#include <algorithm>

namespace
{
template <typename DataType>
bool derivVectors(const typename DataType::VecCoord& x0, const typename DataType::VecCoord& x1, typename DataType::VecDeriv& d)
{
    unsigned int sz0 = x0.size();
    unsigned int szmin = std::min(sz0,x1.size());

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

template <typename DataType>
bool derivRigid3Vectors(const typename DataType::VecCoord& x0, const typename DataType::VecCoord& x1, typename DataType::VecDeriv& d)
{
    unsigned int sz0 = x0.size();
    unsigned int szmin = std::min(sz0,x1.size());

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


template <typename DataType>
double computeDot(const typename DataType::Deriv& v0, const typename DataType::Deriv& v1)
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

template <class DataType>
LCPForceFeedback<DataType>::LCPForceFeedback()
    : forceCoef(initData(&forceCoef, 0.03, "forceCoef","multiply haptic force by this coef."))
{
    this->f_listening.setValue(true);
    mLcp[0] = 0;
    mLcp[1] = 0;
    mLcp[2] = 0;
    mCurBufferId = 0;
    mNextBufferId = 0;
    mIsCuBufferInUse = false;
}


template <class DataType>
void LCPForceFeedback<DataType>::init()
{
    core::objectmodel::BaseContext* c = this->getContext();

    this->ForceFeedback::init();
    if(!c)
    {
        serr << "LCPForceFeedback has no current context. Initialisation failed." << sendl;
        return;
    }

    mastersolver = c->get< odesolver::MasterContactSolver >();

    if (!mastersolver)
    {
        serr << "LCPForceFeedback has no binding MasterContactSolver. Initialisation failed." << sendl;
        return;
    }

    mState = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataType> *> (c->getMechanicalState());
    if (!mState)
    {
        serr << "LCPForceFeedback has no binding MechanicalState. Initialisation failed." << sendl;
        return;
    }

    //sout << "init LCPForceFeedback done " << sendl;
};



template <class DataType>
void LCPForceFeedback<DataType>::computeForce(const typename DataType::VecCoord& state, typename DataType::VecDeriv& forces)
{
    const unsigned int stateSize = state.size();
    // Resize du vecteur force. Initialization à 0 ?
    forces.resize(stateSize);


    if(!mastersolver||!mState)
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

    typename DataType::VecConst& constraints = mConstraints[mCurBufferId];
    std::vector<int> &id_buf = mId_buf[mCurBufferId];
    typename DataType::VecCoord &val = mVal[mCurBufferId];
    component::odesolver::LCP* lcp = mLcp[mCurBufferId];

    if(!lcp)
    {
        mIsCuBufferInUse = false;
        return;
    }



    const unsigned int numConstraints = constraints.size();

    if((lcp->getMu() > 0.0) && (numConstraints!=0))
    {
        typename DataType::VecDeriv dx;
        derivVectors<DataType>(val,state,dx);

        // Modify Dfree
        for(unsigned int c1 = 0; c1 < numConstraints; c1++)
        {
            int indexC1 = id_buf[c1];
            ConstraintIterator itConstraint;
            std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c1].data();

            for(itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
            {
                lcp->getDfree()[indexC1] += computeDot<DataType>(itConstraint->second, dx[itConstraint->first]);
            }
        }

        double tol = lcp->getTolerance();
        int max = 100;

        tol *= 0.001;

        helper::nlcp_gaussseidelTimed(lcp->getNbConst(), lcp->getDfree(), lcp->getW(), lcp->getF(), lcp->getMu(), tol, max, true, 0.0008);

        // Restore Dfree
        for(unsigned int c1 = 0; c1 < numConstraints; c1++)
        {
            int indexC1 = id_buf[c1];

            ConstraintIterator itConstraint;
            std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c1].data();

            for(itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
            {
                lcp->getDfree()[indexC1] -= computeDot<DataType>(itConstraint->second, dx[itConstraint->first]);
            }
        }

        for(unsigned int c1 = 0; c1 < numConstraints; c1++)
        {
            int indexC1 = id_buf[c1];
            if (lcp->getF()[indexC1] != 0.0)
            {
                ConstraintIterator itConstraint;
                std::pair< ConstraintIterator, ConstraintIterator > iter=constraints[c1].data();

                for(itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
                {
                    forces[itConstraint->first]/*.x()*/ += itConstraint->second/*.x()*/ * lcp->getF()[indexC1];
                }
            }
        }
        for(unsigned int i=0; i<stateSize; ++i)
        {
            forces[i] *= forceCoef.getValue();
        }
    }
    mIsCuBufferInUse = false;
}

template <typename DataType>
void LCPForceFeedback<DataType>::handleEvent(sofa::core::objectmodel::Event *event)
{
    if(!dynamic_cast<sofa::simulation::AnimateEndEvent*>(event))
        return;

    if(!mastersolver)
        return;
    if(!mState)
        return;
    component::odesolver::LCP* new_lcp = mastersolver->getLCP();
    if(!new_lcp)
        return;

    // Find available buffer

    unsigned char buf_index=0;
    unsigned char cbuf_index=mCurBufferId;
    unsigned char nbuf_index=mNextBufferId;
    if(buf_index==cbuf_index||buf_index==nbuf_index)
        buf_index++;
    if(buf_index==cbuf_index||buf_index==nbuf_index)
        buf_index++;

    // Compute constraints, id_buf lcp and val for the current lcp.

    typename DataType::VecConst& constraints = mConstraints[buf_index];
    std::vector<int>& id_buf = mId_buf[buf_index];
    typename DataType::VecCoord& val = mVal[buf_index];

    // Update LCP
    mLcp[buf_index] = new_lcp;

    // Update Val
    val = *mState->getXfree();

    // Update constraints and id_buf
    constraints.clear();
    id_buf.clear();
    for(unsigned int c1 = 0; c1 < mState->getC()->size(); c1++)
    {
        int indexC1 = mState->getConstraintId()[c1];
        id_buf.push_back(indexC1);
        typename DataType::SparseVecDeriv v;
        ConstraintIterator itConstraint;

        std::pair< ConstraintIterator, ConstraintIterator > iter;
        iter.first=(*mState->getC())[c1].data().first;
        iter.second=(*mState->getC())[c1].data().second;

        for(itConstraint=iter.first; itConstraint!=iter.second; itConstraint++)
        {
            v.add(itConstraint->first, itConstraint->second);
        }
        constraints.push_back(v);
    }

    // valid buffer
    mNextBufferId = buf_index;

    // Lock lcp to prevent its use by the SOfa thread while it is used by haptic thread
    if(mIsCuBufferInUse)
        mastersolver->lockLCP(mLcp[mCurBufferId],mLcp[mNextBufferId]);
    else
        mastersolver->lockLCP(mLcp[mNextBufferId]);
}




//
// Those functions are here for compatibility with the sofa::component::controller::Forcefeedback scheme
//

template <typename DataType>
void LCPForceFeedback<DataType>::computeForce(double , double, double, double, double, double, double, double&, double&, double&)
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

} // namespace controller
} // namespace component
} // namespace sofa
