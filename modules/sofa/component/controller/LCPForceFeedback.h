/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_H
#define SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_H

#include <sofa/component/component.h>
#include <sofa/component/controller/ForceFeedback.h>
#include <sofa/component/container/MechanicalObject.h>

namespace sofa
{

namespace component
{
namespace odesolver
{
class MasterContactSolver;
class LCP;
}

namespace controller
{
using namespace std;

/**
* LCP force field
*/
template <class DataType>
class SOFA_COMPONENT_CONTROLLER_API LCPForceFeedback : public sofa::component::controller::ForceFeedback
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(LCPForceFeedback,DataType),sofa::component::controller::ForceFeedback);
    typedef defaulttype::SparseConstraint<typename DataType::Deriv> SparseConstraint;
    typedef typename SparseConstraint::const_data_iterator ConstraintIterator;

    void init();
    Data<double> forceCoef;
    virtual void computeForce(double , double, double, double, double, double, double, double&, double&, double&);

    virtual void computeForce(const typename DataType::VecCoord& state, typename DataType::VecDeriv& forces);

    //void computeForce(double pitch0, double yaw0, double roll0, double z0, double pitch1, double yaw1, double roll1, double z1, double& fpitch0, double& fyaw0, double& froll0, double& fz0, double& fpitch1, double& fyaw1, double& froll1, double& fz1);

    LCPForceFeedback();
    // Saves all constraint from the simulation thread
    void handleEvent(sofa::core::objectmodel::Event *event);

protected:
    component::odesolver::LCP* lcp, *next_lcp;
    core::componentmodel::behavior::MechanicalState<DataType> *mState; ///< The omni try to follow this mechanical state.
    typename DataType::VecCoord mVal[3];
    typename DataType::VecConst mConstraints[3];
    std::vector<int> mId_buf[3];
    component::odesolver::LCP* mLcp[3];
    /* 	typename DataType::VecConst *constraint; */
    /* 	std::vector<int> *id_buf; */
    /* 	typename DataType::VecCoord *val; */
    unsigned char mNextBufferId; // Next buffer id to be use
    unsigned char mCurBufferId; // Current buffer id in use
    bool mIsCuBufferInUse; // Is current buffer currently in use right now


    //core::componentmodel::behavior::MechanicalState<defaulttype::Vec1dTypes> *mState1d; ///< The omni try to follow this mechanical state.
    sofa::component::odesolver::MasterContactSolver* mastersolver;
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif
