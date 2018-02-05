/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_H
#define SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_H
#include "config.h"

#include <SofaHaptics/MechanicalStateForceFeedback.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/helper/system/thread/CTime.h>


namespace sofa
{

namespace component
{

namespace constraintset { class ConstraintProblem; class ConstraintSolverImpl; }


namespace controller
{

/**
* LCP force field
*/
template <class TDataTypes>
class LCPForceFeedback : public sofa::component::controller::MechanicalStateForceFeedback<TDataTypes>
{
public:

    SOFA_CLASS(SOFA_TEMPLATE(LCPForceFeedback,TDataTypes),sofa::component::controller::MechanicalStateForceFeedback<TDataTypes>);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv::RowConstIterator MatrixDerivRowConstIterator;
    typedef typename DataTypes::MatrixDeriv::ColConstIterator MatrixDerivColConstIterator;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename DataTypes::MatrixDeriv::ColIterator MatrixDerivColIterator;

    void init() override;

    void draw( const core::visual::VisualParams* ) override
    {
        // draw the haptic_freq in the openGL window
        dmsg_info() << "haptic_freq = " << std::fixed << haptic_freq << " Hz   " << '\xd';
    }

    Data< double > forceCoef;
    //Data< double > momentCoef;

    Data< double > solverTimeout;

    // deriv (or not) the rotations when updating the violations
    Data <bool> d_derivRotations;

    // Enable/disable constraint haptic influence from all frames
    Data< bool > d_localHapticConstraintAllFrames;

    virtual void computeForce(SReal x, SReal y, SReal z, SReal u, SReal v, SReal w, SReal q, SReal& fx, SReal& fy, SReal& fz) override;
    virtual void computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &world_H_tool, const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &V_tool_world, sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world ) override;
    virtual void computeForce(const  VecCoord& state,  VecDeriv& forces) override;

    //void computeForce(double pitch0, double yaw0, double roll0, double z0, double pitch1, double yaw1, double roll1, double z1, double& fpitch0, double& fyaw0, double& froll0, double& fz0, double& fpitch1, double& fyaw1, double& froll1, double& fz1);
protected:
    LCPForceFeedback();
    virtual ~LCPForceFeedback()
    {
        delete(_timer);
    }

    virtual void updateStats();
    virtual bool updateConstraintProblem();
    virtual void doComputeForce(const  VecCoord& state,  VecDeriv& forces);


public:
    void handleEvent(sofa::core::objectmodel::Event *event) override;


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast< core::behavior::MechanicalState<DataTypes>* >(context->getMechanicalState()) == NULL)
            return false;
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const LCPForceFeedback<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    //component::constraintset::LCP* lcp, *next_lcp;
    core::behavior::MechanicalState<DataTypes> *mState; ///< The device try to follow this mechanical state.
    VecCoord mVal[3];
    MatrixDeriv mConstraints[3];
    std::vector<int> mId_buf[3];
    component::constraintset::ConstraintProblem* mCP[3];
    /* 	std::vector<int> *id_buf; */
    /* 	typename DataType::VecCoord *val; */
    unsigned char mNextBufferId; // Next buffer id to be use
    unsigned char mCurBufferId; // Current buffer id in use
    bool mIsCuBufferInUse; // Is current buffer currently in use right now

    //core::behavior::MechanicalState<defaulttype::Vec1dTypes> *mState1d; ///< The device try to follow this mechanical state.
    sofa::component::constraintset::ConstraintSolverImpl* constraintSolver;
    // timer: verifies the time rates of the haptic loop
    helper::system::thread::CTime *_timer;
    helper::system::thread::ctime_t time_buf;
    int timer_iterations;
    double haptic_freq;
    unsigned int num_constraints;
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_HAPTICS_API LCPForceFeedback<defaulttype::Vec1dTypes>;
extern template class SOFA_HAPTICS_API LCPForceFeedback<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_HAPTICS_API LCPForceFeedback<defaulttype::Vec1fTypes>;
extern template class SOFA_HAPTICS_API LCPForceFeedback<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_H
