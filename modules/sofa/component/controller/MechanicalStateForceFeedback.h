#ifndef SOFA_COMPONENT_CONTROLLER_MECHANICALSTATEFORCEFEEDBACK_H
#define SOFA_COMPONENT_CONTROLLER_MECHANICALSTATEFORCEFEEDBACK_H

#include <sofa/component/component.h>
#include <sofa/component/controller/ForceFeedback.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/constraintset/ConstraintSolverImpl.h>

namespace sofa
{

namespace component
{

namespace controller
{

using namespace std;
using namespace helper::system::thread;
using namespace core::behavior;
using namespace core;


template<class TDataTypes>
class SOFA_COMPONENT_CONTROLLER_API MechanicalStateForceFeedback : public sofa::component::controller::ForceFeedback
{

public:

    SOFA_CLASS(SOFA_TEMPLATE(MechanicalStateForceFeedback,TDataTypes),sofa::component::controller::ForceFeedback);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

    simulation::Node *context;

    MechanicalStateForceFeedback(void) {};

    virtual void init() {context = dynamic_cast<simulation::Node *>(this->getContext());};
    virtual void computeForce(SReal x, SReal y, SReal z, SReal u, SReal v, SReal w, SReal q, SReal& fx, SReal& fy, SReal& fz) = 0;
    virtual void computeForce(const  VecCoord& state,  VecDeriv& forces) = 0;
    virtual void computeWrench(const SolidTypes<SReal>::Transform &, const SolidTypes<SReal>::SpatialVector &, SolidTypes<SReal>::SpatialVector & )=0;

    virtual void setReferencePosition(SolidTypes<SReal>::Transform& /*referencePosition*/) {};
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif
