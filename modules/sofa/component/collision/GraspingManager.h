#ifndef SOFA_COMPONENT_COLLISION_GRASPINGMANAGER_H
#define SOFA_COMPONENT_COLLISION_GRASPINGMANAGER_H

#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/collision/ContactManager.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>

#include <sofa/core/componentmodel/behavior/BaseController.h>
#include <set>

namespace sofa
{

namespace component
{

namespace collision
{

class GraspingManager : public core::componentmodel::behavior::BaseController
{
public:
    typedef TriangleModel::DataTypes DataTypes;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Real Real;

    typedef core::CollisionModel ToolModel;
    typedef core::componentmodel::behavior::MechanicalState<defaulttype::Vec1dTypes> ToolDOFs;

    Data < bool > active;
    Data < char > keyEvent;
    Data < char > keySwitchEvent;
    Data < double > openAngle;
    Data < double > closedAngle;

protected:
    std::set<ToolModel*> modelTools;
    ToolDOFs* mstateTool;
    core::componentmodel::collision::ContactManager* contactManager;
    bool wasActive;

public:
    GraspingManager();

    virtual ~GraspingManager();

    virtual void init();

    virtual void reset();

    virtual void handleEvent(sofa::core::objectmodel::Event* event);

    virtual void doGrasp();

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
