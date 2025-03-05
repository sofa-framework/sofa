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
#include <sofa/core/CollisionModel.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/type/RGBAColor.h>

using sofa::type::RGBAColor ;

namespace sofa::core
{

std::vector<int> BaseCollisionElementIterator::emptyVector; ///< empty vector to be able to initialize the iterator to an empty pair

void CollisionModel::bwdInit()
{
    getColor4f(); //init the color to default value

    if (l_collElemActiver.get() == nullptr)
    {
        myCollElemActiver = CollisionElementActiver::getDefaultActiver();
        msg_info() << "no CollisionElementActiver found." << this->getName();
    }
    else
    {
        myCollElemActiver = dynamic_cast<CollisionElementActiver *> (l_collElemActiver.get());

        if (myCollElemActiver == nullptr)
        {
            myCollElemActiver = CollisionElementActiver::getDefaultActiver();
            msg_error() << "no dynamic cast possible for CollisionElementActiver." << this->getName();
        }
        else
        {
            msg_info() << "CollisionElementActiver named" << l_collElemActiver.get()->getName() << " found !" << this->getName();
        }
    }

}

void CollisionModel::setColor4f(const float *c)
{
    color.setValue(sofa::type::RGBAColor(c[0],c[1],c[2],c[3]));
}


/// Get a color that can be used to display this CollisionModel
const float* CollisionModel::getColor4f()
{

    //TODO FIXME because of: https://github.com/sofa-framework/sofa/issues/64
    static const float defaultColorSimulatedMovingActive[4] = {1, 0.5f, 0, 1};

    static const float defaultColorSimulatedMoving[4] = {0.5f, 0.25f, 0, 1};

    static const float defaultColorSimulatedActive[4] = {1, 0, 0, 1};

    static const float defaultColorSimulated[4] = {0.5f, 0, 0, 1};

    static const float defaultColorMovingActive[4] = {0, 1, 0.5f, 1};

    static const float defaultColorMoving[4] = {0, 0.5f, 0.25f, 1};

    static const float defaultColorActive[4] = {0.5f, 0.5f, 0.5f, 1};

    static const float defaultColor[4] = {0.25f, 0.25f, 0.25f, 1};

    if (color.isSet())
        return color.getValue().data();
    else if (isSimulated())
        if (isMoving())
            if (isActive()) {setColor4f(defaultColorSimulatedMovingActive); return defaultColorSimulatedMovingActive;}
            else            {setColor4f(defaultColorSimulatedMoving); return defaultColorSimulatedMoving;}
        else if (isActive()) {setColor4f(defaultColorSimulatedActive); return defaultColorSimulatedActive;}
        else            {setColor4f(defaultColorSimulated); return defaultColorSimulated;}
    else if (isMoving())
        if (isActive()) {setColor4f(defaultColorMovingActive); return defaultColorMovingActive;}
        else            {setColor4f(defaultColorMoving); return defaultColorMoving;}
    else if (isActive()) {setColor4f(defaultColorActive); return defaultColorActive;}
    else            {setColor4f(defaultColor); return defaultColor;}
}

/// Constructor
CollisionModel::CollisionModel()
    : bActive(initData(&bActive, true, "active", "flag indicating if this collision model is active and should be included in default collision detections"))
    , bMoving(initData(&bMoving, true, "moving", "flag indicating if this object is changing position between iterations"))
    , bSimulated(initData(&bSimulated, true, "simulated", "flag indicating if this object is controlled by a simulation"))
    , bSelfCollision(initData(&bSelfCollision, false, "selfCollision", "flag indication if the object can self collide"))
    , proximity(initData(&proximity, 0.0_sreal, "proximity", "Distance to the actual (visual) surface"))
    , contactStiffness(initData(&contactStiffness, 10.0_sreal, "contactStiffness", "Contact stiffness"))
    , contactFriction(initData(&contactFriction, 0.0_sreal, "contactFriction", "Contact friction coefficient (dry or viscous or unused depending on the contact method)"))
    , contactRestitution(initData(&contactRestitution, 0.0_sreal, "contactRestitution", "Contact coefficient of restitution"))
    , contactResponse(initData(&contactResponse, "contactResponse", "if set, indicate to the ContactManager that this model should use the given class of contacts.\nNote that this is only indicative, and in particular if both collision models specify a different class it is up to the manager to choose."))
    , color(initData(&color, sofa::type::RGBAColor(1,0,0,1), "color", "color used to display the collision model if requested"))
    , group(initData(&group,"group","IDs of the groups containing this model. No collision can occur between collision models included in a common group (e.g. allowing the same object to have multiple collision models)"))
    , size(0)
    , d_numberOfContacts(initData(&d_numberOfContacts, (Size)0, "numberOfContacts", "Number of collision models this collision model is currently attached to"))
    , previous(initLink("previous", "Previous (coarser / upper / parent level) CollisionModel in the hierarchy."))
    , next(initLink("next", "Next (finer / lower / child level) CollisionModel in the hierarchy."))
    , userData(nullptr)
    , l_collElemActiver(initLink("collisionElementActiver", "CollisionElementActiver component that activates or deactivates collision element(s) during execution"))
{
    d_numberOfContacts.setReadOnly(true);
}

/// Set the previous (coarser / upper / parent level) CollisionModel in the hierarchy.
void CollisionModel::setPrevious(CollisionModel::SPtr val)
{
    const CollisionModel::SPtr p = previous.get();
    if (p == val) return;
    if (p)
    {
        if (p->next.get()) p->next.get()->previous.reset();
        p->next.set(nullptr);
    }
    if (val)
    {
        if (val->next.get()) val->next.get()->previous.set(nullptr);
    }
    previous.set(val);
    if (val)
        val->next.set(this);
}

/// Return the first (i.e. root) CollisionModel in the hierarchy.
CollisionModel* CollisionModel::getFirst()
{
    CollisionModel *cm = this;
    CollisionModel *cm2;
    while ((cm2 = cm->getPrevious())!=nullptr)
        cm = cm2;
    return cm;
}

/// Return the last (i.e. leaf) CollisionModel in the hierarchy.
CollisionModel* CollisionModel::getLast()
{
    CollisionModel *cm = this;
    CollisionModel *cm2;
    while ((cm2 = cm->getNext())!=nullptr)
        cm = cm2;
    return cm;
}

bool CollisionModel::canCollideWith(CollisionModel* model)
{
    if (model->getContext() == this->getContext()) // models are in the Node -> is self collision activated?
        return bSelfCollision.getValue();

    const auto& myGroups = this->group.getValue();
    if (myGroups.empty()) // a collision model without any group always collides
        return true;

    const auto& modelGroups = model->group.getValue();
    if (modelGroups.empty()) // a collision model without any group always collides
        return true;

    std::set<int>::const_iterator myGroupsFirst = myGroups.cbegin();
    const std::set<int>::const_iterator myGroupsLast = myGroups.cend();

    std::set<int>::const_iterator modelGroupsFirst = modelGroups.cbegin();
    const std::set<int>::const_iterator modelGroupsLast = modelGroups.cend();

    // Collision models don't collide if they have a common group
    while (myGroupsFirst != myGroupsLast && modelGroupsFirst != modelGroupsLast)
    {
        if (*myGroupsFirst < *modelGroupsFirst)
        {
            ++myGroupsFirst;
        }
        else if (*myGroupsFirst > *modelGroupsFirst)
        {
            ++modelGroupsFirst;
        }
        else
        {
            return false;
        }
    }

    return true;
}



bool CollisionModel::insertInNode( objectmodel::BaseNode* node )
{
    node->addCollisionModel(this);
    Inherit1::insertInNode(node);
    return true;
}

bool CollisionModel::removeInNode( objectmodel::BaseNode* node )
{
    node->removeCollisionModel(this);
    Inherit1::removeInNode(node);
    return true;
}
} // namespace sofa::core

