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
#ifndef SOFA_COMPONENT_COLLISION_MOUSEINTERACTOR_H
#define SOFA_COMPONENT_COLLISION_MOUSEINTERACTOR_H


#include <sofa/component/collision/initCollision.h>

#include <sofa/core/BehaviorModel.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/componentmodel/behavior/BaseForceField.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>


#include <sofa/component/collision/BarycentricContactMapper.h>
#include <sofa/component/collision/TopologicalChangeManager.h>
#include <sofa/component/collision/RayModel.h>
#include <sofa/component/forcefield/StiffSpringForceField.h>

#include <sofa/component/constraint/FixedConstraint.h>
#include <sofa/simulation/common/Node.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace collision
{

struct BodyPicked
{
    BodyPicked():body(NULL), mstate(NULL) {};
    sofa::core::CollisionModel *body;
    sofa::core::componentmodel::behavior::BaseMechanicalState *mstate;
    unsigned int indexCollisionElement;
    defaulttype::Vector3 point;
    double dist;
    double rayLength;
};

class SOFA_COMPONENT_COLLISION_API BaseMouseInteractor : public core::BehaviorModel
{
    typedef sofa::component::collision::RayModel MouseCollisionModel;
public:
    BaseMouseInteractor(): collisionModel(NULL),isAttached(false), isRemovingElement(false), isIncising(false),distanceFromMouse(0) {};

    void updatePosition( double dt);

    virtual void draw();

    void clear()
    {
        doReleaseBody();
        doReleaseFixations();
    };

    //Basic operations available with the Mouse
    /// Attach a body to the mouse
    virtual void doAttachBody(const BodyPicked& body, double stiffness)=0;
    /// Release the attached body
    virtual void doReleaseBody()=0;
    /// Remove the collision element under the mouse
    virtual void doRemoveCollisionElement(const BodyPicked& body)=0;
    /// Process to an incision
    virtual void doInciseBody(const helper::fixed_array< BodyPicked,2 > &incision)=0;

    virtual void doFixParticle(const BodyPicked& body, double stiffness)=0;
    virtual void doReleaseFixations()=0;

    SReal getDistanceFromMouse() const {return distanceFromMouse;};
    bool isMouseAttached() const { return isAttached;};


    void setMouseRayModel( component::collision::RayModel* model)
    {
        mouseCollision=model;
    }


    void setCollisionElement( sofa::core::CollisionModel *body, unsigned int index)
    {
        collisionModel=body;
        indexCollisionElement=index;
    }


protected:
    MouseCollisionModel  *mouseCollision;
    sofa::core::CollisionModel *collisionModel;
    unsigned int indexCollisionElement;

    helper::fixed_array< BodyPicked,2 > elementsPicked;

    bool isAttached;
    bool isRemovingElement;
    bool isIncising;

    SReal distanceFromMouse;

    sofa::component::collision::TopologicalChangeManager topologyChangeManager;
};



/**
 *  \brief class to execute specific tasks of the Mouse
 *
 */
template <class DataTypes>
class MouseInteractor : public BaseMouseInteractor
{
public:
    typedef sofa::component::container::MechanicalObject< DataTypes >         MouseContainer;
    typedef sofa::component::collision::BaseContactMapper< DataTypes >        MouseContactMapper;
    //typedef sofa::component::forcefield::VectorSpringForceField< DataTypes >  MouseForceField;
    typedef sofa::component::forcefield::StiffSpringForceField< DataTypes >   MouseForceField;

    typedef typename DataTypes::Coord Coord;
public:
    MouseInteractor():mouseInSofa(NULL), mapper(NULL), forcefield(NULL) {};
    ~MouseInteractor() { doReleaseBody();}

    void draw();

    void init();

    //Basic operations available with the Mouse
    /// Attach a body to the mouse
    void doAttachBody(const BodyPicked& body, double stiffness);
    /// Release the attached body
    void doReleaseBody();
    /// Remove the collision element under the mouse
    void doRemoveCollisionElement(const BodyPicked& body);
    /// Process to an incision
    void doInciseBody(const helper::fixed_array< BodyPicked,2 > &incision);
    /// Fix the particle picked
    void doFixParticle(const BodyPicked& body, double stiffness);
    /// Release the attached body
    void doReleaseFixations();

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }
    static std::string templateName(const MouseInteractor<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
protected:

    MouseContainer       *mouseInSofa;
    MouseContactMapper   *mapper;
    std::map< core::CollisionModel*, MouseContactMapper* > mapperFixations;
    MouseForceField      *forcefield;
    std::vector< simulation::Node * > fixations;
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_COLLISION_MOUSEINTERACTOR_CPP)
extern template class SOFA_COMPONENT_COLLISION_API MouseInteractor<defaulttype::Vec3Types>;
#endif



} // namespace collision

} // namespace component

} // namespace sofa

#endif
