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

#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace collision
{

struct BodyPicked
{
    BodyPicked():body(NULL) {};
    sofa::core::CollisionModel *body;
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

    //Basic operations available with the Mouse
    /// Attach a body to the mouse
    virtual void doAttachBody(const BodyPicked& body, double stiffness)=0;
    /// Release the attached body
    virtual void doReleaseBody()=0;
    /// Remove the collision element under the mouse
    virtual void doRemoveCollisionElement(const BodyPicked& body)=0;
    /// Process to an incision
    virtual void doInciseBody(const helper::fixed_array< BodyPicked,2 > &incision)=0;


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
class SOFA_COMPONENT_COLLISION_API MouseInteractor : public BaseMouseInteractor
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

    void reset() {doReleaseBody();};

    //Basic operations available with the Mouse
    /// Attach a body to the mouse
    void doAttachBody(const BodyPicked& body, double stiffness);
    /// Release the attached body
    void doReleaseBody();
    /// Remove the collision element under the mouse
    void doRemoveCollisionElement(const BodyPicked& body);
    /// Process to an incision
    void doInciseBody(const helper::fixed_array< BodyPicked,2 > &incision);


protected:

    MouseContainer       *mouseInSofa;
    MouseContactMapper   *mapper;
    MouseForceField      *forcefield;

};


} // namespace collision

} // namespace component

} // namespace sofa

#endif
