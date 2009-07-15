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
#include <sofa/component/collision/MouseInteractor.h>
#include <sofa/helper/system/gl.h>

namespace sofa
{

namespace component
{

namespace collision
{

//Basic operations available with the Mouse
/// Attach a body to the mouse
template <class DataTypes>
void MouseInteractor<DataTypes>::doAttachBody(const BodyPicked& picked, double stiffness)
{
    if (!picked.body) return;
    mapper = MouseContactMapper::Create(picked.body);
    if (!mapper)
    {
        std::cerr << "Problem with Mouse Mapper creation : " << std::endl;
        return;
    }

    std::string name = "contactMouse";
    core::componentmodel::behavior::MechanicalState<DataTypes>* mstateCollision = mapper->createMapping(name.c_str());
    mapper->resize(1);

    const typename DataTypes::Coord pointPicked=picked.point;
    const int idx=picked.indexCollisionElement;
    typename DataTypes::Real r=0.0;

    const int index = mapper->addPoint(pointPicked, idx, r);
    mapper->update();

    distanceFromMouse = picked.rayLength;
//         const defaulttype::Vector3 &orientation =  mouseCollision->getRay(0).direction();
//         const defaulttype::Vector3 repulsionVector = orientation*picked.dist;
    const double friction=0.0;

    forcefield = new MouseForceField(mouseInSofa, mstateCollision); forcefield->setName("Spring-Mouse-Contact");
    forcefield->addSpring(0,index, stiffness, friction, picked.dist);

    mouseCollision->getRay(0).origin() += mouseCollision->getRay(0).direction()*distanceFromMouse;

    sofa::core::BaseMapping *mapping;
    getContext()->get(mapping); assert(mapping);
    mapping->updateMapping();


    mstateCollision->getContext()->addObject(forcefield);
    forcefield->init();
    isAttached=true;
}

/// Release the attached body
template <class DataTypes>
void MouseInteractor<DataTypes>::doReleaseBody()
{
    if (mapper)
    {
        forcefield->cleanup();
        forcefield->getContext()->removeObject(forcefield);
        delete forcefield; forcefield=NULL;

        mapper->cleanup();
        delete mapper; mapper=NULL;
        distanceFromMouse = 0;
    }
    isAttached=false;
}


/// Remove the collision element under the mouse
template <class DataTypes>
void MouseInteractor<DataTypes>::doRemoveCollisionElement(const BodyPicked& picked)
{
    if (!picked.body) return;
    isRemovingElement=true;
}


/// Process to an incision
template <class DataTypes>
void MouseInteractor<DataTypes>::doInciseBody(const helper::fixed_array< BodyPicked,2 > &incision)
{

    if (!incision[0].body || incision[0].body != incision[1].body) return;
    isIncising = true;
    elementsPicked=incision;
}


template <class DataTypes>
void MouseInteractor<DataTypes>::init()
{
    BaseMouseInteractor::init();
    mouseInSofa = dynamic_cast< MouseContainer*>(this->getContext()->getMechanicalState());
    assert(mouseInSofa);

    this->getContext()->get(mouseCollision);
    assert(mouseCollision);
}


template <class DataTypes>
void MouseInteractor<DataTypes>::draw()
{
    if (forcefield)
    {
        bool b = forcefield->getContext()->getShowInteractionForceFields();
        forcefield->getContext()->setShowInteractionForceFields(true);
        forcefield->draw();
        forcefield->getContext()->setShowInteractionForceFields(b);
    }
    BaseMouseInteractor::draw();
}
}
}
}
