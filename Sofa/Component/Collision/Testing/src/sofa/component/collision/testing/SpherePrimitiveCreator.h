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
#pragma once

#include <sofa/component/statecontainer/MechanicalObject.h>

#include <sofa/component/collision/geometry/SphereModel.h>

#include<sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <array>

using sofa::type::Quat;
using sofa::type::Vec3;

typedef sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> MechanicalObject3;
typedef sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Rigid3Types> MechanicalObjectRigid3;

using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::New;
using sofa::component::statecontainer::MechanicalObject;
using namespace sofa::defaulttype;

namespace sofa::collision_test
{

inline sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr makeRigidSphere(const Vec3& p, SReal radius, const Vec3& v, const double* angles, const int* order,
    sofa::simulation::Node::SPtr& father)
{
    //creating node containing OBBModel
    const sofa::simulation::Node::SPtr sphere = father->createChild("sphere");

    //creating a mechanical object which will be attached to the OBBModel
    const MechanicalObjectRigid3::SPtr sphereDOF = New<MechanicalObjectRigid3>();

    //editing DOF related to the OBBCollisionModel<sofa::defaulttype::Rigid3Types> to be created, size is 1 because it contains just one OBB
    sphereDOF->resize(1);
    Data<MechanicalObjectRigid3::VecCoord>& dpositions = *sphereDOF->write(sofa::core::VecId::position());
    MechanicalObjectRigid3::VecCoord& positions = *dpositions.beginEdit();

    //we create a frame that we will rotate like it is specified by the parameters angles and order
    Vec3 x(1, 0, 0);
    Vec3 y(0, 1, 0);
    Vec3 z(0, 0, 1);

    std::vector<Vec3> axes;

    axes.push_back(x);
    axes.push_back(y);
    axes.push_back(z);

    //creating an array of functions which are the rotation so as to perform the rotations in a for loop
    auto rot = [](double angle, Vec3& x, Vec3& y, Vec3& z, Vec3 axis)
    {
        const Quat<SReal> rotx(axis, angle);
        x = rotx.rotate(x); y = rotx.rotate(y); z = rotx.rotate(z);
    };

    //performing the rotations of the frame x,y,z

    for (int i = 0; i < 3; ++i)
        rot(angles[order[i]], x, y, z, axes[order[i]]);


    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = Rigid3Types::Coord(p, Quat<SReal>::createQuaterFromFrame(x, y, z));

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObjectRigid3::VecDeriv>& dvelocities = *sphereDOF->write(sofa::core::VecId::velocity());

    MechanicalObjectRigid3::VecDeriv& velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();


    sphere->addObject(sphereDOF);

    //creating an RigidSphereModel and attaching it to the same node than obbDOF
    sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphereCollisionModel = New<sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Rigid3Types>>();
    sphere->addObject(sphereCollisionModel);

    //editing the RigidSphereModel
    sphereCollisionModel->init();
    sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Rigid3Types>::VecReal& vecRad = *(sphereCollisionModel->radius.beginEdit());

    vecRad[0] = radius;

    sphereCollisionModel->radius.endEdit();

    return sphereCollisionModel;
}

inline sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr makeSphere(const Vec3& p, SReal radius, const Vec3& v, sofa::simulation::Node::SPtr& father)
{
    //creating node containing OBBModel
    const sofa::simulation::Node::SPtr sphere = father->createChild("sphere");

    //creating a mechanical object which will be attached to the OBBModel
    const MechanicalObject3::SPtr sphereDOF = New<MechanicalObject3>();

    //editing DOF related to the OBBCollisionModel<sofa::defaulttype::Rigid3Types> to be created, size is 1 because it contains just one OBB
    sphereDOF->resize(1);
    Data<MechanicalObject3::VecCoord>& dpositions = *sphereDOF->write(sofa::core::VecId::position());
    MechanicalObject3::VecCoord& positions = *dpositions.beginEdit();

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = p;

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObject3::VecDeriv>& dvelocities = *sphereDOF->write(sofa::core::VecId::velocity());

    MechanicalObject3::VecDeriv& velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();


    sphere->addObject(sphereDOF);

    //creating an RigidSphereModel and attaching it to the same node than obbDOF
    sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphereCollisionModel = New<sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>>();
    sphere->addObject(sphereCollisionModel);

    //editting the RigidSphereModel
    sphereCollisionModel->init();
    sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>::VecReal& vecRad = *(sphereCollisionModel->radius.beginEdit());

    vecRad[0] = radius;

    sphereCollisionModel->radius.endEdit();

    return sphereCollisionModel;
}

} // namespace sofa::collision_test
