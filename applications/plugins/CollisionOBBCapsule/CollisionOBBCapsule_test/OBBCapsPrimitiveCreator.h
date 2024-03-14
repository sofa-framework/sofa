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

#include <sofa/component/topology/container/constant/MeshTopology.h>
#include <sofa/component/statecontainer/MechanicalObject.h>

#include <CollisionOBBCapsule/geometry/OBBModel.h>
#include <CollisionOBBCapsule/geometry/CapsuleModel.h>

#include<sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <array>

using sofa::type::Quat;
using sofa::type::Vec3;

typedef sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> MechanicalObject3;
typedef sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Rigid3Types> MechanicalObjectRigid3;

using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::New;
using namespace sofa::defaulttype;

namespace sofa
{
    
inline collisionobbcapsule::geometry::OBBCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr makeOBB(const Vec3 & p,const double *angles,const int *order,const Vec3 &v,const Vec3 &extents, sofa::simulation::Node::SPtr &father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr obb = father->createChild("obb");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObjectRigid3::SPtr obbDOF = New<MechanicalObjectRigid3>();

    //editing DOF related to the OBBCollisionModel<sofa::defaulttype::Rigid3Types> to be created, size is 1 because it contains just one OBB
    obbDOF->resize(1);
    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *obbDOF->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

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
        Quat<SReal> rotx(axis, angle);
        x = rotx.rotate(x); y = rotx.rotate(y); z = rotx.rotate(z);
    };

    //performing the rotations of the frame x,y,z

    for (int i = 0; i < 3; ++i)
        rot(angles[order[i]], x, y, z, axes[order[i]]);

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = Rigid3Types::Coord(p,Quat<SReal>::createQuaterFromFrame(x,y,z));

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *obbDOF->write( sofa::core::VecId::velocity() );

    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();


    obb->addObject(obbDOF);

    //creating an OBBCollisionModel<sofa::defaulttype::Rigid3Types> and attaching it to the same node than obbDOF
    collisionobbcapsule::geometry::OBBCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr obbCollisionModel = New<collisionobbcapsule::geometry::OBBCollisionModel<sofa::defaulttype::Rigid3Types>>();
    obb->addObject(obbCollisionModel);

    //editting the OBBModel
    obbCollisionModel->init();
    Data<collisionobbcapsule::geometry::OBBCollisionModel<sofa::defaulttype::Rigid3Types>::VecCoord> & dVecCoord = obbCollisionModel->writeExtents();
    collisionobbcapsule::geometry::OBBCollisionModel<sofa::defaulttype::Rigid3Types>::VecCoord & vecCoord = *(dVecCoord.beginEdit());

    vecCoord[0] = extents;

    dVecCoord.endEdit();

    return obbCollisionModel;
}


inline collisionobbcapsule::geometry::CapsuleCollisionModel<sofa::defaulttype::Vec3Types>::SPtr makeCap(const Vec3& p0, const Vec3& p1, double radius, const Vec3& v,
    sofa::simulation::Node::SPtr& father) {
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr cap = father->createChild("cap");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObject3::SPtr capDOF = New<MechanicalObject3>();

    //editing DOF related to the OBBCollisionModel<sofa::defaulttype::Rigid3Types> to be created, size is 1 because it contains just one OBB
    capDOF->resize(2);
    Data<MechanicalObject3::VecCoord>& dpositions = *capDOF->write(sofa::core::VecId::position());
    MechanicalObject3::VecCoord& positions = *dpositions.beginEdit();

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = p0;
    positions[1] = p1;

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObject3::VecDeriv>& dvelocities = *capDOF->write(sofa::core::VecId::velocity());

    MechanicalObject3::VecDeriv& velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    velocities[1] = v;
    dvelocities.endEdit();

    cap->addObject(capDOF);

    //creating a topology necessary for capsule
    sofa::component::topology::container::constant::MeshTopology::SPtr bmt = New<sofa::component::topology::container::constant::MeshTopology>();
    bmt->addEdge(0, 1);
    cap->addObject(bmt);

    //creating an OBBCollisionModel<sofa::defaulttype::Rigid3Types> and attaching it to the same node than obbDOF
    collisionobbcapsule::geometry::CapsuleCollisionModel<sofa::defaulttype::Vec3Types>::SPtr capCollisionModel = New<collisionobbcapsule::geometry::CapsuleCollisionModel<sofa::defaulttype::Vec3Types>>();
    cap->addObject(capCollisionModel);


    //editting the OBBModel
    capCollisionModel->init();
    Data<collisionobbcapsule::geometry::CapsuleCollisionModel<sofa::defaulttype::Vec3Types>::VecReal>& dVecReal = capCollisionModel->writeRadii();
    collisionobbcapsule::geometry::CapsuleCollisionModel<sofa::defaulttype::Vec3Types>::VecReal& vecReal = *(dVecReal.beginEdit());

    vecReal[0] = radius;

    dVecReal.endEdit();

    return capCollisionModel;
}

}
