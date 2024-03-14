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

#include <sofa/component/collision/geometry/TriangleModel.h>

#include<sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <array>

using sofa::type::Vec3;

typedef sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> MechanicalObject3;
typedef sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Rigid3Types> MechanicalObjectRigid3;

using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::New;
using sofa::component::statecontainer::MechanicalObject;
using namespace sofa::defaulttype;

namespace sofa::collision_test
{

inline sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>::SPtr makeTri(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& v, sofa::simulation::Node::SPtr& father)
{
    //creating node containing TriangleModel
    const sofa::simulation::Node::SPtr tri = father->createChild("tri");

    //creating a mechanical object which will be attached to the OBBModel
    const MechanicalObject3::SPtr triDOF = New<MechanicalObject3>();

    //editing DOF related to the TriangleCollisionModel<sofa::defaulttype::Vec3Types> to be created, size is 3 (3 points) because it contains just one Triangle
    triDOF->resize(3);
    Data<MechanicalObject3::VecCoord>& dpositions = *triDOF->write(sofa::core::VecId::position());
    MechanicalObject3::VecCoord& positions = *dpositions.beginEdit();

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = p0;
    positions[1] = p1;
    positions[2] = p2;

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObject3::VecDeriv>& dvelocities = *triDOF->write(sofa::core::VecId::velocity());

    MechanicalObject3::VecDeriv& velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    velocities[1] = v;
    velocities[2] = v;

    dvelocities.endEdit();

    tri->addObject(triDOF);

    //creating a topology necessary for capsule
    const sofa::component::topology::container::constant::MeshTopology::SPtr bmt = New<sofa::component::topology::container::constant::MeshTopology>();
    bmt->addTriangle(0, 1, 2);
    bmt->addEdge(0, 1);
    bmt->addEdge(1, 2);
    bmt->addEdge(2, 0);
    tri->addObject(bmt);

    //creating an OBBCollisionModel<sofa::defaulttype::Rigid3Types> and attaching it to the same node than obbDOF
    sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>::SPtr triCollisionModel = New<sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>>();
    tri->addObject(triCollisionModel);


    //editting the OBBModel
    triCollisionModel->init();

    return triCollisionModel;
}

} // namespce sofa::collision_test
