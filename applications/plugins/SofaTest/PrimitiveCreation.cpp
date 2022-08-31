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

#include "PrimitiveCreation.h"
#include <sofa/component/statecontainer/MechanicalObject.h>

typedef sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> MechanicalObject3;
typedef sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Rigid3Types> MechanicalObjectRigid3;

using sofa::core::objectmodel::New;
using sofa::component::statecontainer::MechanicalObject;
using namespace sofa::type;
using namespace sofa::defaulttype;

namespace sofa{

     namespace PrimitiveCreationTest{

void rotx(double ax,Vec3 & x,Vec3 & y,Vec3 & z){
    Vec3 ix = Vec3(1,0,0);

    Quat<SReal> rotx(ix,ax);

    x = rotx.rotate(x);y = rotx.rotate(y);z = rotx.rotate(z);
}

void roty(double angle,Vec3 & x,Vec3 & y,Vec3 & z){
    Vec3 iy = Vec3(0,1,0);

    Quat<SReal> rot(iy,angle);

    x = rot.rotate(x);y = rot.rotate(y);z = rot.rotate(z);
}

void rotz(double angle,Vec3 & x,Vec3 & y,Vec3 & z){
    Vec3 iz = Vec3(0,0,1);

    Quat<SReal> rot(iz,angle);

    x = rot.rotate(x);y = rot.rotate(y);z = rot.rotate(z);
}

sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>::SPtr makeTri(const Vec3 & p0,const Vec3 & p1,const Vec3 & p2,const Vec3 & v, sofa::simulation::Node::SPtr &father){
    //creating node containing TriangleModel
    sofa::simulation::Node::SPtr tri = father->createChild("tri");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObject3::SPtr triDOF = New<MechanicalObject3>();

    //editing DOF related to the TriangleCollisionModel<sofa::defaulttype::Vec3Types> to be created, size is 3 (3 points) because it contains just one Triangle
    triDOF->resize(3);
    Data<MechanicalObject3::VecCoord> & dpositions = *triDOF->write( sofa::core::VecId::position() );
    MechanicalObject3::VecCoord & positions = *dpositions.beginEdit();

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = p0;
    positions[1] = p1;
    positions[2] = p2;

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObject3::VecDeriv> & dvelocities = *triDOF->write( sofa::core::VecId::velocity() );

    MechanicalObject3::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    velocities[1] = v;
    velocities[2] = v;

    dvelocities.endEdit();

    tri->addObject(triDOF);

    //creating a topology necessary for capsule
    sofa::component::topology::container::constant::MeshTopology::SPtr bmt = New<sofa::component::topology::container::constant::MeshTopology>();
    bmt->addTriangle(0,1,2);
    bmt->addEdge(0,1);
    bmt->addEdge(1,2);
    bmt->addEdge(2,0);
    tri->addObject(bmt);

    //creating an OBBCollisionModel<sofa::defaulttype::Rigid3Types> and attaching it to the same node than obbDOF
    sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>::SPtr triCollisionModel = New<sofa::component::collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>>();
    tri->addObject(triCollisionModel);


    //editting the OBBModel
    triCollisionModel->init();

    return triCollisionModel;
}

sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr makeRigidSphere(const Vec3 & p,SReal radius,const Vec3 &v,const double *angles,const int *order,
                                                                            sofa::simulation::Node::SPtr & father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr sphere = father->createChild("sphere");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObjectRigid3::SPtr sphereDOF = New<MechanicalObjectRigid3>();

    //editing DOF related to the OBBCollisionModel<sofa::defaulttype::Rigid3Types> to be created, size is 1 because it contains just one OBB
    sphereDOF->resize(1);
    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *sphereDOF->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    //we create a frame that we will rotate like it is specified by the parameters angles and order
    Vec3 x(1,0,0);
    Vec3 y(0,1,0);
    Vec3 z(0,0,1);

    //creating an array of functions which are the rotation so as to perform the rotations in a for loop
    typedef void (*rot)(double,Vec3&,Vec3&,Vec3&);
    rot rotations[3];
    rotations[0] = &rotx;
    rotations[1] = &roty;
    rotations[2] = &rotz;

    //performing the rotations of the frame x,y,z
    for(int i = 0 ; i < 3 ; ++i)
        (*rotations[order[i]])(angles[order[i]],x,y,z);


    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = Rigid3Types::Coord(p, Quat<SReal>::createQuaterFromFrame(x,y,z));

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *sphereDOF->write( sofa::core::VecId::velocity() );

    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();


    sphere->addObject(sphereDOF);

    //creating an RigidSphereModel and attaching it to the same node than obbDOF
    sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphereCollisionModel = New<sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Rigid3Types>>();
    sphere->addObject(sphereCollisionModel);

    //editing the RigidSphereModel
    sphereCollisionModel->init();
    sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Rigid3Types>::VecReal & vecRad = *(sphereCollisionModel->radius.beginEdit());

    vecRad[0] = radius;

    sphereCollisionModel->radius.endEdit();

    return sphereCollisionModel;
}


sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr makeSphere(const Vec3 & p,SReal radius,const Vec3 & v,sofa::simulation::Node::SPtr & father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr sphere = father->createChild("sphere");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObject3::SPtr sphereDOF = New<MechanicalObject3>();

    //editing DOF related to the OBBCollisionModel<sofa::defaulttype::Rigid3Types> to be created, size is 1 because it contains just one OBB
    sphereDOF->resize(1);
    Data<MechanicalObject3::VecCoord> & dpositions = *sphereDOF->write( sofa::core::VecId::position() );
    MechanicalObject3::VecCoord & positions = *dpositions.beginEdit();

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = p;

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObject3::VecDeriv> & dvelocities = *sphereDOF->write( sofa::core::VecId::velocity() );

    MechanicalObject3::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();


    sphere->addObject(sphereDOF);

    //creating an RigidSphereModel and attaching it to the same node than obbDOF
    sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphereCollisionModel = New<sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>>();
    sphere->addObject(sphereCollisionModel);

    //editting the RigidSphereModel
    sphereCollisionModel->init();
    sofa::component::collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>::VecReal & vecRad = *(sphereCollisionModel->radius.beginEdit());

    vecRad[0] = radius;

    sphereCollisionModel->radius.endEdit();

    return sphereCollisionModel;
}

}// namespace PrimitiveCreationTest
}//namespace sofa

