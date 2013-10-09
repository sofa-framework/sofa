#include "PrimitiveCreation.h"

namespace sofa{

void rotx(double ax,Vec3d & x,Vec3d & y,Vec3d & z){
    Vec3d ix = Vec3d(1,0,0);

    Quaternion rotx(ix,ax);

    x = rotx.rotate(x);y = rotx.rotate(y);z = rotx.rotate(z);
}

void roty(double angle,Vec3d & x,Vec3d & y,Vec3d & z){
    Vec3d iy = Vec3d(0,1,0);

    Quaternion rot(iy,angle);

    x = rot.rotate(x);y = rot.rotate(y);z = rot.rotate(z);
}

void rotz(double angle,Vec3d & x,Vec3d & y,Vec3d & z){
    Vec3d iz = Vec3d(0,0,1);

    Quaternion rot(iz,angle);

    x = rot.rotate(x);y = rot.rotate(y);z = rot.rotate(z);
}


sofa::component::collision::OBBModel::SPtr makeOBB(const Vec3d & p,const double *angles,const int *order,const Vec3d &v,const Vec3d &extents, sofa::simulation::Node::SPtr &father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr obb = father->createChild("obb");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObjectRigid3::SPtr obbDOF = New<MechanicalObjectRigid3>();

    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
    obbDOF->resize(1);
    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *obbDOF->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    //we create a frame that we will rotate like it is specified by the parameters angles and order
    Vec3d x(1,0,0);
    Vec3d y(0,1,0);
    Vec3d z(0,0,1);

    //creating an array of functions which are the rotation so as to perform the rotations in a for loop
    typedef void (*rot)(double,Vec3d&,Vec3d&,Vec3d&);
    rot rotations[3];
    rotations[0] = &rotx;
    rotations[1] = &roty;
    rotations[2] = &rotz;

    //performing the rotations of the frame x,y,z
    for(int i = 0 ; i < 3 ; ++i)
        (*rotations[order[i]])(angles[order[i]],x,y,z);


    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = Rigid3Types::Coord(p,Quaternion::createQuaterFromFrame(x,y,z));

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *obbDOF->write( sofa::core::VecId::velocity() );

    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();


    obb->addObject(obbDOF);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::OBBModel::SPtr obbCollisionModel = New<sofa::component::collision::OBBModel >();
    obb->addObject(obbCollisionModel);

    //editting the OBBModel
    obbCollisionModel->init();
    Data<sofa::component::collision::OBBModel::VecCoord> & dVecCoord = obbCollisionModel->writeExtents();
    sofa::component::collision::OBBModel::VecCoord & vecCoord = *(dVecCoord.beginEdit());

    vecCoord[0] = extents;

    dVecCoord.endEdit();

    return obbCollisionModel;
}


sofa::component::collision::TriangleModel::SPtr makeTri(const Vec3d & p0,const Vec3d & p1,const Vec3d & p2,const Vec3d & v, sofa::simulation::Node::SPtr &father){
    //creating node containing TriangleModel
    sofa::simulation::Node::SPtr tri = father->createChild("tri");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObject3d::SPtr triDOF = New<MechanicalObject3d>();

    //editing DOF related to the TriangleModel to be created, size is 3 (3 points) because it contains just one Triangle
    triDOF->resize(3);
    Data<MechanicalObject3d::VecCoord> & dpositions = *triDOF->write( sofa::core::VecId::position() );
    MechanicalObject3d::VecCoord & positions = *dpositions.beginEdit();

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = p0;
    positions[1] = p1;
    positions[2] = p2;

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObject3d::VecDeriv> & dvelocities = *triDOF->write( sofa::core::VecId::velocity() );

    MechanicalObject3d::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    velocities[1] = v;
    velocities[2] = v;

    dvelocities.endEdit();

    tri->addObject(triDOF);

    //creating a topology necessary for capsule
    sofa::component::topology::MeshTopology::SPtr bmt = New<sofa::component::topology::MeshTopology>();
    bmt->addTriangle(0,1,2);
    bmt->addEdge(0,1);
    bmt->addEdge(1,2);
    bmt->addEdge(2,0);
    tri->addObject(bmt);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::TriangleModel::SPtr triCollisionModel = New<sofa::component::collision::TriangleModel >();
    tri->addObject(triCollisionModel);


    //editting the OBBModel
    triCollisionModel->init();

    return triCollisionModel;
}


sofa::component::collision::CapsuleModel::SPtr makeCap(const Vec3d & p0,const Vec3d & p1,double radius,const Vec3d & v,
                                                                   sofa::simulation::Node::SPtr & father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr cap = father->createChild("cap");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObject3d::SPtr capDOF = New<MechanicalObject3d>();

    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
    capDOF->resize(2);
    Data<MechanicalObject3d::VecCoord> & dpositions = *capDOF->write( sofa::core::VecId::position() );
    MechanicalObject3d::VecCoord & positions = *dpositions.beginEdit();

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = p0;
    positions[1] = p1;

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObject3d::VecDeriv> & dvelocities = *capDOF->write( sofa::core::VecId::velocity() );

    MechanicalObject3d::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    velocities[1] = v;
    dvelocities.endEdit();

    cap->addObject(capDOF);

    //creating a topology necessary for capsule
    sofa::component::topology::MeshTopology::SPtr bmt = New<sofa::component::topology::MeshTopology>();
    bmt->addEdge(0,1);
    cap->addObject(bmt);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::CapsuleModel::SPtr capCollisionModel = New<sofa::component::collision::CapsuleModel >();
    cap->addObject(capCollisionModel);


    //editting the OBBModel
    capCollisionModel->init();
    Data<sofa::component::collision::CapsuleModel::VecReal> & dVecReal = capCollisionModel->writeRadii();
    sofa::component::collision::CapsuleModel::VecReal & vecReal = *(dVecReal.beginEdit());

    vecReal[0] = radius;

    dVecReal.endEdit();

    return capCollisionModel;
}


sofa::component::collision::RigidSphereModel::SPtr makeRigidSphere(const Vec3d & p,SReal radius,const Vec3d &v,const double *angles,const int *order,
                                                                            sofa::simulation::Node::SPtr & father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr sphere = father->createChild("sphere");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObjectRigid3::SPtr sphereDOF = New<MechanicalObjectRigid3>();

    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
    sphereDOF->resize(1);
    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *sphereDOF->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    //we create a frame that we will rotate like it is specified by the parameters angles and order
    Vec3d x(1,0,0);
    Vec3d y(0,1,0);
    Vec3d z(0,0,1);

    //creating an array of functions which are the rotation so as to perform the rotations in a for loop
    typedef void (*rot)(double,Vec3d&,Vec3d&,Vec3d&);
    rot rotations[3];
    rotations[0] = &rotx;
    rotations[1] = &roty;
    rotations[2] = &rotz;

    //performing the rotations of the frame x,y,z
    for(int i = 0 ; i < 3 ; ++i)
        (*rotations[order[i]])(angles[order[i]],x,y,z);


    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = Rigid3Types::Coord(p,Quaternion::createQuaterFromFrame(x,y,z));

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *sphereDOF->write( sofa::core::VecId::velocity() );

    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();


    sphere->addObject(sphereDOF);

    //creating an RigidSphereModel and attaching it to the same node than obbDOF
    sofa::component::collision::RigidSphereModel::SPtr sphereCollisionModel = New<sofa::component::collision::RigidSphereModel >();
    sphere->addObject(sphereCollisionModel);

    //editting the RigidSphereModel
    sphereCollisionModel->init();
    sofa::component::collision::RigidSphereModel::VecReal & vecRad = *(sphereCollisionModel->radius.beginEdit());

    vecRad[0] = radius;

    sphereCollisionModel->radius.endEdit();

    return sphereCollisionModel;
}


sofa::component::collision::SphereModel::SPtr makeSphere(const Vec3d & p,SReal radius,const Vec3d & v,sofa::simulation::Node::SPtr & father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr sphere = father->createChild("sphere");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObject3::SPtr sphereDOF = New<MechanicalObject3>();

    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
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
    sofa::component::collision::SphereModel::SPtr sphereCollisionModel = New<sofa::component::collision::SphereModel >();
    sphere->addObject(sphereCollisionModel);

    //editting the RigidSphereModel
    sphereCollisionModel->init();
    sofa::component::collision::SphereModel::VecReal & vecRad = *(sphereCollisionModel->radius.beginEdit());

    vecRad[0] = radius;

    sphereCollisionModel->radius.endEdit();

    return sphereCollisionModel;
}

}

