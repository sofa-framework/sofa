#include "BulletTriangleModel.h"
#include "btBulletCollisionCommon.h"
#include "BtDynamicMesh.h"

namespace sofa
{

namespace component
{

namespace collision
{

template <class DataTypes>
TBulletTriangleModel<DataTypes>::TBulletTriangleModel()
    : TTriangleModel<DataTypes>()
    , margin(initData(&margin, (SReal)0.04, "margin","Margin used for collision detection within bullet"))
    , _bt_mesh(0x0)
    , _bt_gmesh(0x0)
{}


static btRigidBody* localCreateRigidBody(float mass, const btTransform& startTransform,btCollisionShape* shape,float /*processingThreshold*/)
{
    btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

    //rigidbody is dynamic if and only if mass is non zero, otherwise static
//    bool isDynamic = (mass != 0.f);

    btVector3 localInertia(0,0,0);
//    if (isDynamic)
//        shape->calculateLocalInertia(mass,localInertia);

    //using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects

    btRigidBody* body = new btRigidBody(mass,0,shape,localInertia);

    body->setWorldTransform(startTransform);
 //	body->setContactProcessingThreshold(0.5f);
	
    return body;
}

template <class DataTypes>
void TBulletTriangleModel<DataTypes>::initBullet(){
    sofa::core::objectmodel::BaseObject::f_listening.setValue(true);
    _bt_mesh = new btTriangleMesh();

    const SeqTriangles & tri = *(this->triangles);//this->_topology->getTriangles();
    const VecCoord & pos = mstate->read(core::ConstVecCoordId::position())->getValue();
    int npoints = mstate->getSize();
    int nTri = _topology->getNbTriangles();

    _bt_mesh->preallocateIndices(nTri * 3);
    _bt_mesh->preallocateVertices(npoints);

    for(int i = 0 ; i < npoints ; ++i){
        btVector3 btP(pos[i][0],pos[i][1],pos[i][2]);
        _bt_mesh->findOrAddVertex(btP,false);
    }

    for(int i = 0 ; i < nTri ; ++i){
        _bt_mesh->addIndex(tri[i][0]);
        _bt_mesh->addIndex(tri[i][1]);
        _bt_mesh->addIndex(tri[i][2]);
    }
    _bt_mesh->getIndexedMeshArray()[0].m_numTriangles = nTri;

    //_bt_gmesh = new btBvhTriangleMeshShape(_bt_mesh,true,true);
    _bt_gmesh = new BtDynamicMesh(_bt_mesh);//new btGImpactMeshShape(_bt_mesh);//
    //_bt_gmesh->setMargin(this->getProximity());
    //double margin = 0.5;//0.5;
    _bt_gmesh->setMargin(margin.getValue());

    btTransform startTransform;
    startTransform.setIdentity();
    startTransform.setOrigin(btVector3(0,0,0));

    _bt_collision_object = localCreateRigidBody(1,startTransform,_bt_gmesh,margin.getValue());///PROCESSING THRESHOLD ??? CONTINUE HERE MORENO !!!!
}

template <class DataTypes>
void TBulletTriangleModel<DataTypes>::init(){
    TTriangleModel<DataTypes>::init();
    initBullet();
}


template <class DataTypes>
void TBulletTriangleModel<DataTypes>::reinit(){
    if(_bt_mesh)
        delete _bt_mesh;
    if(_bt_gmesh)
        delete _bt_gmesh;
    if(_bt_collision_object)
        delete _bt_collision_object;

    init();
}

//template <class MyReal,class ToRead,class ToFill>
//void myFillFunc(const ToRead & pos,int numverts,ToFill vertexbase,int vertexStride){
//    int mystride = vertexStride/sizeof(MyReal);
//    MyReal* castVertexBase = (MyReal*)(vertexbase);
//    for(int i = 0 ; i < numverts ; ++i){
//        castVertexBase[0] = pos[i][0];
//        castVertexBase[1] = pos[i][1];
//        castVertexBase[2] = pos[i][2];

//        castVertexBase += mystride;
//    }
//}

//template <class DataTypes>
//void TBulletTriangleModel<DataTypes>::updateBullet(){
//    //_bt_collision_object->setActivationState(DISABLE_SIMULATION);
//    unsigned char *vertexbase;
//    int numverts;
//    PHY_ScalarType type;
//    int vertexStride;
//    unsigned char *indexbase;
//    int indexstride;
//    int numfaces;
//    PHY_ScalarType indicestype;

//    _bt_mesh->getLockedVertexIndexBase(&vertexbase,numverts,type,vertexStride,&indexbase,indexstride,numfaces,indicestype);

//    const VecCoord & pos = mstate->read(core::ConstVecCoordId::position())->getValue();
//    assert(mstate->getSize() == numverts);

//    if(type == PHY_FLOAT){
//        myFillFunc<float>(pos,numverts,vertexbase,vertexStride);
//    }
//    else if(type == PHY_DOUBLE){
//        myFillFunc<double>(pos,numverts,vertexbase,vertexStride);
//    }
//    else{
//        std::cerr<<"in BulletTriangleModel.inl : not a double nor a float !"<<std::endl;
//        exit(-1);
//    }

//    _bt_gmesh->postUpdate();
//    //_bt_collision_object->setActivationState(ACTIVE_TAG);
//}


template <class DataTypes>
template <class MyReal,class ToRead,class ToFill>
void TBulletTriangleModel<DataTypes>::myFillFunc(const ToRead & pos,int numverts,ToFill vertexbase,int vertexStride){
    int mystride = vertexStride/sizeof(MyReal);
    MyReal* castVertexBase = (MyReal*)(vertexbase);
    for(int i = 0 ; i < numverts ; ++i){
        castVertexBase[0] = pos[i][0];
        castVertexBase[1] = pos[i][1];
        castVertexBase[2] = pos[i][2];

        castVertexBase += mystride;
    }
}

template <class DataTypes>
void TBulletTriangleModel<DataTypes>::updateBullet(){
    //_bt_collision_object->setActivationState(DISABLE_SIMULATION);
    unsigned char *vertexbase;
    int numverts;
    PHY_ScalarType type;
    int vertexStride;
    unsigned char *indexbase;
    int indexstride;
    int numfaces;
    PHY_ScalarType indicestype;

    _bt_mesh->getLockedVertexIndexBase(&vertexbase,numverts,type,vertexStride,&indexbase,indexstride,numfaces,indicestype);

    const VecCoord & pos = mstate->read(core::ConstVecCoordId::position())->getValue();
    assert(mstate->getSize() == numverts);

    if(type == PHY_FLOAT){
        myFillFunc<float>(pos,numverts,vertexbase,vertexStride);
    }
    else if(type == PHY_DOUBLE){
        myFillFunc<double>(pos,numverts,vertexbase,vertexStride);
    }
    else{
        std::cerr<<"in BulletTriangleModel.inl : not a double nor a float !"<<std::endl;
        exit(-1);
    }

    _bt_gmesh->postUpdate();
    //_bt_collision_object->setActivationState(ACTIVE_TAG);
    assert(goodSofaBulletLink());
}


template <class DataTypes>
void TBulletTriangleModel<DataTypes>::handleEvent(sofa::core::objectmodel::Event * ev){
    if(dynamic_cast<sofa::simulation::CollisionBeginEvent*>(ev)){
        updateBullet();
//        _bt_gmesh->updateBound();
//        _bt_gmesh->postUpdate();
    }
}


template <class MyReal,class ToRead,class ToFill>
bool sameVertices(const ToRead & pos,int numverts,ToFill vertexbase,int vertexStride){
    MyReal tol = 1e-5;
    int mystride = vertexStride/sizeof(MyReal);
    MyReal* castVertexBase = (MyReal*)(vertexbase);
    for(int i = 0 ; i < numverts ; ++i){
        for(int j = 0 ; j < 3 ; ++j){
            if(fabs(castVertexBase[j] - pos[i][j]) > tol){
                std::cerr<<"castVertexBase[j] "<<castVertexBase[j]<<" pos[i][j] "<<pos[i][j]<<std::endl;
                return false;
            }
        }

        castVertexBase += mystride;
    }

    return true;
}

template <class DataTypes>
bool TBulletTriangleModel<DataTypes>::goodSofaBulletLink()const{
    const VecCoord & s_X = this->getX();

    const unsigned char *vertexbase;
    int numverts;
    PHY_ScalarType type;
    int vertexStride;
    const unsigned char *indexbase;
    int indexstride;
    int numfaces;
    PHY_ScalarType indicestype;

    _bt_mesh->getLockedReadOnlyVertexIndexBase(&vertexbase,numverts,type,vertexStride,&indexbase,indexstride,numfaces,indicestype);

    assert(indicestype == PHY_INTEGER);

    if(type == PHY_DOUBLE){
        if(!sameVertices<double>(s_X,numverts,vertexbase,vertexStride)){
            return false;
        }
    }
    else{
        if(!sameVertices<float>(s_X,numverts,vertexbase,vertexStride)){
                    return false;
        }
    }

    const int * b_indices = (int*)indexbase;

    const sofa::core::topology::BaseMeshTopology::SeqTriangles & triz = *(this->triangles);

    for(unsigned int i = 0 ; i < triz.size() ; ++i){
        for(int j = 0 ; j < 3 ; ++j){
            if(triz[i][j] != (unsigned int)b_indices[i*3 + j])
                return false;
        }
    }

    return true;
}



//template <class DataTypes>
//void TBulletTriangleModel<DataTypes>::computeBoundingTree(int maxDepth=0){
//    (void)(maxDepth);


//}


}}}
