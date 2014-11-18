#include "BulletConvexHullModel.h"
#include <SofaBaseMechanics/MechanicalObject.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

template <class DataTypes>
TBulletConvexHullModel<DataTypes>::TBulletConvexHullModel()
    : margin(initData(&margin, (SReal)0.04, "margin","Margin used for collision detection within bullet"))
    , computeConvexHullDecomposition(initData(&computeConvexHullDecomposition,false,"computeConvexHullDecomposition","compute convex hull decomposition using HACD"))
    , drawConvexHullDecomposition(initData(&drawConvexHullDecomposition,false,"drawConvexHullDecomposition","draw convex hull decomposition using"))
    , CHPoints(initData(&CHPoints,"CHPoints", "points defining the convex hull"))
    , computeNormals(initData(&computeNormals, true, "computeNormals", "set to false to disable computation of triangles normal"))
    , positionDefined(initData(&positionDefined,false,"positionDefined","set to true if the collision model position is defined in the mechanical object" ))
    , concavityThreeshold(initData(&concavityThreeshold, (SReal)100, "concavityThreeshold","Threeshold used in the decomposition"))
    , _mstate(NULL)
{
    bmsh = 0x0;
    enum_type = -1;
}


template <class DataTypes>
void TBulletConvexHullModel<DataTypes>::resize(int size)
{
    this->core::CollisionModel::resize(size);
}


template <class DataTypes>
void TBulletConvexHullModel<DataTypes>::init(){
    this->CollisionModel::init();
    _mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes >* > (this->getContext()->getMechanicalState());

    if(_mstate == 0x0){
        std::cerr<<"BulletConvexHullModel requires a Rigid Mechanical Object"<<std::endl;
        return;
    }


    bmsh = 0x0;
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    bmsh = context->get<sofa::core::topology::BaseMeshTopology>(core::objectmodel::BaseContext::SearchDown);

    initBullet();
}


template <class DataTypes>
void TBulletConvexHullModel<DataTypes>::reinit(){
    this->CollisionModel::reinit();
}

static btRigidBody* localCreateRigidBody(float mass, const btTransform& startTransform,btCollisionShape* shape,float processingThreshold)
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
    body->setContactProcessingThreshold(processingThreshold);

    return body;
}


template <class DataTypes>
void TBulletConvexHullModel<DataTypes>::initBullet(){
    sofa::core::objectmodel::BaseObject::f_listening.setValue(true);
    VecCoord & ch_pts = (*CHPoints.beginEdit());

//    _bt_cshape.setMargin(margin.getValue());

    if(!positionDefined.getValue()){
        _bary.set(0,0,0);
        for(unsigned int i = 0 ; i < ch_pts.size() ; ++i){
            _bary += ch_pts[i];
        }

        _bary /= ch_pts.size();

        _mstate->resize(1);

        Data<typename sofa::component::container::MechanicalObject<DataTypes>::VecCoord>* dpositions = _mstate->write( sofa::core::VecId::position() );
        typename sofa::component::container::MechanicalObject<DataTypes>::VecCoord & positions = *dpositions->beginEdit();

        typename DataTypes::Coord one_position(_bary,Quaternion(0,0,0,1));
        positions[0] = one_position;

        dpositions->endEdit();

        for(unsigned int i = 0 ; i < ch_pts.size() ; ++i){
            ch_pts[i] -= _bary;
        }
    }

    const Quaternion & ori = _mstate->read(core::ConstVecCoordId::position())->getValue()[0].getOrientation();
    const Coord & center = _mstate->read(core::ConstVecCoordId::position())->getValue()[0].getCenter();

    _bt_trans.setRotation(btQuaternion(ori[0],ori[1],ori[2],ori[3]));
    _bt_trans.setOrigin(btVector3(center[0],center[1],center[2]));

    btTransform identity;
    identity.setIdentity();
    if(!computeConvexHullDecomposition.getValue()){

        btVector3 * bt_CHPoints = new btVector3[ch_pts.size()];

        for(unsigned int i = 0 ; i < ch_pts.size() ; ++i){
            bt_CHPoints[i][0] = ch_pts[i][0];
            bt_CHPoints[i][1] = ch_pts[i][1];
            bt_CHPoints[i][2] = ch_pts[i][2];
        }

        btConvexHullShape * ch_shape = new btConvexHullShape(reinterpret_cast<btScalar*>(bt_CHPoints),ch_pts.size());
        ch_shape->setMargin(margin.getValue());
        _garbage.push(ch_shape);

        _bt_cshape.addChildShape(identity,ch_shape);

        delete [] bt_CHPoints;
    }
    else{
        const core::topology::BaseMeshTopology::SeqTriangles & tri = bmsh->getTriangles();

        HACD::Vec3<HACD::Real> * hacd_pts = new HACD::Vec3<HACD::Real>[ch_pts.size()];
        HACD::Vec3<long> * hacd_tri = new HACD::Vec3<long>[tri.size()];
        HACD::HACD convex_decomposotion;

        for(unsigned int i = 0 ; i < ch_pts.size() ; ++i){
            hacd_pts[i].X() = ch_pts[i][0];
            hacd_pts[i].Y() = ch_pts[i][1];
            hacd_pts[i].Z() = ch_pts[i][2];
        }

        for(unsigned int i = 0 ; i < tri.size() ; ++i){
            hacd_tri[i].X() = tri[i][0];
            hacd_tri[i].Y() = tri[i][1];
            hacd_tri[i].Z() = tri[i][2];
        }

        convex_decomposotion.SetNPoints(ch_pts.size());
        convex_decomposotion.SetPoints(hacd_pts);
        convex_decomposotion.SetNTriangles(tri.size());
        convex_decomposotion.SetTriangles(hacd_tri);

        ///convex_decompisition parameters
//        convex_decomposotion.SetNVerticesPerCH(100);
//        convex_decomposotion.SetScaleFactor(1000.0);
        convex_decomposotion.SetConcavity(concavityThreeshold.getValue());
        //////////////////////////////////

        convex_decomposotion.Compute();

        int n_ch = convex_decomposotion.GetNClusters();

        unsigned int max_pts = 0;
        unsigned int max_tri = 0;
        for(int i = 0 ; i < n_ch ; ++i){
            max_pts = std::max(max_pts,(unsigned int)convex_decomposotion.GetNPointsCH(i));
            max_tri = std::max(max_tri,(unsigned int)convex_decomposotion.GetNTrianglesCH(i));
        }

        HACD::Vec3<HACD::Real> * deco_pts = new HACD::Vec3<HACD::Real>[max_pts];
        HACD::Vec3<long> * deco_tri = new HACD::Vec3<long>[tri.size()];
        btVector3 * bt_CHPoints = new btVector3[max_pts];

        ///for drawing
        if(drawConvexHullDecomposition.getValue()){
            _ch_deco_tri.resize(n_ch);
            _ch_deco_pts.resize(n_ch);
            _ch_deco_norms.resize(n_ch);
            _ch_deco_colors.resize(n_ch);
        }

        for(int i = 0 ; i < n_ch ; ++i){

            if(drawConvexHullDecomposition.getValue()){
                for(int j = 0 ; j < 3 ; ++j){
                    _ch_deco_colors[i][j] = (float)rand()/RAND_MAX;
                }

                float color_norm = _ch_deco_colors[i][0] + _ch_deco_colors[i][1] + _ch_deco_colors[i][2];

                for(int j = 0 ; j < 3 ; ++j)
                    _ch_deco_colors[i][j] /= color_norm;

                _ch_deco_colors[i][3] = 1;
            }


            convex_decomposotion.GetCH(i,deco_pts,deco_tri);

            int n_deco_pts = convex_decomposotion.GetNPointsCH(i);
            for(int j = 0 ; j < n_deco_pts ; ++j){
                bt_CHPoints[j][0] = deco_pts[j].X();
                bt_CHPoints[j][1] = deco_pts[j].Y();
                bt_CHPoints[j][2] = deco_pts[j].Z();
            }

            btConvexHullShape * ch_shape = new btConvexHullShape(reinterpret_cast<btScalar*>(bt_CHPoints),n_deco_pts);
            ch_shape->setMargin(margin.getValue());
            _garbage.push(ch_shape);

            _bt_cshape.addChildShape(identity,ch_shape);


            ///for drawing
            if(drawConvexHullDecomposition.getValue()){

                //convex hull decomposition points storage
                _ch_deco_pts[i].resize(n_deco_pts);

                for(int j = 0 ; j < n_deco_pts ; ++j){
                    _ch_deco_pts[i][j][0] = bt_CHPoints[j][0];
                    _ch_deco_pts[i][j][1] = bt_CHPoints[j][1];
                    _ch_deco_pts[i][j][2] = bt_CHPoints[j][2];
                }

                //convex hull decomposition triangles and normal storage
                int n_deco_tri = convex_decomposotion.GetNTrianglesCH(i);

                _ch_deco_tri[i].resize(n_deco_tri);
                _ch_deco_norms[i].resize(n_deco_tri);

                for(int j = 0 ; j < n_deco_tri ; ++j){
                    _ch_deco_tri[i][j][0] = deco_tri[j].X();
                    _ch_deco_tri[i][j][1] = deco_tri[j].Y();
                    _ch_deco_tri[i][j][2] = deco_tri[j].Z();

                    const Vector3 & pt1 = _ch_deco_pts[i][_ch_deco_tri[i][j][0]];
                    const Vector3 & pt2 = _ch_deco_pts[i][_ch_deco_tri[i][j][1]];
                    const Vector3 & pt3 = _ch_deco_pts[i][_ch_deco_tri[i][j][2]];

                    _ch_deco_norms[i][j] = cross(pt2-pt1,pt3-pt1);
                }
            }
        }

        delete [] deco_tri;
        delete [] deco_pts;
        delete [] hacd_pts;
        delete [] hacd_tri;
        delete [] bt_CHPoints;
    }

    CHPoints.endEdit();

    _bt_cshape.setMargin(margin.getValue());

    _bt_trans.setRotation(btQuaternion(ori[0],ori[1],ori[2],ori[3]));
    _bt_trans.setOrigin(btVector3(center[0],center[1],center[2]));

    _bt_collision_object = localCreateRigidBody(1,_bt_trans,&_bt_cshape,margin.getValue());///PROCESSING THRESHOLD ??? CONTINUE HERE MORENO !!!!
    _bt_collision_object->activate(true);
}

template <class DataTypes>
void TBulletConvexHullModel<DataTypes>::updateBullet(){
    const Quaternion & ori = _mstate->read(core::ConstVecCoordId::position())->getValue()[0].getOrientation();
    const Coord & center = _mstate->read(core::ConstVecCoordId::position())->getValue()[0].getCenter();

    _bt_trans.setIdentity();
    _bt_trans.setOrigin(btVector3(center[0],center[1],center[2]));
    _bt_trans.setRotation(btQuaternion(ori[0],ori[1],ori[2],ori[3]));


    _bt_collision_object->setWorldTransform(_bt_trans);        
}

template <class DataTypes>
void TBulletConvexHullModel<DataTypes>::draw_without_decomposition(const core::visual::VisualParams* vparams){
    const core::topology::BaseMeshTopology::SeqTriangles & tri = bmsh->getTriangles();

    const VecCoord & msh_pts = *(CHPoints.beginEdit());
    if (vparams->displayFlags().getShowCollisionModels())
    {
        vparams->drawTool()->setPolygonMode(2,true);
        vparams->drawTool()->setPolygonMode(1,false);

        std::vector< Vector3 > points;
        std::vector< sofa::defaulttype::Vec<3,int> > indices;
        std::vector< Vector3 > normals;

        int index=0;
        for (unsigned int i=0; i<tri.size(); i++)
        {
            const Coord & pt1 = msh_pts[tri[i][0]];
            const Coord & pt2 = msh_pts[tri[i][1]];
            const Coord & pt3 = msh_pts[tri[i][2]];

            normals.push_back(cross(pt2-pt1,pt3-pt1));
            normals[normals.size() - 1].normalize();

            points.push_back(pt1);
            points.push_back(pt2);
            points.push_back(pt3);
            indices.push_back(sofa::defaulttype::Vec<3,int>(index,index+1,index+2));
            index+=3;
        }

        vparams->drawTool()->setLightingEnabled(true);
        vparams->drawTool()->drawTriangles(points, indices, normals, Vec<4,float>(getColor4f()));
        vparams->drawTool()->setLightingEnabled(false);
        vparams->drawTool()->setPolygonMode(0,false);
    }

    CHPoints.endEdit();
}

template <class DataTypes>
void TBulletConvexHullModel<DataTypes>::draw_decomposition(const core::visual::VisualParams* vparams){
    if (vparams->displayFlags().getShowCollisionModels())
    {
        vparams->drawTool()->setPolygonMode(2,true);
        vparams->drawTool()->setPolygonMode(1,false);

        for (unsigned int i=0; i<_ch_deco_colors.size(); i++)
        {
            vparams->drawTool()->setLightingEnabled(true);
            vparams->drawTool()->drawTriangles(_ch_deco_pts[i], _ch_deco_tri[i], _ch_deco_norms[i], _ch_deco_colors[i]);
            vparams->drawTool()->setLightingEnabled(false);
            vparams->drawTool()->setPolygonMode(0,false);
        }
    }
}

template <class DataTypes>
void TBulletConvexHullModel<DataTypes>::draw(const core::visual::VisualParams* vparams){
    if(!bmsh)
        return;

    const Quaternion & ori = _mstate->read(core::ConstVecCoordId::position())->getValue()[0].getOrientation();
    const Coord & center = _mstate->read(core::ConstVecCoordId::position())->getValue()[0].getCenter();

    Matrix3 sofa_mat;
    ori.toMatrix(sofa_mat);
    sofa_mat.transpose();

    double open_gl_mat[16];
    int k = 0;

    for(int i = 0  ; i < 3 ; ++i){
        for(int j = 0 ; j < 3 ; ++j){
            open_gl_mat[k] = sofa_mat[i][j];
            ++k;
        }
        ++k;
    }

    open_gl_mat[12] = (double)center[0];
    open_gl_mat[13] = (double)center[1];
    open_gl_mat[14] = (double)center[2];
    open_gl_mat[15] = 1;

    open_gl_mat[3]  = 0;
    open_gl_mat[7]  = 0;
    open_gl_mat[11] = 0;

    glPushMatrix();

    ////then make rotation in opengl
    glMultMatrixd(open_gl_mat);

    if(drawConvexHullDecomposition.getValue())
        draw_decomposition(vparams);
    else
        draw_without_decomposition(vparams);

    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    glPopMatrix();
}


template <class DataTypes>
void TBulletConvexHullModel<DataTypes>::handleEvent(sofa::core::objectmodel::Event * ev){
    if(dynamic_cast<sofa::simulation::CollisionBeginEvent*>(ev)){      
        updateBullet();
    }
}

}
}
}
