#ifndef BULLET_TRIANGLE_MODEL
#define BULLET_TRIANGLE_MODEL

#include <SofaMeshCollision/TriangleModel.h>
#include "BulletCollisionModel.h"
#include <sofa/simulation/CollisionBeginEvent.h>
#include <BulletCollisionDetection/config.h>

namespace sofa
{

namespace component
{

namespace collision
{

template<class TDataTypes>
class TBulletTriangleModel : public sofa::component::collision::TTriangleModel<TDataTypes>,public BulletCollisionModel
{
public:
    //SOFA_CLASS2(SOFA_TEMPLATE(TBulletTriangleModel, TDataTypes),SOFA_TEMPLATE(sofa::component::collision::TTriangleModel, TDataTypes),BulletCollisionModel);
    SOFA_CLASS(SOFA_TEMPLATE(TBulletTriangleModel, TDataTypes),SOFA_TEMPLATE(sofa::component::collision::TTriangleModel, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    //typedef TBtTriangle<DataTypes> Element;

    sofa::core::objectmodel::Data<SReal> margin; ///< Margin used for collision detection within bullet

    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    virtual void initBullet();
    virtual void updateBullet();

    inline virtual void computeBoundingTree(int/* maxDepth*/){
        //_bt_collision_object
        _bt_gmesh->updateBound();
        //_bt_gmesh->refitTree();
        //_bt_gmesh->postUpdate();
    }

//    inline virtual void computeBoundingTree(int/* maxDepth*/){
//        //_bt_collision_object
//        //_bt_gmesh->updateBound();
//        const Vector3 & min = this->mstate->f_bbox.getValue().minBBox();
//        const Vector3 & max = this->mstate->f_bbox.getValue().maxBBox();

//        std::cout<<"min "<<min<<std::endl;
//        std::cout<<"max "<<max<<std::endl;

//        btVector3 btmin(min[0],min[1],min[2]);
//        btVector3 btmax(max[0],max[1],max[2]);

//        _bt_gmesh->refitTree(btmin,btmax);
//        //_bt_gmesh->postUpdate();
//    }

    //virtual void computeBoundingTree(int maxDepth=0);

    inline virtual ~TBulletTriangleModel(){
        delete _bt_mesh;
        delete _bt_gmesh;
    }

    virtual void init();

    virtual void reinit();

    void handleEvent(sofa::core::objectmodel::Event * ev);

 	inline virtual void setMargin(SReal m){ *margin.beginEdit() = m; margin.endEdit(); }

    bool goodSofaBulletLink()const;

protected:
    template <class MyReal,class ToRead,class ToFill>
    void myFillFunc(const ToRead & pos,int numverts,ToFill vertexbase,int vertexStride);


    btTriangleMesh * _bt_mesh;
    btGImpactMeshShape * _bt_gmesh;
    //btBvhTriangleMeshShape * _bt_gmesh;

    TBulletTriangleModel();
private:
    using sofa::component::collision::TTriangleModel<TDataTypes>::mstate;
    using sofa::component::collision::TTriangleModel<TDataTypes>::_topology;

    //SOFA_CLASS(SOFA_TEMPLATE(TTriangleModel, TDataTypes), core::CollisionModel);


};

typedef TBulletTriangleModel<defaulttype::Vec3Types> BulletTriangleModel;


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BULLETCOLLISIONDETECTION)
#ifndef SOFA_FLOAT
extern template class SOFA_BULLETCOLLISIONDETECTION_API TBulletTriangleModel<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BULLETCOLLISIONDETECTION_API TBulletTriangleModel<defaulttype::Vec3fTypes>;
#endif
#endif

}}}
#endif
