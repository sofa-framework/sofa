#ifndef BULLET_OBB_MODEL
#define BULLET_OBB_MODEL

#include <SofaBaseCollision/OBBModel.h>
#include "BulletCollisionModel.h"
#include <sofa/simulation/CollisionBeginEvent.h>
#include <BulletCollisionDetection/config.h>
#include <stack>

namespace sofa
{

namespace component
{

namespace collision
{


//class SofaBox : public btBoxShape{
//public:

//    SofaBox( const btVector3& boxHalfExtents);

//    virtual void getAabb(const btTransform &t, btVector3 &aabbMin, btVector3 &aabbMax) const;
//};

template<class TDataTypes>
class TBulletOBBModel : public sofa::component::collision::TOBBModel<TDataTypes>,public BulletCollisionModel
{
public:
    //SOFA_CLASS2(SOFA_TEMPLATE(TBulletOBBModel, TDataTypes),SOFA_TEMPLATE(sofa::component::collision::TTriangleModel, TDataTypes),BulletCollisionModel);
    SOFA_CLASS(SOFA_TEMPLATE(TBulletOBBModel, TDataTypes),SOFA_TEMPLATE(sofa::component::collision::TOBBModel, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::Coord::Pos Coord;
    typedef helper::vector<Coord> VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::Quat Quaternion;
    //typedef TBtTriangle<DataTypes> Element;

    sofa::core::objectmodel::Data<SReal> margin; ///< Margin used for collision detection within bullet

    virtual void initBullet();
    virtual void updateBullet();

    inline virtual void computeBoundingTree(int/* maxDepth*/){
        _bt_cshape->recalculateLocalAabb();
    }

    inline virtual ~TBulletOBBModel();

    virtual void init();

    virtual void reinit();

    void handleEvent(sofa::core::objectmodel::Event * ev);

 	inline virtual void setMargin(SReal m){ *margin.beginEdit() = m; margin.endEdit(); }

protected:
    TBulletOBBModel();
    TBulletOBBModel(core::behavior::MechanicalState<DataTypes>* _mstate );

    std::stack<btCollisionShape*> _garbage;
    btCompoundShape * _bt_cshape;

    void cleanGarbage();
};

typedef TBulletOBBModel<defaulttype::RigidTypes> BulletOBBModel;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BULLETCOLLISIONDETECTION)
#ifndef SOFA_FLOAT
extern template class SOFA_BULLETCOLLISIONDETECTION_API TBulletOBBModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BULLETCOLLISIONDETECTION_API TBulletOBBModel<defaulttype::Rigid3fTypes>;
#endif
#endif

}}}
#endif
