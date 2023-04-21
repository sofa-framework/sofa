#ifndef BULLET_CAPSULE_MODEL
#define BULLET_CAPSULE_MODEL

#include <CollisionOBBCapsule/geometry/CapsuleModel.h>


#include "BulletCollisionModel.h"
#include <sofa/simulation/CollisionBeginEvent.h>
#include <CollisionOBBCapsule/geometry/CapsuleModel.h>
#include <CollisionOBBCapsule/geometry/RigidCapsuleModel.h>
#include <BulletCollisionDetection/config.h>
#include <stack>

namespace sofa
{

namespace component
{

namespace collision
{

class BulletSoftCapsule : public btCapsuleShape{
public:
    BulletSoftCapsule(btScalar radius,btScalar height) : btCapsuleShape(radius,height){}

    template <class MyReal>
    inline void setHeight(MyReal height){
        this->m_implicitShapeDimensions.setValue(getRadius(),0.5f*height,getRadius());
    }
};

template<class TDataTypes>
class TBulletCapsuleModel : public collisionobbcapsule::geometry::CapsuleCollisionModel<TDataTypes>,public BulletCollisionModel
{
public:
    //SOFA_CLASS2(SOFA_TEMPLATE(TBulletCapsuleModel, TDataTypes),SOFA_TEMPLATE(sofa::component::collision::TriangleCollisionModel, TDataTypes),BulletCollisionModel);
    SOFA_CLASS(SOFA_TEMPLATE(TBulletCapsuleModel, TDataTypes),SOFA_TEMPLATE(collisionobbcapsule::geometry::CapsuleCollisionModel, TDataTypes));

    using Inherit = collisionobbcapsule::geometry::CapsuleCollisionModel<TDataTypes>;

    //typedef typename GCapsuleCollisionModel<sofa::defaulttype::Vec3Types>::DataTypes DataTypes;
    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::Real Real;
    //typedef TBtTriangle<DataTypes> Element;

    sofa::core::objectmodel::Data<SReal> margin; ///< Margin used for collision detection within bullet

    virtual void initBullet();
    virtual void updateBullet();

    inline virtual void computeBoundingTree(int/* maxDepth*/){
        _bt_cshape->recalculateLocalAabb();
    }

    inline virtual ~TBulletCapsuleModel();

    virtual void init();

    virtual void reinit();

    void handleEvent(sofa::core::objectmodel::Event * ev);

    inline virtual void setMargin(SReal margin){_bt_cshape->setMargin(margin);}

protected:
    TBulletCapsuleModel();
    TBulletCapsuleModel(core::behavior::MechanicalState<DataTypes>* _mstate );

    std::stack<btCollisionShape*> _garbage;
    btCompoundShape * _bt_cshape;

    void cleanGarbage();

    static void makeBtQuat(const Coord & dir,btQuaternion & quat);
};

typedef TBulletCapsuleModel<defaulttype::Vec3Types> BulletCapsuleModel;
typedef TBulletCapsuleModel<defaulttype::RigidTypes> BulletRigidCapsuleModel;

#if  !defined(SOFA_BUILD_BULLETCOLLISIONDETECTION)
extern template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCapsuleModel<defaulttype::Vec3Types>;
extern template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCapsuleModel<defaulttype::Rigid3Types>;

#endif

}}}
#endif
