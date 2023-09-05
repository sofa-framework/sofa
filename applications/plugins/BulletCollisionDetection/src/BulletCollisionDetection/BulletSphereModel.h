#ifndef BULLET_SPHERE_MODEL
#define BULLET_SPHERE_MODEL

#include <sofa/component/collision/geometry/SphereModel.h>
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

template<class TDataTypes>
class TBulletSphereModel : public sofa::component::collision::geometry::SphereCollisionModel<TDataTypes>,public BulletCollisionModel
{
public:
    //SOFA_CLASS2(SOFA_TEMPLATE(TBulletSphereModel, TDataTypes),SOFA_TEMPLATE(sofa::component::collision::TriangleCollisionModel, TDataTypes),BulletCollisionModel);
    SOFA_CLASS(SOFA_TEMPLATE(TBulletSphereModel, TDataTypes),SOFA_TEMPLATE(sofa::component::collision::geometry::SphereCollisionModel, TDataTypes));

    using Inherit = sofa::component::collision::geometry::SphereCollisionModel<TDataTypes>;

    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecReal VecReal;
    //typedef TBtTriangle<DataTypes> Element;

    sofa::core::objectmodel::Data<SReal> margin; ///< Margin used for collision detection within bullet

    virtual void initBullet();
    virtual void updateBullet();

    inline virtual void computeBoundingTree(int/* maxDepth*/){
        _bt_cshape->recalculateLocalAabb();
    }

    inline virtual ~TBulletSphereModel();

    virtual void init();

    virtual void reinit();

    void handleEvent(sofa::core::objectmodel::Event * ev);

	inline virtual void setMargin(SReal m){ *margin.beginEdit() = m; margin.endEdit(); }

protected:
    TBulletSphereModel();
    TBulletSphereModel(core::behavior::MechanicalState<DataTypes>* _mstate );

    std::stack<btCollisionShape*> _garbage;
    btCompoundShape * _bt_cshape;

    void cleanGarbage();
};

typedef TBulletSphereModel<defaulttype::Vec3Types> BulletSphereModel;

#if  !defined(SOFA_BUILD_BULLETCOLLISIONDETECTION)
extern template class SOFA_BULLETCOLLISIONDETECTION_API TBulletSphereModel<defaulttype::Vec3Types>;

#endif

}}}
#endif
