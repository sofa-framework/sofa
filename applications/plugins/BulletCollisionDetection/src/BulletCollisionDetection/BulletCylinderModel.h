#ifndef BULLET_CYLINDER_MODEL
#define BULLET_CYLINDER_MODEL

#include <sofa/component/collision/geometry/CylinderModel.h>
#include "BulletCollisionModel.h"
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/core/CollisionModel.h>
#include <BulletCollisionDetection/config.h>
#include <stack>

//WARNING : if you want to take account of intersections involving BulletCylinderModel,
//uncomment code in BulletCollisionDetection.h beginning at line 173

namespace sofa
{

namespace component
{

namespace collision
{

template<class TDataTypes>
class TBulletCylinderModel : public sofa::component::collision::geometry::CylinderCollisionModel<TDataTypes>,public BulletCollisionModel
{
public:
    //SOFA_CLASS2(SOFA_TEMPLATE(TBulletCylinderModel, TDataTypes),SOFA_TEMPLATE(sofa::component::collision::TriangleCollisionModel, TDataTypes),BulletCollisionModel);
    //SOFA_CLASS(SOFA_TEMPLATE(TBulletCylinderModel, TDataTypes),SOFA_TEMPLATE(TCylinderModel, TDataTypes));
	SOFA_CLASS(SOFA_TEMPLATE(TBulletCylinderModel, TDataTypes),SOFA_TEMPLATE(sofa::component::collision::geometry::CylinderCollisionModel, TDataTypes));

    using Inherit = sofa::component::collision::geometry::CylinderCollisionModel<TDataTypes>;
	
    //typedef typename GCylinderCollisionModel<sofa::defaulttype::Rigid3Types>::DataTypes DataTypes;
    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::Real Real;
	typedef typename type::vector<typename DataTypes::Vec3> VecAxisCoord;

    //typedef TBtTriangle<DataTypes> Element;

    sofa::core::objectmodel::Data<SReal> margin; ///< Margin used for collision detection within bullet

    virtual void initBullet();
    virtual void updateBullet();

    inline virtual void computeBoundingTree(int/* maxDepth*/){
        _bt_cshape->recalculateLocalAabb();
    }

    inline virtual ~TBulletCylinderModel();

    virtual void init();

    virtual void reinit();

    void handleEvent(sofa::core::objectmodel::Event * ev);

 	inline virtual void setMargin(SReal m){ *margin.beginEdit() = m; margin.endEdit(); }

protected:
    TBulletCylinderModel();
    TBulletCylinderModel(core::behavior::MechanicalState<DataTypes>* _mstate );

    std::stack<btCollisionShape*> _garbage;
    btCompoundShape * _bt_cshape;//or maybe something else ?

    void cleanGarbage();
};

typedef TBulletCylinderModel<defaulttype::RigidTypes> BulletCylinderModel;

#if  !defined(SOFA_BUILD_BULLETCOLLISIONDETECTION)
extern template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCylinderModel<defaulttype::Rigid3Types>;//je pense que les cylinders sont définis sur des rigides dans bullet

#endif

}}}
#endif
