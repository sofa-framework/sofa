#ifndef BULLET_CONVEX_HULL_MODEL
#define BULLET_CONVEX_HULL_MODEL

#include <sofa/core/CollisionModel.h>
#include "BulletCollisionModel.h"
#include <sofa/simulation/CollisionBeginEvent.h>
#include <BulletCollisionDetection/config.h>
#include <sofa/core/visual/VisualParams.h>
#include <stack>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include <HACD/hacdHACD.h>
#pragma GCC diagnostic pop



namespace sofa
{

namespace component
{

namespace collision
{

template<class TDataTypes>
class TBulletConvexHullModel;

template<class TDataTypes>
class TBulletConvexHull : public core::TCollisionElementIterator< TBulletConvexHullModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Coord::Pos Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Quat Quaternion;

    typedef TBulletConvexHullModel<DataTypes> ParentModel;

    TBulletConvexHull(ParentModel* model, int index);

    explicit TBulletConvexHull(const core::CollisionElementIterator& i);
};

template<class DataTypes>
inline TBulletConvexHull<DataTypes>::TBulletConvexHull(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TBulletConvexHull<DataTypes>::TBulletConvexHull(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
}


template<class TDataTypes>
class TBulletConvexHullModel : public sofa::core::CollisionModel ,public BulletCollisionModel
{
public:
    //SOFA_CLASS2(SOFA_TEMPLATE(TBulletConvexHullModel, TDataTypes),SOFA_TEMPLATE(sofa::component::collision::TTriangleModel, TDataTypes),BulletCollisionModel);
    SOFA_CLASS(SOFA_TEMPLATE(TBulletConvexHullModel, TDataTypes),sofa::core::CollisionModel);

    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::Coord::Pos Coord;
    typedef helper::vector<Coord> VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::Quat Quaternion;
    typedef TBulletConvexHull<DataTypes> Element;

    sofa::core::objectmodel::Data<SReal> margin; ///< Margin used for collision detection within bullet

    virtual void init();

    virtual void reinit();

    // -- CollisionModel interface

    virtual void resize(int size);

    //void draw(const core::visual::VisualParams*,int index);

    void draw(const core::visual::VisualParams* vparams);    

    inline virtual void computeBoundingTree(int/* maxDepth*/){
        _bt_cshape.recalculateLocalAabb();
    }

    virtual bool canCollideWithElement(int index, CollisionModel* model2, int index2){
        if(this == model2)
            return false;

        return CollisionModel::canCollideWithElement(index,model2,index2);
    }

    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return _mstate; }
    const core::behavior::MechanicalState<DataTypes>* getMechanicalState() const { return _mstate; }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const TBulletConvexHullModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    // -- Bullet interface

    virtual void initBullet();
    virtual void updateBullet();

    inline virtual ~TBulletConvexHullModel(){
        while(!_garbage.empty()){
            delete _garbage.top();
            _garbage.pop();
        }
    }

    void handleEvent(sofa::core::objectmodel::Event * ev);

    inline virtual void setMargin(SReal margin){_bt_cshape.setMargin(margin);}

    inline const Coord & center()const{
        return DataTypes::getCPos(_mstate->read(core::ConstVecCoordId::position())->getValue()[0]);
    }

    inline const Quaternion & orientation()const{
        return _mstate->read(core::ConstVecCoordId::position())->getValue()[0].getOrientation();
    }

    Data<bool> computeConvexHullDecomposition; ///< compute convex hull decomposition using HACD
    Data<bool> drawConvexHullDecomposition; ///< draw convex hull decomposition using
    Data<VecCoord> CHPoints; ///< points defining the convex hull
    Data<bool> computeNormals; ///< set to false to disable computation of triangles normal
    Data<bool> positionDefined; ///< set to true if the collision model position is defined in the mechanical object
    Data<SReal> concavityThreeshold; ///< Threeshold used in the decomposition
protected:
    void draw_without_decomposition(const core::visual::VisualParams* vparams);
    void draw_decomposition(const core::visual::VisualParams* vparams);

    Coord _bary;
    btTransform _bt_trans;
    core::behavior::MechanicalState<DataTypes>* _mstate;

    TBulletConvexHullModel();
    TBulletConvexHullModel(core::behavior::MechanicalState<DataTypes>* _mstate );

    std::stack<btCollisionShape*> _garbage;
    btCompoundShape _bt_cshape;

    sofa::core::topology::BaseMeshTopology* bmsh;

    std::vector<std::vector< defaulttype::Vector3 > > _ch_deco_pts;//convex hull decomposition triangles, used only for drawing convex hulls
    std::vector<std::vector< sofa::defaulttype::Vec<3,int> > > _ch_deco_tri;
    std::vector<std::vector< defaulttype::Vector3 > > _ch_deco_norms;

    std::vector<defaulttype::Vec<4,float> > _ch_deco_colors;

};

typedef TBulletConvexHullModel<defaulttype::RigidTypes> BulletConvexHullModel;
typedef TBulletConvexHull<defaulttype::RigidTypes> BulletConvexHull;


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BULLETCOLLISIONDETECTION)
#ifndef SOFA_FLOAT
extern template class SOFA_BULLETCOLLISIONDETECTION_API TBulletConvexHullModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BULLETCOLLISIONDETECTION_API TBulletConvexHullModel<defaulttype::Rigid3fTypes>;
#endif
#endif

}}}
#endif
