/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_SPHEREMODEL_H
#define SOFA_COMPONENT_COLLISION_SPHEREMODEL_H
#include "config.h"

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/accessor.h>
//#include <SofaMeshCollision/RigidContactMapper.h>

namespace sofa
{

namespace component
{

namespace collision
{


template<class DataTypes>
class TSphereModel;

//template <class TDataTypes>
//class TSphere;

template<class TDataTypes>
class TSphere : public core::TCollisionElementIterator< TSphereModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename TDataTypes::CPos Coord;

    typedef TSphereModel<DataTypes> ParentModel;

    TSphere(ParentModel* model, int index);

    explicit TSphere(const core::CollisionElementIterator& i);

    const Coord& center() const;
    const typename TDataTypes::Coord & rigidCenter() const;
    const Coord& p() const;
    const Coord& pFree() const;
    const Coord& v() const;

    /// Return true if the element stores a free position vector
    bool hasFreePosition() const;

    Real r() const;

    sofa::defaulttype::Vector3 getContactPointByNormal( const sofa::defaulttype::Vector3& contactNormal )
    {
        return center() - contactNormal * r();
    }

    sofa::defaulttype::Vector3 getContactPointWithSurfacePoint( const sofa::defaulttype::Vector3& surfacePoint )
    {
        return surfacePoint;
    }

};

// Specializations
#ifndef SOFA_FLOAT
template <> SOFA_BASE_COLLISION_API
sofa::defaulttype::Vector3 TSphere<defaulttype::Vec3dTypes >::getContactPointByNormal( const sofa::defaulttype::Vector3& /*contactNormal*/ );
template <> SOFA_BASE_COLLISION_API
sofa::defaulttype::Vector3 TSphere<defaulttype::Vec3dTypes >::getContactPointWithSurfacePoint( const sofa::defaulttype::Vector3& );
#endif
#ifndef SOFA_DOUBLE
template <> SOFA_BASE_COLLISION_API
sofa::defaulttype::Vector3 TSphere<defaulttype::Vec3fTypes >::getContactPointByNormal( const sofa::defaulttype::Vector3& /*contactNormal*/ );
template <> SOFA_BASE_COLLISION_API
sofa::defaulttype::Vector3 TSphere<defaulttype::Vec3fTypes >::getContactPointWithSurfacePoint( const sofa::defaulttype::Vector3& );
#endif

template< class TDataTypes>
class TSphereModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TSphereModel, TDataTypes), core::CollisionModel);

    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;

    typedef typename DataTypes::CPos Coord;
    //typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecReal VecReal;
    typedef TSphere<DataTypes> Element;
    friend class TSphere<DataTypes>;
protected:
    TSphereModel();

    TSphereModel(core::behavior::MechanicalState<TDataTypes>* _mstate );
public:
    virtual void init() override;

    // -- CollisionModel interface

    virtual void resize(int size) override;

    virtual void computeBoundingTree(int maxDepth=0) override;

    virtual void computeContinuousBoundingTree(SReal dt, int maxDepth=0) override;

    void draw(const core::visual::VisualParams*,int index) override;

    void draw(const core::visual::VisualParams* vparams) override;


    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return mstate; }

    const VecReal& getR() const { return this->radius.getValue(); }

    Real getRadius(const int i) const;

    const Coord & velocity(int index)const;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<TDataTypes>*>(context->getMechanicalState()) == NULL && context->getMechanicalState() != NULL)
            return false;

        return BaseObject::canCreate(obj, context, arg);
    }

    template<class T>
    static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj;
        core::behavior::MechanicalState<TDataTypes>* _mstate = NULL;

        if( context)
        {
            _mstate = dynamic_cast<core::behavior::MechanicalState<TDataTypes>*>(context->getMechanicalState());
            if (_mstate)
                obj = sofa::core::objectmodel::New<T>(_mstate);
            else
                obj = sofa::core::objectmodel::New<T>();

            context->addObject(obj);
        }

        if (arg) obj->parse(arg);

        return obj;
    }


    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const TSphereModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    //TODO(dmarchal) guideline de sofa.
    Data< VecReal > radius; ///< Radius of each sphere
    Data< SReal > defaultRadius; ///< Default Radius
    Data< bool > d_showImpostors; ///< Draw spheres as impostors instead of "real" spheres


    virtual void computeBBox(const core::ExecParams* params, bool onlyVisible=false) override;

protected:
    core::behavior::MechanicalState<DataTypes>* mstate;
};



template<class DataTypes>
inline TSphere<DataTypes>::TSphere(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TSphere<DataTypes>::TSphere(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
}

template<class DataTypes>
inline const typename TSphere<DataTypes>::Coord& TSphere<DataTypes>::center() const { return DataTypes::getCPos(this->model->mstate->read(core::ConstVecCoordId::position())->getValue()[this->index]); }

template<class DataTypes>
inline const typename DataTypes::Coord & TSphere<DataTypes>::rigidCenter() const { return this->model->mstate->read(core::ConstVecCoordId::position())->getValue()[this->index];}

template<class DataTypes>
inline const typename TSphere<DataTypes>::Coord& TSphere<DataTypes>::p() const { return DataTypes::getCPos(this->model->mstate->read(core::ConstVecCoordId::position())->getValue()[this->index]);}

template<class DataTypes>
inline const typename TSphere<DataTypes>::Coord& TSphere<DataTypes>::pFree() const { return (*this->model->mstate->read(core::ConstVecCoordId::freePosition())).getValue()[this->index]; }

template<class DataTypes>
inline const typename TSphereModel<DataTypes>::Coord& TSphereModel<DataTypes>::velocity(int index) const { return DataTypes::getDPos(mstate->read(core::ConstVecDerivId::velocity())->getValue()[index]);}

template<class DataTypes>
inline const typename TSphere<DataTypes>::Coord& TSphere<DataTypes>::v() const { return DataTypes::getDPos(this->model->mstate->read(core::ConstVecDerivId::velocity())->getValue()[this->index]); }

template<class DataTypes>
inline typename DataTypes::Real TSphere<DataTypes>::r() const { return (Real) this->model->getRadius((unsigned)this->index); }

template<class DataTypes>
inline bool TSphere<DataTypes>::hasFreePosition() const { return this->model->mstate->read(core::ConstVecCoordId::freePosition())->isSet(); }


typedef TSphereModel<sofa::defaulttype::Vec3Types> SphereModel;
typedef TSphere<sofa::defaulttype::Vec3Types> Sphere;

typedef TSphereModel<sofa::defaulttype::Rigid3Types> RigidSphereModel;
typedef TSphere<sofa::defaulttype::Rigid3Types> RigidSphere;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_SPHEREMODEL_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_COLLISION_API TSphere<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_COLLISION_API TSphereModel<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_COLLISION_API TSphereModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_COLLISION_API TSphere<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_COLLISION_API TSphereModel<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_COLLISION_API TSphereModel<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif
