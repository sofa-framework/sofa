/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/collision/geometry/config.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::collision::geometry
{

template<class DataTypes>
class SphereCollisionModel;

template<class TDataTypes>
class TSphere : public core::TCollisionElementIterator< SphereCollisionModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename TDataTypes::CPos Coord;
    static_assert(std::is_same_v<TSphere<DataTypes>::Coord, Coord>, "Data mismatch");

    typedef SphereCollisionModel<DataTypes> ParentModel;

    using Index = sofa::Index;

    TSphere(ParentModel* model, Index index);

    explicit TSphere(const core::CollisionElementIterator& i);

    const Coord& center() const;
    const typename TDataTypes::Coord & rigidCenter() const;
    const Coord& p() const;
    const Coord& pFree() const;
    const Coord& v() const;

    /// Return true if the element stores a free position vector
    bool hasFreePosition() const;

    Real r() const;

    sofa::type::Vec3 getContactPointByNormal( const sofa::type::Vec3& contactNormal )
    {
        return center() - contactNormal * r();
    }

    sofa::type::Vec3 getContactPointWithSurfacePoint( const sofa::type::Vec3& surfacePoint )
    {
        return surfacePoint;
    }
};

// Specializations
template <> SOFA_COMPONENT_COLLISION_GEOMETRY_API
sofa::type::Vec3 TSphere<defaulttype::Vec3Types >::getContactPointByNormal( const sofa::type::Vec3& /*contactNormal*/ );
template <> SOFA_COMPONENT_COLLISION_GEOMETRY_API
sofa::type::Vec3 TSphere<defaulttype::Vec3Types >::getContactPointWithSurfacePoint( const sofa::type::Vec3& );


template< class TDataTypes>
class SphereCollisionModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SphereCollisionModel, TDataTypes), core::CollisionModel);

    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;

    typedef typename DataTypes::CPos Coord;
    //typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecReal VecReal;
    typedef TSphere<DataTypes> Element;
    friend class TSphere<DataTypes>;
protected:
    SphereCollisionModel();

    SphereCollisionModel(core::behavior::MechanicalState<TDataTypes>* _mstate );

    void drawCollisionModel(const core::visual::VisualParams* vparams) override;

public:
    void init() override;

    // -- CollisionModel interface

    void resize(sofa::Size size) override;

    void computeBoundingTree(int maxDepth=0) override;

    void computeContinuousBoundingTree(SReal dt, int maxDepth=0) override;

    void draw(const core::visual::VisualParams*, sofa::Index index) override;

    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return mstate; }

    const VecReal& getR() const { return this->d_radius.getValue(); }

    Real getRadius(const sofa::Index i) const;

    const Coord & velocity(sofa::Index index)const;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<TDataTypes>*>(context->getMechanicalState()) == nullptr && context->getMechanicalState() != nullptr)
        {
            arg->logError(std::string("No mechanical state with the datatype '") + DataTypes::Name() +
                          "' found in the context node.");
            return false;
        }

        return BaseObject::canCreate(obj, context, arg);
    }

    template<class T>
    static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj;

        if( context)
        {
            auto* _mstate = dynamic_cast<core::behavior::MechanicalState<TDataTypes>*>(context->getMechanicalState());
            if (_mstate)
                obj = sofa::core::objectmodel::New<T>(_mstate);
            else
                obj = sofa::core::objectmodel::New<T>();

            context->addObject(obj);
        }
        else
        {
            obj = sofa::core::objectmodel::New<T>();
        }

        if (arg && obj) obj->parse(arg);

        return obj;
    }

    Data< VecReal > d_radius; ///< Radius of each sphere
    Data< SReal > d_defaultRadius; ///< Default radius
    Data< bool > d_showImpostors; ///< Draw spheres as impostors instead of "real" spheres


    void computeBBox(const core::ExecParams* params, bool onlyVisible=false) override;

protected:
    core::behavior::MechanicalState<DataTypes>* mstate;
    SingleLink<SphereCollisionModel<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
};

template<class DataTypes>
inline TSphere<DataTypes>::TSphere(ParentModel* model, Index index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TSphere<DataTypes>::TSphere(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
}

template<class DataTypes>
inline const typename TSphere<DataTypes>::Coord& TSphere<DataTypes>::center() const { return DataTypes::getCPos(this->model->mstate->read(core::vec_id::read_access::position)->getValue()[this->index]); }

template<class DataTypes>
inline const typename DataTypes::Coord & TSphere<DataTypes>::rigidCenter() const { return this->model->mstate->read(core::vec_id::read_access::position)->getValue()[this->index];}

template<class DataTypes>
inline const typename TSphere<DataTypes>::Coord& TSphere<DataTypes>::p() const { return DataTypes::getCPos(this->model->mstate->read(core::vec_id::read_access::position)->getValue()[this->index]);}

template<class DataTypes>
inline const typename TSphere<DataTypes>::Coord& TSphere<DataTypes>::pFree() const { return (*this->model->mstate->read(core::vec_id::read_access::freePosition)).getValue()[this->index]; }

template<class DataTypes>
inline const typename SphereCollisionModel<DataTypes>::Coord& SphereCollisionModel<DataTypes>::velocity(sofa::Index index) const { return DataTypes::getDPos(mstate->read(core::vec_id::read_access::velocity)->getValue()[index]);}

template<class DataTypes>
inline const typename TSphere<DataTypes>::Coord& TSphere<DataTypes>::v() const { return DataTypes::getDPos(this->model->mstate->read(core::vec_id::read_access::velocity)->getValue()[this->index]); }

template<class DataTypes>
inline typename DataTypes::Real TSphere<DataTypes>::r() const { return (Real) this->model->getRadius((unsigned)this->index); }

template<class DataTypes>
inline bool TSphere<DataTypes>::hasFreePosition() const { return this->model->mstate->read(core::vec_id::read_access::freePosition)->isSet(); }

using Sphere = TSphere<sofa::defaulttype::Vec3Types>;
using RigidSphere = TSphere<sofa::defaulttype::Rigid3Types>;
using RigidSphereModel = SphereCollisionModel<sofa::defaulttype::Rigid3Types>;

#if !defined(SOFA_COMPONENT_COLLISION_SPHERECOLLISIONMODEL_CPP)
extern template class SOFA_COMPONENT_COLLISION_GEOMETRY_API TSphere<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_COLLISION_GEOMETRY_API SphereCollisionModel<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_COLLISION_GEOMETRY_API SphereCollisionModel<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::collision::geometry
