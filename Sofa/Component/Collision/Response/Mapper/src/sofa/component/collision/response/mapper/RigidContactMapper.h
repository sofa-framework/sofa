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
#include <sofa/component/collision/response/mapper/config.h>

#include <sofa/helper/Factory.h>
#include <sofa/component/mapping/nonlinear/RigidMapping.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/collision/response/mapper/BaseContactMapper.h>
#include <sofa/component/collision/geometry/SphereModel.h>
#include <sofa/component/collision/geometry/CylinderModel.h>
#include <sofa/simulation/fwd.h>

namespace sofa::component::collision::response::mapper
{

/// Base class for all mappers using RigidMapping
template < class TCollisionModel, class DataTypes >
class RigidContactMapper : public BaseContactMapper<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef core::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::behavior::MechanicalState<typename RigidContactMapper::DataTypes> MMechanicalState;
    typedef component::statecontainer::MechanicalObject<typename RigidContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::nonlinear::RigidMapping< InDataTypes, typename RigidContactMapper::DataTypes > MMapping;

    using Index = sofa::Index;

    MCollisionModel* model;
    simulation::NodeSPtr child;
    typename MMapping::SPtr mapping;
    typename MMechanicalState::SPtr outmodel;
    Size nbp;

protected:
    RigidContactMapper();

public:

    void setCollisionModel(MCollisionModel* model)
    {
        this->model = model;
    }

    void cleanup();

    MMechanicalState* createMapping(const char* name="contactPoints");

    void resize(Size size)
    {
        if (mapping != nullptr)
            mapping->clear(size);
        if (outmodel != nullptr)
            outmodel->resize(size);
        nbp = 0;
    }

    Index addPoint(const Coord& P, Index index, Real&)
    {
        Index i = nbp++;
        if (outmodel->getSize() <= i)
            outmodel->resize(i+1);
        if (mapping)
        {
            i = mapping->addPoint(P,index);
        }
        else
        {
            helper::WriteAccessor<Data<VecCoord> > xData = *outmodel->write(core::VecCoordId::position());
            xData.wref()[i] = P;
        }
        return i;
    }

    void update()
    {
        if (mapping != nullptr)
        {
            core::BaseMapping* map = mapping.get();
            map->apply(core::mechanicalparams::defaultInstance(), core::VecCoordId::position(), core::ConstVecCoordId::position());
            map->applyJ(core::mechanicalparams::defaultInstance(), core::VecDerivId::velocity(), core::ConstVecDerivId::velocity());
        }
    }

    void updateXfree()
    {
        if (mapping != nullptr)
        {
            core::BaseMapping* map = mapping.get();
            map->apply(core::mechanicalparams::defaultInstance(), core::VecCoordId::freePosition(), core::ConstVecCoordId::freePosition());
            map->applyJ(core::mechanicalparams::defaultInstance(), core::VecDerivId::freeVelocity(), core::ConstVecDerivId::freeVelocity());
        }
    }
};


template <class TVec3Types>
class ContactMapper<collision::geometry::RigidSphereModel,TVec3Types > : public RigidContactMapper<collision::geometry::RigidSphereModel, TVec3Types >{
    public:
        sofa::Index addPoint(const typename TVec3Types::Coord & P, sofa::Index index,typename TVec3Types::Real & r)
        {
            const collision::geometry::RigidSphere e(this->model, index);
            const typename collision::geometry::SphereCollisionModel<sofa::defaulttype::Rigid3Types>::DataTypes::Coord & rCenter = e.rigidCenter();
            const typename TVec3Types::Coord & cP = P - rCenter.getCenter();
            const type::Quat<SReal> & ori = rCenter.getOrientation();

            //r = e.r();

            return RigidContactMapper<collision::geometry::RigidSphereModel,TVec3Types >::addPoint(ori.inverseRotate(cP),index,r);
        }
};

template <class TVec3Types>
class ContactMapper<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>,TVec3Types > : public RigidContactMapper<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, TVec3Types >{
    public:
        sofa::Index addPoint(const typename TVec3Types::Coord & P, sofa::Index index,typename TVec3Types::Real & r)
        {
            const typename TVec3Types::Coord & cP = P - this->model->center(index);
            const type::Quat<SReal> & ori = this->model->orientation(index);

            return RigidContactMapper<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>,TVec3Types >::addPoint(ori.inverseRotate(cP),index,r);
        }
};

#if !defined(SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_CPP)
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API ContactMapper<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API ContactMapper<collision::geometry::RigidSphereModel, defaulttype::Vec3Types>;

// Manual declaration of non-specialized members, to avoid warnings from MSVC.
extern template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API void RigidContactMapper<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>::cleanup();
extern template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API core::behavior::MechanicalState<defaulttype::Vec3Types>* RigidContactMapper<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, defaulttype::Vec3Types>::createMapping(const char*);
extern template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API void RigidContactMapper<collision::geometry::RigidSphereModel, defaulttype::Vec3Types>::cleanup();
extern template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API core::behavior::MechanicalState<defaulttype::Vec3Types>* RigidContactMapper<collision::geometry::RigidSphereModel, defaulttype::Vec3Types>::createMapping(const char*);
#endif

} //namespace sofa::component::collision::response::mapper
