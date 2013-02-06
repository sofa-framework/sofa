/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_H
#define SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/Factory.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/mapping/SubsetMapping.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/collision/BaseContactMapper.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/TetrahedronModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/RigidSphereModel.h>
#include <sofa/component/collision/OBBModel.h>
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/core/VecId.h>
#include <iostream>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;


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
    typedef component::container::MechanicalObject<typename RigidContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::RigidMapping< InDataTypes, typename RigidContactMapper::DataTypes > MMapping;

    MCollisionModel* model;
    simulation::Node::SPtr child;
    typename MMapping::SPtr mapping;
    typename MMechanicalState::SPtr outmodel;
    int nbp;

protected:
    RigidContactMapper()
        : model(NULL), child(NULL), mapping(NULL), outmodel(NULL), nbp(0)
    {
    }

public:

    void setCollisionModel(MCollisionModel* model)
    {
        this->model = model;
    }

    void cleanup();

    MMechanicalState* createMapping(const char* name="contactPoints");

    void resize(int size)
    {
        if (mapping!=NULL)
            mapping->clear(size);
        if (outmodel!=NULL)
            outmodel->resize(size);
        nbp = 0;
    }

    int addPoint(const Coord& P, int index, Real&)
    {
        int i = nbp++;
        if ((int)outmodel->getX()->size() <= i)
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
        if (mapping!=NULL)
        {
            core::BaseMapping* map = mapping.get();
            map->apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::position(), core::ConstVecCoordId::position());
            map->applyJ(core::MechanicalParams::defaultInstance(), core::VecDerivId::velocity(), core::ConstVecDerivId::velocity());
        }
    }

    void updateXfree()
    {
        if (mapping!=NULL)
        {
            core::BaseMapping* map = mapping.get();
            map->apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::freePosition(), core::ConstVecCoordId::freePosition());
        }
    }
};


template <class TVec3Types>
class ContactMapper<RigidSphereModel,TVec3Types > : public RigidContactMapper<RigidSphereModel, TVec3Types >{
    public:
        int addPoint(const typename TVec3Types::Coord & P, int index,typename TVec3Types::Real & r)
        {
            RigidSphere e(this->model, index);
            const typename TVec3Types::Coord & cP = P - e.center();
            const RigidSphereModel::Quaternion & ori = e.orientation();

            r = e.r();

            return RigidContactMapper<RigidSphereModel,TVec3Types >::addPoint(ori.inverseRotate(cP),index,r);
        }
};


template <class TVec3Types>
class ContactMapper<OBBModel,TVec3Types > : public RigidContactMapper<OBBModel, TVec3Types >{
    public:
        int addPoint(const typename TVec3Types::Coord & P, int index,typename TVec3Types::Real & r)
        {
            const typename TVec3Types::Coord & cP = P - this->model->center(index);
            const RigidSphereModel::Quaternion & ori = this->model->orientation(index);

            return RigidContactMapper<OBBModel,TVec3Types >::addPoint(ori.inverseRotate(cP),index,r);
        }
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_MESH_COLLISION)
extern template class SOFA_MESH_COLLISION_API ContactMapper<RigidSphereModel,Vec3Types>;
extern template class SOFA_MESH_COLLISION_API ContactMapper<OBBModel,Vec3Types>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_H */
