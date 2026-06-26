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

#include <sofa/component/mapping/linear/BarycentricMapping.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/collision/response/mapper/BaseContactMapper.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>
#include <sofa/component/collision/geometry/LineCollisionModel.h>

namespace sofa::component::collision::response::mapper
{

/// Base class for all mappers using BarycentricMapping
template < class TCollisionModel, class TDataTypes >
class BarycentricContactMapper : public BaseContactMapper<TDataTypes>
{
public:
    typedef TDataTypes DataTypes;

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef typename MCollisionModel::Topology InTopology;
    typedef core::behavior::MechanicalState< InDataTypes> InMechanicalState;
    typedef core::behavior::MechanicalState<  typename BarycentricContactMapper::DataTypes> MMechanicalState;
    typedef component::statecontainer::MechanicalObject<typename BarycentricContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::linear::BarycentricMapping< InDataTypes, typename BarycentricContactMapper::DataTypes > MMapping;
    typedef mapping::linear::TopologyBarycentricMapper<InDataTypes, typename BarycentricContactMapper::DataTypes> MMapper;
    MCollisionModel* model;
    typename MMapping::SPtr mapping;
    typename MMapper::SPtr mapper;

    BarycentricContactMapper()
        : model(nullptr), mapping(nullptr), mapper(nullptr)
    {
    }

    void setCollisionModel(MCollisionModel* model)
    {
        this->model = model;
    }

    void cleanup() override;

    MMechanicalState* createMapping(const char* name="contactPoints") override;

    void resize(sofa::Size size) override
    {
        if (mapping != nullptr)
        {
            mapper->clear();
            mapping->getMechTo()[0]->resize(size);
        }
    }

    void update() override
    {
        if (mapping != nullptr)
        {
            core::BaseMapping* map = mapping.get();
            map->apply(core::mechanicalparams::defaultInstance(), core::vec_id::write_access::position, core::vec_id::read_access::position);
            map->applyJ(core::mechanicalparams::defaultInstance(), core::vec_id::write_access::velocity, core::vec_id::read_access::velocity);
        }
    }

    void updateXfree() override
    {
        if (mapping != nullptr)
        {
            core::BaseMapping* map = mapping.get();
            map->apply(core::mechanicalparams::defaultInstance(), core::vec_id::write_access::freePosition, core::vec_id::read_access::freePosition);
            map->applyJ(core::mechanicalparams::defaultInstance(), core::vec_id::write_access::freeVelocity, core::vec_id::read_access::freeVelocity);
        }
    }

    void updateX0()
    {
        if (mapping != nullptr)
        {
            core::BaseMapping* map = mapping.get();
            map->apply(core::mechanicalparams::defaultInstance(), core::vec_id::write_access::restPosition, core::vec_id::read_access::restPosition);
        }
    }
};

/// Mapper for LineModel
template<class DataTypes>
class ContactMapper<collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, DataTypes> : public BarycentricContactMapper<collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    sofa::Index addPoint(const Coord& P, sofa::Index index, Real&)
    {
        return this->mapper->createPointInLine(P, this->model->getElemEdgeIndex(index), &this->model->getMechanicalState()->read(core::vec_id::read_access::position)->getValue());
    }
    sofa::Index addPointB(const Coord& /*P*/, sofa::Index index, Real& /*r*/, const type::Vec3& baryP)
    {
        return this->mapper->addPointInLine(this->model->getElemEdgeIndex(index), baryP.ptr());
    }

    inline sofa::Index addPointB(const Coord& P, sofa::Index index, Real& r ){return addPoint(P,index,r);}
};

/// Mapper for TriangleModel
template<class DataTypes>
class ContactMapper<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, DataTypes> : public BarycentricContactMapper<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    sofa::Index addPoint(const Coord& P, sofa::Index index, Real&)
    {
        auto nbt = this->model->getCollisionTopology()->getNbTriangles();
        if (index < nbt)
            return this->mapper->createPointInTriangle(P, index, &this->model->getMechanicalState()->read(core::vec_id::read_access::position)->getValue());
        else
        {
            sofa::Index qindex = (index - nbt)/2;
            auto nbq = this->model->getCollisionTopology()->getNbQuads();
            if (qindex < nbq)
                return this->mapper->createPointInQuad(P, qindex, &this->model->getMechanicalState()->read(core::vec_id::read_access::position)->getValue());
            else
            {
                msg_error("ContactMapper<TriangleCollisionModel<sofa::defaulttype::Vec3Types>>") << "Invalid contact element index "<<index<<" on a topology with "<<nbt<<" triangles and "<<nbq<<" quads."<<msgendl
                                                              << "model="<<this->model->getName()<<" size="<<this->model->getSize() ;
                return sofa::InvalidID;
            }
        }
    }
    sofa::Index addPointB(const Coord& P, sofa::Index index, Real& /*r*/, const type::Vec3& baryP)
    {

        auto nbt = this->model->getCollisionTopology()->getNbTriangles();
        if (index < nbt)
            return this->mapper->addPointInTriangle(index, baryP.ptr());
        else
        {
            // TODO: barycentric coordinates usage for quads
            sofa::Index qindex = (index - nbt)/2;
            auto nbq = this->model->getCollisionTopology()->getNbQuads();
            if (qindex < nbq)
                return this->mapper->createPointInQuad(P, qindex, &this->model->getMechanicalState()->read(core::vec_id::read_access::position)->getValue());
            else
            {
                msg_error("ContactMapper<TriangleCollisionModel<sofa::defaulttype::Vec3Types>>") << "Invalid contact element index "<<index<<" on a topology with "<<nbt<<" triangles and "<<nbq<<" quads."<<msgendl
                            << "model="<<this->model->getName()<<" size="<<this->model->getSize() ;
                return sofa::InvalidID;
            }
        }
    }

    inline sofa::Index addPointB(const Coord& P, sofa::Index index, Real& r ){return addPoint(P,index,r);}

};

#if !defined(SOFA_COMPONENT_COLLISION_BARYCENTRICCONTACTMAPPER_CPP)
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API ContactMapper<collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, sofa::defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API ContactMapper<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, sofa::defaulttype::Vec3Types>;

#  ifdef _MSC_VER
// Manual declaration of non-specialized members, to avoid warnings from MSVC.
extern template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API void BarycentricContactMapper<collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, defaulttype::Vec3Types>::cleanup();
extern template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API core::behavior::MechanicalState<defaulttype::Vec3Types>* BarycentricContactMapper<collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, defaulttype::Vec3Types>::createMapping(const char*);
extern template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API void BarycentricContactMapper<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, defaulttype::Vec3Types>::cleanup();
extern template SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API core::behavior::MechanicalState<defaulttype::Vec3Types>* BarycentricContactMapper<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, defaulttype::Vec3Types>::createMapping(const char*);
#  endif // _MSC_VER
#endif

} //namespace sofa::component::collision::response::mapper
