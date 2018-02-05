/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICCONTACTMAPPER_H
#define SOFA_COMPONENT_COLLISION_BARYCENTRICCONTACTMAPPER_H
#include "config.h"

#include <sofa/helper/system/config.h>
#include <sofa/helper/Factory.h>
#include <SofaBaseMechanics/BarycentricMapping.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <SofaRigid/RigidMapping.h>
#include <SofaBaseMechanics/SubsetMapping.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <SofaBaseCollision/BaseContactMapper.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <iostream>


namespace sofa
{

namespace component
{

namespace collision
{

/// Base class for all mappers using BarycentricMapping
template < class TCollisionModel, class DataTypes >
class BarycentricContactMapper : public BaseContactMapper<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef typename MCollisionModel::Topology InTopology;
    typedef core::behavior::MechanicalState< InDataTypes> InMechanicalState;
    typedef core::behavior::MechanicalState<  typename BarycentricContactMapper::DataTypes> MMechanicalState;
    typedef component::container::MechanicalObject<typename BarycentricContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::BarycentricMapping< InDataTypes, typename BarycentricContactMapper::DataTypes > MMapping;
    typedef mapping::TopologyBarycentricMapper<InDataTypes, typename BarycentricContactMapper::DataTypes> MMapper;
    MCollisionModel* model;
    typename MMapping::SPtr mapping;
    typename MMapper::SPtr mapper;

    BarycentricContactMapper()
        : model(NULL), mapping(NULL), mapper(NULL)
    {
    }

    void setCollisionModel(MCollisionModel* model)
    {
        this->model = model;
    }

    void cleanup();

    MMechanicalState* createMapping(const char* name="contactPoints");

    void resize(int size)
    {
        if (mapping!=NULL)
        {
            mapper->clear();
            mapping->getMechTo()[0]->resize(size);
        }
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
            map->applyJ(core::MechanicalParams::defaultInstance(), core::VecDerivId::freeVelocity(), core::ConstVecDerivId::freeVelocity());
        }
    }

    void updateX0()
    {
        if (mapping!=NULL)
        {
            core::BaseMapping* map = mapping.get();
            map->apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::restPosition(), core::ConstVecCoordId::restPosition());
        }
    }
};

/// Mapper for LineModel
template<class DataTypes>
class ContactMapper<LineModel, DataTypes> : public BarycentricContactMapper<LineModel, DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    int addPoint(const Coord& P, int index, Real&)
    {
        return this->mapper->createPointInLine(P, this->model->getElemEdgeIndex(index), &this->model->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue());
    }
    int addPointB(const Coord& /*P*/, int index, Real& /*r*/, const defaulttype::Vector3& baryP)
    {
        return this->mapper->addPointInLine(this->model->getElemEdgeIndex(index), baryP.ptr());
    }

    inline int addPointB(const Coord& P, int index, Real& r ){return addPoint(P,index,r);}
};

/// Mapper for TriangleModel
template<class DataTypes>
class ContactMapper<TriangleModel, DataTypes> : public BarycentricContactMapper<TriangleModel, DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    int addPoint(const Coord& P, int index, Real&)
    {
        int nbt = this->model->getMeshTopology()->getNbTriangles();
        if (index < nbt)
            return this->mapper->createPointInTriangle(P, index, &this->model->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue());
        else
        {
            int qindex = (index - nbt)/2;
            int nbq = this->model->getMeshTopology()->getNbQuads();
            if (qindex < nbq)
                return this->mapper->createPointInQuad(P, qindex, &this->model->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue());
            else
            {
                msg_error("ContactMapper<TriangleModel>") << "Invalid contact element index "<<index<<" on a topology with "<<nbt<<" triangles and "<<nbq<<" quads."<<msgendl
                                                              << "model="<<this->model->getName()<<" size="<<this->model->getSize() ;
                return -1;
            }
        }
    }
    int addPointB(const Coord& P, int index, Real& /*r*/, const defaulttype::Vector3& baryP)
    {

        int nbt = this->model->getMeshTopology()->getNbTriangles();
        if (index < nbt)
            return this->mapper->addPointInTriangle(index, baryP.ptr());
        else
        {
            // TODO: barycentric coordinates usage for quads
            int qindex = (index - nbt)/2;
            int nbq = this->model->getMeshTopology()->getNbQuads();
            if (qindex < nbq)
                return this->mapper->createPointInQuad(P, qindex, &this->model->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue());
            else
            {
                msg_error("ContactMapper<TriangleModel>") << "Invalid contact element index "<<index<<" on a topology with "<<nbt<<" triangles and "<<nbq<<" quads."<<msgendl
                            << "model="<<this->model->getName()<<" size="<<this->model->getSize() ;
                return -1;
            }
        }
    }

    inline int addPointB(const Coord& P, int index, Real& r ){return addPoint(P,index,r);}

};


template <class DataTypes>
class ContactMapper<CapsuleModel, DataTypes> : public BarycentricContactMapper<CapsuleModel, DataTypes>{
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

public:
    int addPoint(const Coord& P, int index, Real& r){
        r = this->model->radius(index);

        SReal baryCoords[1];
        const Coord & p0 = this->model->point1(index);
        const Coord pA = this->model->point2(index) - p0;
        Coord pos = P - p0;
        baryCoords[0] = ( ( pos*pA ) /pA.norm2() );

        if(baryCoords[0] > 1)
            baryCoords[0] = 1;
        else if(baryCoords[0] < 0)
            baryCoords[0] = 0;

        return this->mapper->addPointInLine ( index, baryCoords );
    }
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_MESH_COLLISION)
extern template class SOFA_MESH_COLLISION_API ContactMapper<LineModel, sofa::defaulttype::Vec3Types>;
extern template class SOFA_MESH_COLLISION_API ContactMapper<TriangleModel, sofa::defaulttype::Vec3Types>;
extern template class SOFA_MESH_COLLISION_API ContactMapper<CapsuleModel, sofa::defaulttype::Vec3Types>;

#  ifdef _MSC_VER
// Manual declaration of non-specialized members, to avoid warnings from MSVC.
extern template SOFA_MESH_COLLISION_API void BarycentricContactMapper<LineModel, defaulttype::Vec3Types>::cleanup();
extern template SOFA_MESH_COLLISION_API core::behavior::MechanicalState<defaulttype::Vec3Types>* BarycentricContactMapper<LineModel, defaulttype::Vec3Types>::createMapping(const char*);
extern template SOFA_MESH_COLLISION_API void BarycentricContactMapper<TriangleModel, defaulttype::Vec3Types>::cleanup();
extern template SOFA_MESH_COLLISION_API core::behavior::MechanicalState<defaulttype::Vec3Types>* BarycentricContactMapper<TriangleModel, defaulttype::Vec3Types>::createMapping(const char*);
extern template SOFA_MESH_COLLISION_API void BarycentricContactMapper<CapsuleModel, defaulttype::Vec3Types>::cleanup();
extern template SOFA_MESH_COLLISION_API core::behavior::MechanicalState<defaulttype::Vec3Types>* BarycentricContactMapper<CapsuleModel, defaulttype::Vec3Types>::createMapping(const char*);
#  endif
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif
