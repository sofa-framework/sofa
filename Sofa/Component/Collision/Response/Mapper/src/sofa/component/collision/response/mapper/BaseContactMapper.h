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
#include <sofa/core/CollisionModel.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa::component::collision::response::mapper
{

class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API GenerateStringID
{
public :
    static const int length {15};
    static std::string generate() ;
};


/// This class will be specialized to whatever mapper is required
template < class TCollisionModel, class DataTypes = typename TCollisionModel::DataTypes >
class ContactMapper;

/// Base class common to all mappers able to provide a MechanicalState of a given type
template <class TDataTypes>
class BaseContactMapper
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef core::behavior::MechanicalState<DataTypes> MMechanicalState;

    virtual ~BaseContactMapper() {}
    virtual MMechanicalState* createMapping(const char* name = "contactPoints") = 0;
    virtual void cleanup() = 0;
    virtual void resize(Size size) = 0;

    //after detecting a point in collide, this point need to be added to the mapping
    //There are two way for adding the point, by its nature of referentiel : global or local.

    /// Adding a point of the global referentiel to the mapping
    virtual Index addPoint(const Coord& /*P*/, Index /*elementId*/, Real& /*r*/)
    {
        dmsg_warning("BaseContactMapper") << " addPoint is called but not implemented" ;
        return sofa::InvalidID;
    }

    /// Adding a point of the global referentiel to the mapping, also giving the local referentiel
    /// Note that it cannot have the same name as addPoint otherwise it creates errors when a subclass only implement the version without barycoords
    virtual Index addPointB(const Coord& P, Index elementId, Real& r, const sofa::type::Vec3& /*baryP*/)
    {
        return addPoint(P, elementId, r);
    }

    Index addPointB(const Coord& P, Index elementId, Real& r)
    {
        return addPoint(P, elementId, r);
    }


    virtual void update() = 0;
    virtual void updateXfree() = 0;

    typedef helper::Factory< std::string, BaseContactMapper<DataTypes>, core::CollisionModel* > ContactMapperFactory;
    static BaseContactMapper<DataTypes>* Create(core::CollisionModel* model, const std::string& name = std::string("PenalityContactForceField"))
    {
        return ContactMapperFactory::CreateObject(name, model);
    }

    template < class TCollisionModel>
    static ContactMapper<TCollisionModel, DataTypes>* create( ContactMapper<TCollisionModel, DataTypes>*, core::CollisionModel* arg)
    {
        TCollisionModel* model = dynamic_cast<TCollisionModel*>(arg);
        if (model == nullptr) return nullptr;
        ContactMapper<TCollisionModel, DataTypes>* obj = new ContactMapper<TCollisionModel, DataTypes>;
        obj->setCollisionModel(model);
        return obj;
    }


};

template < class Mapper >
class ContactMapperCreator : public helper::Creator < typename Mapper::ContactMapperFactory, Mapper >
{
public:
    typedef helper::Creator < typename Mapper::ContactMapperFactory, Mapper > Inherit;
    ContactMapperCreator(std::string name, bool multi = true)
        : Inherit(name, multi)
    {
    }
};

#if !defined(SOFA_COMPONENT_COLLISION_BASECONTACTMAPPER_CPP)
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API BaseContactMapper<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API BaseContactMapper<defaulttype::Rigid3Types>;

#endif

} // namespace sofa::component::collision::response::mapper


#if !defined(SOFA_COMPONENT_COLLISION_BASECONTACTMAPPER_CPP)
namespace sofa::helper
{
//extern template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API Factory< std::string, sofa::component::collision::response::mapper::BaseContactMapper<defaulttype::Vec2Types>, core::CollisionModel* >;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API Factory< std::string, sofa::component::collision::response::mapper::BaseContactMapper<defaulttype::Vec3Types>, core::CollisionModel* >;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_MAPPER_API Factory< std::string, sofa::component::collision::response::mapper::BaseContactMapper<defaulttype::Rigid3Types>, core::CollisionModel* >;
} // namespace sofa::helper
#endif
