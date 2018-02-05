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
#ifndef SOFA_COMPONENT_COLLISION_BASECONTACTMAPPER_H
#define SOFA_COMPONENT_COLLISION_BASECONTACTMAPPER_H
#include "config.h"

#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/defaulttype/RigidTypes.h>
//#include <SofaBaseMechanics/MechanicalObject.h>

#include <sofa/helper/Factory.h>


namespace sofa
{

namespace core
{

namespace behavior
{
template <class T> class MechanicalState;
} // namespace behavior

} // namespace core

namespace component
{

namespace collision
{

class GenerateStirngID
{
public :

    static const int length = 15;

    static std::string generate(){
        static std::string alphanum = "0123456789!@#$%^&*ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        std::string result;
        result.resize(length);
        for (int i = 0; i < length; i++)
            result[i] = alphanum[rand() % length];

        return result;
    }
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
    virtual void resize(int size) = 0;

    //after detecting a point in collide, this point need to be added to the mapping
    //There are two way for adding the point, by its nature of referentiel : global or local.

    /// Adding a point of the global referentiel to the mapping
    virtual int addPoint(const Coord& /*P*/, int /*elementId*/, Real& /*r*/)
    {
        dmsg_warning("BaseContactMapper") << " addPoint is called but not implemented" ;
        return -1;
    }

    /// Adding a point of the local referentiel(barycentric coordinate) to the mapping
    //TODO use this functions for barycentric contact mapper
    virtual int addBaryPoint(const sofa::defaulttype::Vector3& /*baryP*/, int /*elementId*/, Real& /*r*/)
    {
        dmsg_warning("BaseContactMapper") << " addBaryPoint is called but not implemented" ;
        return -1;
    }

    /// Adding a point of the global referentiel to the mapping, also giving the local referentiel
    /// Note that it cannot have the same name as addPoint otherwise it creates errors when a subclass only implement the version without barycoords
    virtual int addPointB(const Coord& P, int elementId, Real& r, const sofa::defaulttype::Vector3& /*baryP*/)
    {
        return addPoint(P, elementId, r);
    }

    int addPointB(const Coord& P, int elementId, Real& r)
    {
        return addPoint(P, elementId, r);
    }


    virtual void update() = 0;
    virtual void updateXfree() = 0;

    typedef helper::Factory< std::string, BaseContactMapper<DataTypes>, core::CollisionModel* > ContactMapperFactory;
    static BaseContactMapper<DataTypes>* Create(core::CollisionModel* model, const std::string& name = std::string("default"))
    {
        return ContactMapperFactory::CreateObject(name, model);
    }

    template < class TCollisionModel>
    static ContactMapper<TCollisionModel, DataTypes>* create( ContactMapper<TCollisionModel, DataTypes>*, core::CollisionModel* arg)
    {
        TCollisionModel* model = dynamic_cast<TCollisionModel*>(arg);
        if (model == NULL) return NULL;
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

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_BASECONTACTMAPPER_CPP)

#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_COLLISION_API BaseContactMapper<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_COLLISION_API BaseContactMapper<defaulttype::Rigid3fTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_COLLISION_API BaseContactMapper<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_COLLISION_API BaseContactMapper<defaulttype::Rigid3dTypes>;
#endif

#endif

} // namespace collision

} // namespace component

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_BASECONTACTMAPPER_CPP)
namespace helper
{
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Vec3fTypes>, core::CollisionModel* >;
extern template class SOFA_BASE_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Rigid3fTypes>, core::CollisionModel* >;
#endif
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Vec3dTypes>, core::CollisionModel* >;
extern template class SOFA_BASE_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Rigid3dTypes>, core::CollisionModel* >;
#endif
} // namespace helper
#endif

} // namespace sofa

#endif
