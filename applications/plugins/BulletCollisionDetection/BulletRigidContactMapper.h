#ifndef BULLET_RIGID_CONTACT_MAPPER_H
#define BULLET_RIGID_CONTACT_MAPPER_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/Factory.h>
#include <SofaBaseMechanics/BarycentricMapping.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <SofaRigid/RigidMapping.h>
#include <SofaBaseMechanics/SubsetMapping.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <SofaBaseCollision/BaseContactMapper.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMiscCollision/TetrahedronModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/OBBModel.h>
#include <SofaBaseCollision/RigidCapsuleModel.h>
#include <SofaBaseCollision/CylinderModel.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <sofa/core/VecId.h>
#include <iostream>
#include "BulletConvexHullModel.h"

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

/// This is copy-past from RigidContactMapper for linker problems.
/// Base class for all mappers using RigidMapping
template < class TCollisionModel, class DataTypes >
class BulletRigidContactMapper : public BaseContactMapper<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef core::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::behavior::MechanicalState<typename BulletRigidContactMapper::DataTypes> MMechanicalState;
    typedef component::container::MechanicalObject<typename BulletRigidContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::RigidMapping< InDataTypes, typename BulletRigidContactMapper::DataTypes > MMapping;

    MCollisionModel* model;
    simulation::Node::SPtr child;
    typename MMapping::SPtr mapping;
    typename MMechanicalState::SPtr outmodel;
    int nbp;

protected:
    BulletRigidContactMapper()
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
class ContactMapper<BulletConvexHullModel,TVec3Types > : public BulletRigidContactMapper<BulletConvexHullModel, TVec3Types >{
public:
    //typedef BulletRigidContactMapper<BulletConvexHullModel, TVec3Types > Parent;

//I don't know why this is necessary but it is when i want to load this plugin, shit !
//    virtual typename Parent::MMechanicalState* createMapping(const char* name="contactPoints"){return Parent::createMapping(name);}

//    virtual void cleanup(){return Parent::cleanup();}

    int addPoint(const typename TVec3Types::Coord & P, int index,typename TVec3Types::Real & r)
    {
        const typename TVec3Types::Coord & cP = P - this->model->center();
        const Quaternion & ori = this->model->orientation();

        return BulletRigidContactMapper<BulletConvexHullModel,TVec3Types >::addPoint(ori.inverseRotate(cP),index,r);
    }
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BULLETCOLLISIONDETECTION)
extern template class SOFA_BULLETCOLLISIONDETECTION_API ContactMapper<BulletConvexHullModel,Vec3Types>;
#endif

}
}
}

#endif
