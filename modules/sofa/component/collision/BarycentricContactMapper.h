#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICCONTACTMAPPER_H
#define SOFA_COMPONENT_COLLISION_BARYCENTRICCONTACTMAPPER_H

#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/SphereTreeModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <iostream>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

template < class TCollisionModel >
class BarycentricContactMapper
{
public:
    typedef TCollisionModel MCollisionModel;
    typedef core::componentmodel::behavior::MechanicalState<typename MCollisionModel::DataTypes> MMechanicalState;
    typedef component::MechanicalObject<typename MCollisionModel::DataTypes> MMechanicalObject;
    typedef mapping::BarycentricMapping<core::componentmodel::behavior::MechanicalMapping< MMechanicalState, MMechanicalState > > MMapping;
    MCollisionModel* model;
    MMapping* mapping;
    typename MMapping::MeshMapper* mapper;

    BarycentricContactMapper(MCollisionModel* model)
        : model(model), mapping(NULL), mapper(NULL)
    {
    }

    ~BarycentricContactMapper()
    {
        if (mapping!=NULL)
        {
            simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
            if (parent!=NULL)
            {
                simulation::tree::GNode* child = dynamic_cast<simulation::tree::GNode*>(mapping->getContext());
                child->removeObject(mapping->getTo());
                child->removeObject(mapping);
                parent->removeChild(child);
                delete mapping->getTo();
                delete mapping;
                delete child;
            }
        }
    }

    MMechanicalState* createMapping()
    {
        simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: BarycentricContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        simulation::tree::GNode* child = new simulation::tree::GNode("contactPoints"); parent->addChild(child); child->updateContext();
        MMechanicalState* mstate = new MMechanicalObject; child->addObject(mstate);
        mapper = new typename MMapping::MeshMapper(model->getTopology());
        mapping = new MMapping(model->getMechanicalState(), mstate, mapper); child->addObject(mapping);
        return mstate;
    }

    void resize(int size)
    {
        if (mapping!=NULL)
        {
            mapper->clear();
            mapping->getMechTo()->resize(size);
        }
    }

    int addPoint(const Vector3& P, int index);

    void update()
    {
        if (mapping!=NULL)
        {
            mapping->updateMapping();
        }
    }

    double radius(const typename TCollisionModel::Element& /*e*/)
    {
        return 0.0;
    }
};

template<>
inline int BarycentricContactMapper<LineModel>::addPoint(const Vector3& P, int index)
{
    return mapper->createPointInLine(P, index, model->getMechanicalState()->getX());
}

template<>
inline int BarycentricContactMapper<TriangleModel>::addPoint(const Vector3& P, int index)
{
    if (index < model->getTopology()->getNbTriangles())
        return mapper->createPointInTriangle(P, index, model->getMechanicalState()->getX());
    else
        return mapper->createPointInQuad(P, (index - model->getTopology()->getNbTriangles())/2, model->getMechanicalState()->getX());
}

template<>
class BarycentricContactMapper<PointModel>
{
public:
    typedef PointModel MCollisionModel;
    typedef core::componentmodel::behavior::MechanicalState<MCollisionModel::DataTypes> MMechanicalState;
    MCollisionModel* model;

    BarycentricContactMapper(MCollisionModel* model)
        : model(model)
    {
    }

    MMechanicalState* createMapping()
    {
        return model->getMechanicalState();
    }

    void resize(int /*size*/)
    {
    }

    int addPoint(const Vector3& /*P*/, int index)
    {
        return index;
    }

    void update()
    {
    }

    double radius(const Point& /*e*/)
    {
        return 0.0;
    }
};

template <>
class BarycentricContactMapper<SphereTreeModel>
{
public:
    typedef SphereTreeModel MCollisionModel;
    typedef core::componentmodel::behavior::MechanicalState<MCollisionModel::DataTypes> MMechanicalState;
    MCollisionModel* model;

    BarycentricContactMapper(MCollisionModel* model)
        : model(model)
    {
    }

    MMechanicalState* createMapping()
    {
        return model;
    }

    void resize(int /*size*/)
    {
    }

    int addPoint(const Vector3& /*P*/, int index)
    {
        return index;
    }

    void update()
    {
    }

    double radius(const SingleSphere& e)
    {
        return e.r();
    }
};


template <>
class BarycentricContactMapper<SphereModel>
{
public:
    typedef SphereModel MCollisionModel;
    typedef core::componentmodel::behavior::MechanicalState<MCollisionModel::DataTypes> MMechanicalState;
    MCollisionModel* model;

    BarycentricContactMapper(MCollisionModel* model)
        : model(model)
    {
    }

    MMechanicalState* createMapping()
    {
        return model;
    }

    void resize(int /*size*/)
    {
    }

    int addPoint(const Vector3& /*P*/, int index)
    {
        return index;
    }

    void update()
    {
    }

    double radius(const Sphere& e)
    {
        return e.r();
    }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
