#ifndef SOFA_COMPONENTS_BARYCENTRICCONTACTMAPPER_H
#define SOFA_COMPONENTS_BARYCENTRICCONTACTMAPPER_H

#include "BarycentricMapping.h"
#include "Sofa-old/Core/MechanicalObject.h"
#include "Graph/GNode.h"
#include "SphereModel.h"
#include "SphereTreeModel.h"
#include "TriangleModel.h"
#include "LineModel.h"
#include "PointModel.h"

#include <iostream>

namespace Sofa
{

namespace Components
{

using namespace Common;

template < class TCollisionModel >
class BarycentricContactMapper
{
public:
    typedef TCollisionModel MCollisionModel;
    typedef Core::MechanicalModel<typename MCollisionModel::DataTypes> MMechanicalModel;
    typedef Core::MechanicalObject<typename MCollisionModel::DataTypes> MMechanicalObject;
    typedef BarycentricMapping<Core::MechanicalMapping< MMechanicalModel, MMechanicalModel > > MMapping;
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
            Graph::GNode* parent = dynamic_cast<Graph::GNode*>(model->getContext());
            if (parent!=NULL)
            {
                Graph::GNode* child = dynamic_cast<Graph::GNode*>(mapping->getContext());
                child->removeObject(mapping->getTo());
                child->removeObject(mapping);
                parent->removeChild(child);
                delete mapping->getTo();
                delete mapping;
                delete child;
            }
        }
    }

    MMechanicalModel* createMapping()
    {
        Graph::GNode* parent = dynamic_cast<Graph::GNode*>(model->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: BarycentricContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        Graph::GNode* child = new Graph::GNode("contactPoints"); parent->addChild(child); child->updateContext();
        MMechanicalModel* mmodel = new MMechanicalObject; child->addObject(mmodel);
        mapper = new typename MMapping::MeshMapper(model->getTopology());
        mapping = new MMapping(model->getMechanicalModel(), mmodel, mapper); child->addObject(mapping);
        return mmodel;
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
    return mapper->createPointInLine(P, index, model->getMechanicalModel()->getX());
}

template<>
inline int BarycentricContactMapper<TriangleModel>::addPoint(const Vector3& P, int index)
{
    if (index < model->getTopology()->getNbTriangles())
        return mapper->createPointInTriangle(P, index, model->getMechanicalModel()->getX());
    else
        return mapper->createPointInQuad(P, (index - model->getTopology()->getNbTriangles())/2, model->getMechanicalModel()->getX());
}

template<>
class BarycentricContactMapper<PointModel>
{
public:
    typedef PointModel MCollisionModel;
    typedef Core::MechanicalModel<MCollisionModel::DataTypes> MMechanicalModel;
    MCollisionModel* model;

    BarycentricContactMapper(MCollisionModel* model)
        : model(model)
    {
    }

    MMechanicalModel* createMapping()
    {
        return model->getMechanicalModel();
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
    typedef Core::MechanicalModel<MCollisionModel::DataTypes> MMechanicalModel;
    MCollisionModel* model;

    BarycentricContactMapper(MCollisionModel* model)
        : model(model)
    {
    }

    MMechanicalModel* createMapping()
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
    typedef Core::MechanicalModel<MCollisionModel::DataTypes> MMechanicalModel;
    MCollisionModel* model;

    BarycentricContactMapper(MCollisionModel* model)
        : model(model)
    {
    }

    MMechanicalModel* createMapping()
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

} // namespace Components

} // namespace Sofa

#endif
