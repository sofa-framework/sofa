/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
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
    typedef mapping::MeshMapper<typename MCollisionModel::DataTypes, typename MCollisionModel::DataTypes> MeshMapper;
    MCollisionModel* model;
    MMapping* mapping;
    MeshMapper* mapper;

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
        mapper = new MeshMapper(model->getTopology());
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
