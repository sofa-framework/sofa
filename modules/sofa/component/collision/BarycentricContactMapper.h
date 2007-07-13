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
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/SphereTreeModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/DistanceGridCollisionModel.h>
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/component/visualmodel/DrawV.h>
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

template <>
class BarycentricContactMapper<RigidDistanceGridCollisionModel>
{
public:
    typedef RigidDistanceGridCollisionModel MCollisionModel;
    typedef sofa::defaulttype::Vec3Types DataTypes;
    typedef sofa::core::componentmodel::behavior::MechanicalState<Rigid3Types> InMechanicalState;
    typedef sofa::core::componentmodel::behavior::MechanicalState<DataTypes> MMechanicalState;
    typedef sofa::component::MechanicalObject<DataTypes> MMechanicalObject;
    typedef mapping::RigidMapping<core::componentmodel::behavior::MechanicalMapping< InMechanicalState, MMechanicalState > > MMapping;
    MCollisionModel* model;
    MMapping* mapping;
    MMechanicalState* outmodel;
    int nbp;

    BarycentricContactMapper(MCollisionModel* model)
        : model(model), mapping(NULL), outmodel(NULL), nbp(0)
    {
    }

    ~BarycentricContactMapper()
    {
        if (mapping!=NULL)
        {
            simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getRigidModel()->getContext());
            if (parent!=NULL)
            {
                simulation::tree::GNode* child = dynamic_cast<simulation::tree::GNode*>(mapping->getContext());
                simulation::tree::Simulation::unload(child);
                //child->removeObject(outmodel);
                //if (mapping)
                //{
                //	child->removeObject(mapping);
                //	delete mapping;
                //}
                //parent->removeChild(child);
                //delete outmodel;
                //delete child;
            }
        }
    }

    MMechanicalState* createMapping()
    {
        InMechanicalState* instate = model->getRigidModel();
        if (instate!=NULL)
        {
            simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(instate->getContext());
            if (parent==NULL)
            {
                std::cerr << "ERROR: BarycentricContactMapper only works for scenegraph scenes.\n";
                return NULL;
            }
            simulation::tree::GNode* child = new simulation::tree::GNode("contactPoints"); parent->addChild(child); child->updateContext();
            outmodel = new MMechanicalObject; child->addObject(outmodel);
            mapping = new MMapping(model->getRigidModel(), outmodel); child->addObject(mapping);
            return outmodel;
        }
        else
        {
            simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
            if (parent==NULL)
            {
                std::cerr << "ERROR: BarycentricContactMapper only works for scenegraph scenes.\n";
                return NULL;
            }
            simulation::tree::GNode* child = new simulation::tree::GNode("contactPoints"); parent->addChild(child); child->updateContext();
            outmodel = new MMechanicalObject; child->addObject(outmodel);
            mapping = NULL;

            // add velocity visualization
            sofa::component::visualmodel::DrawV* visu = new sofa::component::visualmodel::DrawV;
            child->addObject(visu);
            visu->useAlpha.setValue(true);
            visu->vscale.setValue(model->getContext()->getDt());
            sofa::component::mapping::IdentityMapping< core::Mapping< MMechanicalState , core::componentmodel::behavior::MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > > >* map = new sofa::component::mapping::IdentityMapping< core::Mapping< MMechanicalState , core::componentmodel::behavior::MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > > > ( outmodel, visu );
            child->addObject(map);
            visu->init();
            map->init();



            return outmodel;
        }
    }

    void resize(int size)
    {
        if (mapping!=NULL)
        {
            mapping->clear();
        }
        if (outmodel!=NULL)
        {
            outmodel->resize(size);
        }
        nbp = 0;
    }

    int addPoint(const Vector3& P, int index)
    {
        int i = nbp++; //outmodel->getX()->size();
        if ((int)outmodel->getX()->size() <= i)
            outmodel->resize(i+1);
        if (mapping)
        {
            mapping->index.setValue(index);
            (*outmodel->getX())[i] = P;
        }
        else
        {
            DataTypes::Coord& x = (*outmodel->getX())[i];
            DataTypes::Deriv& v = (*outmodel->getV())[i];
            if (model->isTransformed(index))
            {
                x = model->getTranslation(index) + model->getRotation() * P;
                v = (x - (model->getPrevTranslation(index) + model->getPrevRotation() * P)) * (1.0/model->getContext()->getDt());
            }
            else
            {
                x = P;
                v = DataTypes::Deriv();
            }

            // estimating velocity
            double gdt = model->getPrevDt(index);
            if (gdt > 0.000001)
            {
                DistanceGrid* prevGrid = model->getPrevGrid(index);
                //DistanceGrid* grid = model->getGrid(index);
                //if (prevGrid != NULL && prevGrid != grid && prevGrid->inGrid(P))
                {
                    DistanceGrid::Coord coefs;
                    int i = prevGrid->index(P, coefs);
                    DistanceGrid::Real d = prevGrid->interp(i,coefs);
                    if (rabs(d) < 0.3) // todo : control threshold
                    {
                        DistanceGrid::Coord n = prevGrid->grad(i,coefs);
                        v += n * (d  / ( n.norm() * gdt));
                        //std::cout << "Estimated v at "<<P<<" = "<<v<<" using distance from previous model "<<d<<std::endl;
                    }
                }
            }
            (*outmodel->getV())[i] = v;
        }
        return i;
    }

    void update()
    {
        if (mapping!=NULL)
        {
            mapping->init();
            mapping->updateMapping();
        }
    }

    double radius(const MCollisionModel::Element& /*e*/)
    {
        return 0.0;
    }
};

template <>
class BarycentricContactMapper<FFDDistanceGridCollisionModel>
{
public:
    typedef FFDDistanceGridCollisionModel MCollisionModel;
    typedef sofa::defaulttype::Vec3Types DataTypes;
    typedef sofa::core::componentmodel::behavior::MechanicalState<DataTypes> MMechanicalState;
    typedef sofa::component::MechanicalObject<DataTypes> MMechanicalObject;
    typedef mapping::BarycentricMapping<core::componentmodel::behavior::MechanicalMapping< MMechanicalState, MMechanicalState > > MMapping;
    typedef mapping::RegularGridMapper<DataTypes, DataTypes> MMapper;
    MCollisionModel* model;
    MMapping* mapping;
    MMapper* mapper;

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
        simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getDeformModel()->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: BarycentricContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        simulation::tree::GNode* child = new simulation::tree::GNode("contactPoints"); parent->addChild(child); child->updateContext();
        MMechanicalState* mstate = new MMechanicalObject; child->addObject(mstate);
        mapper = new MMapper(model->getDeformGrid());
        mapping = new MMapping(model->getDeformModel(), mstate, mapper); child->addObject(mapping);
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

    int addPoint(const Vector3& P, int /*index*/)
    {
        Vector3 bary;
        int elem = model->getDeformGrid()->findCube(P,bary[0],bary[1],bary[2]);
        if (elem == -1)
        {
            std::cerr<<"WARNING: BarycentricContactMapper from FFDDistanceGridCollisionModel on point no within any the FFD grid."<<std::endl;
            elem = model->getDeformGrid()->findNearestCube(P,bary[0],bary[1],bary[2]);
        }
        return mapper->addPointInCube(elem,bary.ptr());
    }

    void update()
    {
        if (mapping!=NULL)
        {
            mapping->updateMapping();
        }
    }

    double radius(const MCollisionModel::Element& /*e*/)
    {
        return 0.0;
    }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
