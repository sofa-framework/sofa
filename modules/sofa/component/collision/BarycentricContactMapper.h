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
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/mapping/SubsetMapping.h>
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

/// This class will be specialized to whatever mapper is required
template < class TCollisionModel, class DataTypes = typename TCollisionModel::DataTypes >
class ContactMapper;

/// Base class for all mappers using BarycentricMapping
template < class TCollisionModel, class DataTypes >
class BarycentricContactMapper
{
public:
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef typename MCollisionModel::Topology InTopology;
    typedef core::componentmodel::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MMechanicalState;
    typedef component::MechanicalObject<DataTypes> MMechanicalObject;
    typedef mapping::BarycentricMapping< core::componentmodel::behavior::MechanicalMapping< InMechanicalState, MMechanicalState > > MMapping;
    typedef mapping::TopologyBarycentricMapper<InDataTypes, DataTypes> MMapper;
    MCollisionModel* model;
    MMapping* mapping;
    MMapper* mapper;

    BarycentricContactMapper()
        : model(NULL), mapping(NULL), mapper(NULL)
    {
    }

    void cleanup()
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

    MMechanicalState* createMapping(MCollisionModel* model)
    {
        this->model = model;
        simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: BarycentricContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        simulation::tree::GNode* child = new simulation::tree::GNode("contactPoints"); parent->addChild(child); child->updateSimulationContext();
        MMechanicalState* mstate = new MMechanicalObject; child->addObject(mstate);
        //mapping = new MMapping(model->getMechanicalState(), mstate, model->getTopology());
        //mapper = mapping->getMapper();
        mapper = new mapping::BarycentricMapperMeshTopology<InDataTypes, DataTypes>(model->getTopology());
        mapping = new MMapping(model->getMechanicalState(), mstate, mapper);
        child->addObject(mapping);
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

/// Mapper for LineModel
template<class DataTypes>
class ContactMapper<LineModel, DataTypes> : public BarycentricContactMapper<LineModel, DataTypes>
{
public:
    int addPoint(const Vector3& P, int index)
    {
        return this->mapper->createPointInLine(P, index, this->model->getMechanicalState()->getX());
    }

};

/// Mapper for TriangleMeshModel
template<class DataTypes>
class ContactMapper<TriangleModel, DataTypes> : public BarycentricContactMapper<TriangleModel, DataTypes>
{
public:
    int addPoint(const Vector3& P, int index)
    {
        int nbt = this->model->getTopology()->getNbTriangles();
        if (index < nbt)
            return this->mapper->createPointInTriangle(P, index, this->model->getMechanicalState()->getX());
        else
        {
            int qindex = (index - nbt)/2;
            int nbq = this->model->getTopology()->getNbQuads();
            if (qindex < nbq)
                return this->mapper->createPointInQuad(P, qindex, this->model->getMechanicalState()->getX());
            else
            {
                std::cerr << "ContactMapper<TriangleMeshModel>: ERROR invalid contact element index "<<index<<" on a topology with "<<nbt<<" triangles and "<<nbq<<" quads."<<std::endl;
                std::cerr << "model="<<this->model->getName()<<" size="<<this->model->getSize()<<std::endl;
                return -1;
            }
        }
    }
};

///// Mapper for TriangleSetModel
//template<class DataTypes>
//class ContactMapper<TriangleSetModel, DataTypes> : public BarycentricContactMapper<TriangleSetModel, DataTypes>
//{
//public:
//    int addPoint(const Vector3& P, int index)
//    {
//        return this->mapper->createPointInTriangle(P, index, this->model->getMechanicalState()->getX());
//    }
//};

/// Base class for IdentityMapping based mappers
template<class TCollisionModel, class DataTypes>
class IdentityContactMapper
{
public:
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef core::componentmodel::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MMechanicalState;
    typedef component::MechanicalObject<DataTypes> MMechanicalObject;
    typedef mapping::IdentityMapping< core::componentmodel::behavior::MechanicalMapping< InMechanicalState, MMechanicalState > > MMapping;
    MCollisionModel* model;
    MMapping* mapping;

    IdentityContactMapper()
        : model(NULL), mapping(NULL)
    {
    }

    void cleanup()
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

    MMechanicalState* createMapping(MCollisionModel* model)
    {
        this->model = model;
        simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: IdentityContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        simulation::tree::GNode* child = new simulation::tree::GNode("contactPoints"); parent->addChild(child); child->updateSimulationContext();
        MMechanicalState* mstate = new MMechanicalObject; child->addObject(mstate);
        mapping = new MMapping(model->getMechanicalState(), mstate); child->addObject(mapping);
        return mstate;
    }

    void resize(int size)
    {
    }

    int addPoint(const Vector3& P, int index)
    {
        return index;
    }

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

/// Specialization of IdentityContactMapper when mapping to the same DataTypes, as no mapping is required in this case
template<class TCollisionModel>
class IdentityContactMapper<TCollisionModel, typename TCollisionModel::InDataTypes>
{
public:
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef typename MCollisionModel::InDataTypes DataTypes;
    typedef core::componentmodel::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MMechanicalState;
    MCollisionModel* model;

    IdentityContactMapper()
        : model(NULL)
    {
    }

    void cleanup()
    {
    }

    MMechanicalState* createMapping(MCollisionModel* model)
    {
        this->model = model;
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

    double radius(const typename TCollisionModel::Element& /*e*/)
    {
        return 0.0;
    }
};

/// Mapper for PointModel
template<class DataTypes>
class ContactMapper<PointModel, DataTypes> : public IdentityContactMapper<PointModel, DataTypes>
{
public:
};

/// Mapper for SphereModel
template<class TInDataTypes, class DataTypes>
class ContactMapper<TSphereModel<TInDataTypes>, DataTypes> : public IdentityContactMapper<TSphereModel<TInDataTypes>, DataTypes>
{
public:
    double radius(const TSphere<TInDataTypes>& e)
    {
        return e.r();
    }
};

/// Mapper for SphereTreeModel
template<class DataTypes>
class ContactMapper<SphereTreeModel, DataTypes> : public IdentityContactMapper<SphereTreeModel, DataTypes>
{
public:
    double radius(const SingleSphere& e)
    {
        return e.r();
    }
};

/// Base class for all mappers using RigidMapping
template < class TCollisionModel, class DataTypes >
class RigidContactMapper
{
public:
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef core::componentmodel::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MMechanicalState;
    typedef component::MechanicalObject<DataTypes> MMechanicalObject;
    typedef mapping::RigidMapping< core::componentmodel::behavior::MechanicalMapping< InMechanicalState, MMechanicalState > > MMapping;
    MCollisionModel* model;
    simulation::tree::GNode* child;
    MMapping* mapping;
    MMechanicalState* outmodel;
    int nbp;

    RigidContactMapper()
        : model(NULL), child(NULL), mapping(NULL), outmodel(NULL), nbp(0)
    {
    }

    void cleanup()
    {
        if (child!=NULL)
        {
            simulation::tree::getSimulation()->unload(child);
        }
    }

    MMechanicalState* createMapping(MCollisionModel* model)
    {
        this->model = model;
        InMechanicalState* instate = model->getMechanicalState();
        if (instate!=NULL)
        {
            simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(instate->getContext());
            if (parent==NULL)
            {
                std::cerr << "ERROR: RigidContactMapper only works for scenegraph scenes.\n";
                return NULL;
            }
            child = new simulation::tree::GNode("contactPoints"); parent->addChild(child); child->updateSimulationContext();
            outmodel = new MMechanicalObject; child->addObject(outmodel);
            mapping = new MMapping(instate, outmodel); child->addObject(mapping);
        }
        else
        {
            simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
            if (parent==NULL)
            {
                std::cerr << "ERROR: RigidContactMapper only works for scenegraph scenes.\n";
                return NULL;
            }
            child = new simulation::tree::GNode("contactPoints"); parent->addChild(child); child->updateSimulationContext();
            outmodel = new MMechanicalObject; child->addObject(outmodel);
            mapping = NULL;
        }
        return outmodel;
    }

    void resize(int size)
    {
        if (mapping!=NULL)
            mapping->clear(size);
        if (outmodel!=NULL)
            outmodel->resize(size);
        nbp = 0;
    }

    int addPoint(const Vector3& P, int index)
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
            (*outmodel->getX())[i] = P;
        }
        return i;
    }

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

/// Mapper for RigidDistanceGridCollisionModel
template <class DataTypes>
class ContactMapper<RigidDistanceGridCollisionModel,DataTypes> : public RigidContactMapper<RigidDistanceGridCollisionModel,DataTypes>
{
public:
    typedef RigidContactMapper<RigidDistanceGridCollisionModel,DataTypes> Inherit;
    typedef typename Inherit::MMechanicalState MMechanicalState;
    typedef typename Inherit::MCollisionModel MCollisionModel;
    MMechanicalState* createMapping(MCollisionModel* model)
    {
        MMechanicalState* outmodel = Inherit::createMapping(model);
        if (this->child!=NULL && this->mapping==NULL)
        {
            // add velocity visualization
            sofa::component::visualmodel::DrawV* visu = new sofa::component::visualmodel::DrawV;
            this->child->addObject(visu);
            visu->useAlpha.setValue(true);
            visu->vscale.setValue(model->getContext()->getDt());
            sofa::component::mapping::IdentityMapping< core::Mapping< core::componentmodel::behavior::State<DataTypes>, core::componentmodel::behavior::MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > > >* map = new sofa::component::mapping::IdentityMapping< core::Mapping< core::componentmodel::behavior::State<DataTypes> , core::componentmodel::behavior::MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > > > ( outmodel, visu );
            this->child->addObject(map);
            visu->init();
            map->init();
        }
        return outmodel;
    }

    int addPoint(const Vector3& P, int index)
    {
        int i = Inherit::addPoint(P, index);
        if (!this->mapping)
        {
            MCollisionModel* model = this->model;
            MMechanicalState* outmodel = this->outmodel;
            typename DataTypes::Coord& x = (*outmodel->getX())[i];
            typename DataTypes::Deriv& v = (*outmodel->getV())[i];
            if (model->isTransformed(index))
                x = model->getTranslation(index) + model->getRotation(index) * P;
            else
                x = P;
            v = typename DataTypes::Deriv();

            // estimating velocity
            double gdt = model->getPrevDt(index);
            if (gdt > 0.000001)
            {
                if (model->isTransformed(index))
                {
                    v = (x - (model->getPrevTranslation(index) + model->    getPrevRotation(index) * P)) * (1.0/gdt);
                }
                DistanceGrid* prevGrid = model->getPrevGrid(index);
                //DistanceGrid* grid = model->getGrid(index);
                //if (prevGrid != NULL && prevGrid != grid && prevGrid->inGrid(P))
                {
                    DistanceGrid::Coord coefs;
                    int i = prevGrid->index(P, coefs);
                    DistanceGrid::Real_Sofa d = prevGrid->interp(i,coefs);
                    if (rabs(d) < 0.3) // todo : control threshold
                    {
                        DistanceGrid::Coord n = prevGrid->grad(i,coefs);
                        v += n * (d  / ( n.norm() * gdt));
                        //std::cout << "Estimated v at "<<P<<" = "<<v<<" using distance from previous model "<<d<<std::endl;
                    }
                }
            }
        }
        return i;
    }
};

/// Mapper for FFDDistanceGridCollisionModel
template <class DataTypes>
class ContactMapper<FFDDistanceGridCollisionModel,DataTypes> : public BarycentricContactMapper<FFDDistanceGridCollisionModel,DataTypes>
{
public:
    int addPoint(const Vector3& P, int index)
    {
        Vector3 bary;
        int elem = this->model->getDeformCube(index).elem; //getDeformGrid()->findCube(P,bary[0],bary[1],bary[2]);
        bary = this->model->getDeformCube(index).baryCoords(P);
        //if (elem == -1)
        //{
        //    std::cerr<<"WARNING: BarycentricContactMapper from FFDDistanceGridCollisionModel on point no within any the FFD grid."<<std::endl;
        //    elem = model->getDeformGrid()->findNearestCube(P,bary[0],bary[1],bary[2]);
        //}
        return this->mapper->addPointInCube(elem,bary.ptr());
    }
};

/// Base class for all mappers using SubsetMapping
template < class TCollisionModel, class DataTypes >
class SubsetContactMapper
{
public:
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef core::componentmodel::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MMechanicalState;
    typedef component::MechanicalObject<DataTypes> MMechanicalObject;
    typedef mapping::SubsetMapping< core::componentmodel::behavior::MechanicalMapping< InMechanicalState, MMechanicalState > > MMapping;
    MCollisionModel* model;
    simulation::tree::GNode* child;
    MMapping* mapping;
    MMechanicalState* outmodel;
    int nbp;

    SubsetContactMapper()
        : model(NULL), child(NULL), mapping(NULL), outmodel(NULL), nbp(0)
    {
    }

    void cleanup()
    {
        if (child!=NULL)
        {
            simulation::tree::getSimulation()->unload(child);
        }
    }

    MMechanicalState* createMapping(MCollisionModel* model)
    {
        this->model = model;
        InMechanicalState* instate = model->getMechanicalState();
        if (instate!=NULL)
        {
            simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(instate->getContext());
            if (parent==NULL)
            {
                std::cerr << "ERROR: SubsetContactMapper only works for scenegraph scenes.\n";
                return NULL;
            }
            child = new simulation::tree::GNode("contactPoints"); parent->addChild(child); child->updateSimulationContext();
            outmodel = new MMechanicalObject; child->addObject(outmodel);
            mapping = new MMapping(instate, outmodel); child->addObject(mapping);
        }
        else
        {
            simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
            if (parent==NULL)
            {
                std::cerr << "ERROR: SubsetContactMapper only works for scenegraph scenes.\n";
                return NULL;
            }
            child = new simulation::tree::GNode("contactPoints"); parent->addChild(child); child->updateSimulationContext();
            outmodel = new MMechanicalObject; child->addObject(outmodel);
            mapping = NULL;
        }
        return outmodel;
    }

    void resize(int size)
    {
        if (mapping!=NULL)
            mapping->clear(size);
        if (outmodel!=NULL)
            outmodel->resize(size);
        nbp = 0;
    }

    int addPoint(const Vector3& P, int index)
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
            (*outmodel->getX())[i] = P;
        }
        return i;
    }

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

} // namespace collision

} // namespace component

} // namespace sofa

#endif
