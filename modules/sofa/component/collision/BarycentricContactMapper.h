/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICCONTACTMAPPER_H
#define SOFA_COMPONENT_COLLISION_BARYCENTRICCONTACTMAPPER_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/Factory.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/mapping/SubsetMapping.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/SphereTreeModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/TetrahedronModel.h>
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

/// Base class common to all mappers able to provide a MechanicalState of a given type
template <class TDataTypes>
class BaseContactMapper
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MMechanicalState;
    virtual ~BaseContactMapper() {}
    virtual MMechanicalState* createMapping(const char* name = "contactPoints") = 0;
    virtual void cleanup() = 0;
    virtual void resize(int size) = 0;
    virtual int addPoint(const Coord& c, int index, Real& r) = 0;
    virtual void update() = 0;
    virtual void updateXfree() = 0;

    typedef helper::Factory< std::string, BaseContactMapper<DataTypes>, core::CollisionModel* > ContactMapperFactory;
    static BaseContactMapper<DataTypes>* Create(core::CollisionModel* model, const std::string& name = std::string("default"))
    {
        return ContactMapperFactory::CreateObject(name, model);
    }

    template < class TCollisionModel>
    static void create( ContactMapper<TCollisionModel, DataTypes>*& obj, core::CollisionModel* arg)
    {
        TCollisionModel* model = dynamic_cast<TCollisionModel*>(arg);
        if (model == NULL) return;
        obj = new ContactMapper<TCollisionModel, DataTypes>;
        obj->setCollisionModel(model);
    }

};

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
    typedef core::componentmodel::behavior::MechanicalState< InDataTypes> InMechanicalState;
    typedef core::componentmodel::behavior::MechanicalState<  typename BarycentricContactMapper::DataTypes> MMechanicalState;
    typedef component::container::MechanicalObject<typename BarycentricContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::BarycentricMapping< core::componentmodel::behavior::MechanicalMapping< InMechanicalState, MMechanicalState > > MMapping;
    typedef mapping::TopologyBarycentricMapper<InDataTypes, typename BarycentricContactMapper::DataTypes> MMapper;
    MCollisionModel* model;
    MMapping* mapping;
    MMapper* mapper;

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

    void updateXfree()
    {
        if (mapping!=NULL)
        {
            mapping->propagateXfree();
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
        return this->mapper->createPointInLine(P, index, this->model->getMechanicalState()->getX());
    }

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
            return this->mapper->createPointInTriangle(P, index, this->model->getMechanicalState()->getX());
        else
        {
            int qindex = (index - nbt)/2;
            int nbq = this->model->getMeshTopology()->getNbQuads();
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

/// Mapper for TetrahedronModel
template<class DataTypes>
class ContactMapper<TetrahedronModel, DataTypes> : public BarycentricContactMapper<TetrahedronModel, DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    int addPoint(const Coord& P, int index, Real&)
    {
        Tetrahedron t(this->model, index);
        Vector3 b = t.getBary(P);
        return this->mapper->addPointInTetra(index, b.ptr());
    }
};

/// Base class for IdentityMapping based mappers
template<class TCollisionModel, class DataTypes>
class IdentityContactMapper : public BaseContactMapper<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef core::componentmodel::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::componentmodel::behavior::MechanicalState<typename IdentityContactMapper::DataTypes> MMechanicalState;
    typedef component::container::MechanicalObject<typename IdentityContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::IdentityMapping< core::componentmodel::behavior::MechanicalMapping< InMechanicalState, MMechanicalState > > MMapping;
    MCollisionModel* model;
    MMapping* mapping;

    IdentityContactMapper()
        : model(NULL), mapping(NULL)
    {
    }

    void setCollisionModel(MCollisionModel* model)
    {
        this->model = model;
    }

    void cleanup();

    MMechanicalState* createMapping(const char* name="contactPoints");

    void resize(int /*size*/)
    {
    }

    int addPoint(const Coord&, int index, Real&)
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

    void updateXfree()
    {
        if (mapping!=NULL)
        {
            mapping->propagateXfree();
        }
    }
};

/// Specialization of IdentityContactMapper when mapping to the same DataTypes, as no mapping is required in this case
template<class TCollisionModel>
class IdentityContactMapper<TCollisionModel, typename TCollisionModel::InDataTypes> : public BaseContactMapper<typename TCollisionModel::InDataTypes>
{
public:
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef typename MCollisionModel::InDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef core::componentmodel::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MMechanicalState;
    MCollisionModel* model;

    IdentityContactMapper()
        : model(NULL)
    {
    }

    void setCollisionModel(MCollisionModel* model)
    {
        this->model = model;
    }

    void cleanup()
    {
    }

    MMechanicalState* createMapping(const char* /*name*/="contactPoints")
    {
        if (model==NULL) return NULL;
        return model->getMechanicalState();
    }

    void resize(int /*size*/)
    {
    }

    int addPoint(const Coord& /*P*/, int index, Real&)
    {
        return index;
    }

    void update()
    {
    }

    void updateXfree()
    {
    }
};

/// Mapper for PointModel
template<class DataTypes>
class ContactMapper<PointModel, DataTypes> : public IdentityContactMapper<PointModel, DataTypes>
{
public:
};

/// Mapper for SphereModel
template<class DataTypes>
class ContactMapper<SphereModel, DataTypes> : public IdentityContactMapper<SphereModel, DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    int addPoint(const Coord& /*P*/, int index, Real& r)
    {
        Sphere e(this->model, index);
        r = e.r();
        return index;
    }
};

/// Mapper for SphereTreeModel
template<class DataTypes>
class ContactMapper<SphereTreeModel, DataTypes> : public IdentityContactMapper<SphereTreeModel, DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    int addPoint(const Coord& /*P*/, int index, Real& r)
    {
        SingleSphere e(this->model, index);
        r = e.r();
        return index;
    }
};

/// Base class for all mappers using RigidMapping
template < class TCollisionModel, class DataTypes >
class RigidContactMapper : public BaseContactMapper<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef core::componentmodel::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::componentmodel::behavior::MechanicalState<typename RigidContactMapper::DataTypes> MMechanicalState;
    typedef component::container::MechanicalObject<typename RigidContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::RigidMapping< core::componentmodel::behavior::MechanicalMapping< InMechanicalState, MMechanicalState > > MMapping;
    MCollisionModel* model;
    simulation::Node* child;
    MMapping* mapping;
    MMechanicalState* outmodel;
    int nbp;

    RigidContactMapper()
        : model(NULL), child(NULL), mapping(NULL), outmodel(NULL), nbp(0)
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

    void updateXfree()
    {
        if (mapping!=NULL)
        {
            mapping->propagateXfree();
        }
    }

};

/// Mapper for RigidDistanceGridCollisionModel
template <class DataTypes>
class ContactMapper<RigidDistanceGridCollisionModel,DataTypes> : public RigidContactMapper<RigidDistanceGridCollisionModel,DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef RigidContactMapper<RigidDistanceGridCollisionModel,DataTypes> Inherit;
    typedef typename Inherit::MMechanicalState MMechanicalState;
    typedef typename Inherit::MCollisionModel MCollisionModel;
    MMechanicalState* createMapping(const char* name="contactPoints");

    int addPoint(const Coord& P, int index, Real& r)
    {
        Coord trans = this->model->getInitTranslation();
        int i = Inherit::addPoint(P+trans, index, r);
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
                    SReal d = prevGrid->interp(i,coefs);
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
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    int addPoint(const Coord& P, int index, Real&)
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
class SubsetContactMapper : public BaseContactMapper<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef core::componentmodel::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::componentmodel::behavior::MechanicalState<typename SubsetContactMapper::DataTypes> MMechanicalState;
    typedef component::container::MechanicalObject<typename SubsetContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::SubsetMapping< core::componentmodel::behavior::MechanicalMapping< InMechanicalState, MMechanicalState > > MMapping;
    MCollisionModel* model;
    simulation::Node* child;
    MMapping* mapping;
    MMechanicalState* outmodel;
    int nbp;
    bool needInit;

    SubsetContactMapper()
        : model(NULL), child(NULL), mapping(NULL), outmodel(NULL), nbp(0), needInit(false)
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
            i = mapping->addPoint(index);
            needInit = true;
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
            if (needInit)
            {
                mapping->init();
                needInit = false;
            }
            mapping->updateMapping();
        }
    }

    void updateXfree()
    {
        if (mapping!=NULL)
        {
            if (needInit)
            {
                mapping->init();
                needInit = false;
            }
            mapping->propagateXfree();
        }
    }

    //double radius(const typename TCollisionModel::Element& /*e*/)
    //{
    //    return 0.0;
    //}
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


#if defined(WIN32) && !defined(SOFA_BUILD_COMPONENT_COLLISION)
extern template class SOFA_COMPONENT_COLLISION_API BaseContactMapper<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_COLLISION_API ContactMapper<SphereModel>;
extern template class SOFA_COMPONENT_COLLISION_API ContactMapper<SphereTreeModel>;
extern template class SOFA_COMPONENT_COLLISION_API ContactMapper<PointModel>;
extern template class SOFA_COMPONENT_COLLISION_API ContactMapper<LineModel>;
extern template class SOFA_COMPONENT_COLLISION_API ContactMapper<TriangleModel>;
extern template class SOFA_COMPONENT_COLLISION_API ContactMapper<TetrahedronModel>;
extern template class SOFA_COMPONENT_COLLISION_API ContactMapper<RigidDistanceGridCollisionModel>;
extern template class SOFA_COMPONENT_COLLISION_API ContactMapper<FFDDistanceGridCollisionModel>;
#endif

} // namespace collision

} // namespace component

#if defined(WIN32) && !defined(SOFA_BUILD_COMPONENT_COLLISION)
namespace helper
{
extern template class SOFA_COMPONENT_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Vec3Types>, core::CollisionModel* >;
} // namespace helper
#endif

} // namespace sofa

#endif
