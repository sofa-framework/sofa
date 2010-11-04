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
#ifndef SOFA_COMPONENT_COLLISION_SUBSETCONTACTMAPPER_H
#define SOFA_COMPONENT_COLLISION_SUBSETCONTACTMAPPER_H

#include <sofa/helper/system/config.h>
#include <sofa/helper/Factory.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/mapping/SubsetMapping.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/collision/BaseContactMapper.h>
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


/// Base class for all mappers using SubsetMapping
template < class TCollisionModel, class DataTypes >
class SubsetContactMapper : public BaseContactMapper<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef TCollisionModel MCollisionModel;
    typedef typename MCollisionModel::InDataTypes InDataTypes;
    typedef core::behavior::MechanicalState<InDataTypes> InMechanicalState;
    typedef core::behavior::MechanicalState<typename SubsetContactMapper::DataTypes> MMechanicalState;
    typedef component::container::MechanicalObject<typename SubsetContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::SubsetMapping< InDataTypes, typename SubsetContactMapper::DataTypes > MMapping;
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
            Data<VecCoord>* d_x = outmodel->write(core::VecCoordId::position());
            VecCoord& x = *d_x->beginEdit();
            x[i] = P;
            d_x->endEdit();
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

            ((core::BaseMapping*)mapping)->apply(core::VecCoordId::position(), core::ConstVecCoordId::position());
            ((core::BaseMapping*)mapping)->applyJ(core::VecDerivId::velocity(), core::ConstVecDerivId::velocity());
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

            ((core::BaseMapping*)mapping)->apply(core::VecCoordId::freePosition(), core::ConstVecCoordId::freePosition());
        }
    }

    //double radius(const typename TCollisionModel::Element& /*e*/)
    //{
    //    return 0.0;
    //}
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_COLLISION_SUBSETCONTACTMAPPER_H */
