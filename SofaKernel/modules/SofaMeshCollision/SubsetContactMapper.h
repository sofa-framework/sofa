/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_COLLISION_SUBSETCONTACTMAPPER_H
#define SOFA_COMPONENT_COLLISION_SUBSETCONTACTMAPPER_H
#include "config.h"

#include <sofa/helper/system/config.h>
#include <sofa/helper/Factory.h>
#include <SofaBaseMechanics/BarycentricMapping.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <SofaRigid/RigidMapping.h>
#include <SofaBaseMechanics/SubsetMapping.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <SofaBaseCollision/BaseContactMapper.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
//#include <SofaMiscCollision/TetrahedronModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
//#include <SofaVolumetricData/DistanceGridCollisionModel.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <iostream>


namespace sofa
{

namespace component
{

namespace collision
{


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
    simulation::Node::SPtr child;
    typename MMapping::SPtr mapping;
    typename MMechanicalState::SPtr outmodel;
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
        if ((int)outmodel->getSize() <= i)
            outmodel->resize(i+1);
        if (mapping)
        {
            i = mapping->addPoint(index);
            needInit = true;
        }
        else
        {
            helper::WriteAccessor<Data<VecCoord> > d_x = *outmodel->write(core::VecCoordId::position());
            VecCoord& x = d_x.wref();
            x[i] = P;
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
            core::BaseMapping* map = mapping.get();
            map->apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::position(), core::ConstVecCoordId::position());
            map->applyJ(core::MechanicalParams::defaultInstance(), core::VecDerivId::velocity(), core::ConstVecDerivId::velocity());
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

            core::BaseMapping* map = mapping.get();
            map->apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::freePosition(), core::ConstVecCoordId::freePosition());
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
