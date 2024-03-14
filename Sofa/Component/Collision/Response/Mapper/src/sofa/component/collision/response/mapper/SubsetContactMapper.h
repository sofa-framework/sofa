/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/collision/response/mapper/config.h>

#include <sofa/helper/Factory.h>
#include <sofa/component/mapping/linear/SubsetMapping.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/collision/response/mapper/BaseContactMapper.h>


namespace sofa::component::collision::response::mapper
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
    typedef component::statecontainer::MechanicalObject<typename SubsetContactMapper::DataTypes> MMechanicalObject;
    typedef mapping::linear::SubsetMapping< InDataTypes, typename SubsetContactMapper::DataTypes > MMapping;
    MCollisionModel* model;
    simulation::Node::SPtr child;
    typename MMapping::SPtr mapping;
    typename MMechanicalState::SPtr outmodel;
    using Index = sofa::Index;

    Size nbp;
    bool needInit;

    SubsetContactMapper();

    void setCollisionModel(MCollisionModel* model);

    MMechanicalState* createMapping(const char* name="contactPoints");

    void cleanup();
    void resize(Size size);
    Index addPoint(const Coord& P, Index index, Real&);
    void update();
    void updateXfree();
};

} //namespace sofa::component::collision::response::mapper
