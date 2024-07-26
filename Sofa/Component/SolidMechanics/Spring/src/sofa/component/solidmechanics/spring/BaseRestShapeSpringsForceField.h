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

#include <sofa/component/solidmechanics/spring/config.h>

#include <sofa/type/RGBAColor.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/type/vector.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>

namespace sofa::component::solidmechanics::spring {

template<class DataTypes>
class BaseRestShapeSpringsForceField : public core::behavior::ForceField<DataTypes>
{
    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef type::vector< sofa::Index > VecIndex;
    typedef sofa::core::topology::TopologySubsetIndices DataSubsetIndex;
    typedef type::vector< Real >	 VecReal;
public:

    SOFA_CLASS(SOFA_TEMPLATE(BaseRestShapeSpringsForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;


    SetIndex d_indices;
    Data< VecReal > d_stiffness; ///< stiffness values between the actual position and the rest shape position
    Data< VecReal > d_angularStiffness; ///< angularStiffness assigned when controlling the rotation of the points
    Data< bool > d_drawSpring; ///< draw Spring
    Data< sofa::type::RGBAColor > d_springColor; ///< spring color. (default=[0.0,1.0,0.0,1.0])

    /// Link to be set to the topology container in the component graph.
    SingleLink<BaseRestShapeSpringsForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    BaseRestShapeSpringsForceField();

    virtual void init() override;

    virtual bool checkOutOfBoundsIndices() = 0;



};

}

