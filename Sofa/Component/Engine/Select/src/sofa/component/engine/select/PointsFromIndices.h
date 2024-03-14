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
#include <sofa/component/engine/select/config.h>



#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::engine::select
{

/**
 * This class returns the points given a list of indices.
 */
template <class DataTypes>
class PointsFromIndices : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PointsFromIndices,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;

protected:

    PointsFromIndices();

    ~PointsFromIndices() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    Data<VecCoord> f_X; ///< Position coordinates of the degrees of freedom
    Data<SetIndex> f_indices; ///< Indices of the points
    Data<VecCoord> f_indices_position; ///< Coordinates of the points contained in indices

private:
    bool contains(VecCoord& v, Coord c);
};

#if !defined(SOFA_COMPONENT_ENGINE_POINTSFROMINDICES_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API PointsFromIndices<defaulttype::Vec3Types>;
 
#endif

} //namespace sofa::component::engine::select
