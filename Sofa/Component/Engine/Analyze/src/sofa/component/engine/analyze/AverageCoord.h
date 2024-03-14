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
#include <sofa/component/engine/analyze/config.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/SingleStateAccessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa::component::engine::analyze
{

/**
 * This class computes the average of a set of Coordinates
 */
template <class DataTypes>
class AverageCoord : public core::DataEngine, public core::behavior::SingleStateAccessor<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(AverageCoord,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef unsigned int Index;
    typedef sofa::type::vector<Index> VecIndex;

protected:

    AverageCoord();

    ~AverageCoord() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    Data<VecIndex> d_indices;    ///< indices of the coordinates to average
    Data<unsigned> d_vecId;  ///< index of the vector (default value corresponds to core::VecCoordId::position() )
    Data<Coord> d_average;       ///< result

    void handleEvent(core::objectmodel::Event *event) override;
    void onBeginAnimationStep(const double /*dt*/);
};

#if !defined(SOFA_COMPONENT_ENGINE_AverageCoord_CPP)
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API AverageCoord<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API AverageCoord<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API AverageCoord<defaulttype::Rigid2Types>;
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API AverageCoord<defaulttype::Rigid3Types>;
 
#endif

} //namespace sofa::component::engine::analyze
