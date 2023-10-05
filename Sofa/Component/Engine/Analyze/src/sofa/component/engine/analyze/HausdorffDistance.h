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
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa::component::engine::analyze
{

/**
 * This class compute the Hausdorff distance of two point clouds
 * \todo: mean and mean square error
 */
template <class DataTypes>
class HausdorffDistance : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(HausdorffDistance,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;

protected:

    HausdorffDistance();

    ~HausdorffDistance() override {}

    void handleEvent(core::objectmodel::Event *event) override;
    void onBeginAnimationStep(const double /*dt*/);

public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    //Input
    Data<VecCoord> f_points_1; ///< Points belonging to the first point cloud
    Data<VecCoord> f_points_2; ///< Points belonging to the second point cloud

    //Output
    Data<Real> d12; ///< Distance from point cloud 1 to 2
    Data<Real> d21; ///< Distance from point cloud 2 to 1
    Data<Real> max; ///< Symmetrical Hausdorff distance

    Data<bool> f_update; ///< Recompute every time step

protected:

    void computeDistances();

    Real distance(Coord p, VecCoord S);

};

#if !defined(SOFA_COMPONENT_ENGINE_HAUSDORFFDISTANCE_CPP)
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API HausdorffDistance<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API HausdorffDistance<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API HausdorffDistance<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API HausdorffDistance<defaulttype::Rigid2Types>;
extern template class SOFA_COMPONENT_ENGINE_ANALYZE_API HausdorffDistance<defaulttype::Rigid3Types>;
 
#endif

} //namespace sofa::component::engine::analyze
