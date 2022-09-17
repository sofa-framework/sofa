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

#include <sofa/core/DataEngine.h>

namespace sofa::component::engine::select
{

/**
 * This class find the point at a given distance from a set of points
 */
template <class DataTypes>
class ProximityROI : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ProximityROI,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::CPos CPos;
    typedef type::Vec<3,Real> Vec3;
    typedef type::Vec<6,Real> Vec6;
    typedef sofa::topology::SetIndex SetIndex;

    typedef unsigned int PointID;

protected:
    ProximityROI();
    ~ProximityROI() override {}

public:
    void init() override;
    void reinit() override;
    void doUpdate() override;
    void draw(const core::visual::VisualParams* vparams) override;

public:
    //Input
    Data< type::vector<Vec3> > centers; ///< Center(s) of the sphere(s)
    Data< type::vector<Real> > radii; ///< Radius(i) of the sphere(s)
    Data<unsigned int> f_num; ///< Maximum number of points to select
    Data<VecCoord> f_X0; ///< Rest position coordinates of the degrees of freedom

    //Output
    Data<SetIndex> f_indices; ///< Indices of the points contained in the ROI
    Data<VecCoord > f_pointsInROI; ///< Points contained in the ROI
    Data<type::vector<Real>> f_distanceInROI; ///< distance between the points contained in the ROI and the closest center.

    Data<SetIndex> f_indicesOut; ///< Indices of the points not contained in the ROI

    //Parameter
    Data<bool> p_drawSphere; ///< Draw shpere(s)
    Data<bool> p_drawPoints; ///< Draw Points
    Data<double> _drawSize; ///< rendering size for box and topological elements
};

#if  !defined(SOFA_COMPONENT_ENGINE_PROXIMITYROI_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API ProximityROI<defaulttype::Vec3Types>;
#endif

} //namespace sofa::component::engine::select
