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
#include <sofa/component/engine/generate/config.h>



#include <sofa/core/DataEngine.h>
#include <sofa/type/vector.h>

namespace sofa::component::engine::generate
{

/*
 * This engine join points within a given distance, merging into a new point which is the "average point".
 */

template <class DataTypes>
class JoinPoints : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(JoinPoints,DataTypes),sofa::core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef sofa::type::Vec<3,Real> Vec3;

protected:

    JoinPoints();
    ~JoinPoints() override {}
public:
    void init() override;
    void reinit() override;
    void doUpdate() override;

    //Input
    Data<VecCoord > f_points; ///< Points
    Data<Real> f_distance ; ///< Distance to merge points
    //Output
    Data<VecCoord > f_mergedPoints; ///< Merged Points



private:
    bool getNearestPoint(const typename std::list<Coord>::iterator &itCurrentPoint,
            std::list<Coord>& listPoints,
            std::list<int>& listCoeffs,
            typename std::list<Coord>::iterator &itNearestPoint,
            std::list<int>::iterator &itNearestCoeff,
            const Real& distance);

};

#if !defined(SOFA_COMPONENT_ENGINE_JOINPOINTS_CPP)
extern template class SOFA_COMPONENT_ENGINE_GENERATE_API JoinPoints<sofa::defaulttype::Vec3Types>;
 
#endif

} //namespace sofa::component::engine::generate
