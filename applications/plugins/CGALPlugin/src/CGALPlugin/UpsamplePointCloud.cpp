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

#include <CGALPlugin/UpsamplePointCloud.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/edge_aware_upsample_point_set.h>

#include <sofa/core/ObjectFactory.h>


int UpsamplePointCloudClass = sofa::core::RegisterObject("Generates a denser point cloud from an input point cloud")
        .add< cgal::UpsamplePointCloud >()
        ;


namespace cgal
{

using sofa::helper::ReadAccessor;
using sofa::helper::WriteAccessor;
using sofa::Data;
using sofa::core::objectmodel::ComponentState ;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef std::pair<Point_3, Vector_3> Pwn;
typedef CGAL::Parallel_tag Concurrency_tag;


UpsamplePointCloud::UpsamplePointCloud()
    : d_positionsIn(initData (&d_positionsIn, "position", "Input point cloud positions"))
    , d_normalsIn(initData (&d_normalsIn, "normals", "Input point cloud normals"))
    , d_positionsOut(initData (&d_positionsOut, "outputPosition", "Output denser point cloud positions"))
    , d_normalsOut(initData (&d_normalsOut, "outputNormals", "Output normals of denser point cloud"))
{
    addInput(&d_positionsIn);
    addInput(&d_normalsIn);
    addOutput(&d_positionsOut);
    addOutput(&d_normalsOut);

    setDirtyValue();
}


void UpsamplePointCloud::init()
{
    d_componentState.setValue(ComponentState::Invalid);

    if(d_positionsIn.getValue().empty()){
        msg_error() << "No input positions. The component is disabled.";
        return;
    }

    if(d_normalsIn.getValue().empty()){
        msg_error() << "No input normals. The component is disabled.";
        return;
    }

    d_componentState.setValue(ComponentState::Valid);
}


void UpsamplePointCloud::doUpdate()
{
    if(d_componentState.getValue() == ComponentState::Invalid)
        return;

    ReadAccessor< Data<VecCoord> > positionsIn = d_positionsIn;
    ReadAccessor< Data<VecCoord> > normalsIn = d_normalsIn;
    WriteAccessor< Data<VecCoord> > positionsOut = d_positionsOut;
    WriteAccessor< Data<VecCoord> > normalsOut = d_normalsOut;

    std::vector<Pwn> points;
    auto pointMap = CGAL::First_of_pair_property_map<Pwn>();
    auto normalMap = CGAL::Second_of_pair_property_map<Pwn>();

    for (sofa::Index i = 0; i<positionsIn.size() ; i++)
    {
        Point_3 p( positionsIn[i][0], positionsIn[i][1], positionsIn[i][2] );
        Vector_3 n( normalsIn[i][0], normalsIn[i][1], normalsIn[i][2] );
        points.push_back(std::pair(p,n));
        put(pointMap, *points.begin(), p);
        put(normalMap, *points.begin(), n);
    }

    const double sharpness_angle = 25;   // control sharpness of the result.
    const double edge_sensitivity = 0;    // higher values will sample more points near the edges
    const double neighbor_radius = 0.25;  // initial size of neighborhood.
    const std::size_t number_of_output_points = points.size() * 4;

    CGAL::edge_aware_upsample_point_set<Concurrency_tag>(points,
                                                      std::back_inserter(points),
                                                      CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()).
                                                      normal_map(CGAL::Second_of_pair_property_map<Pwn>()).
                                                      sharpness_angle(sharpness_angle).
                                                      edge_sensitivity(edge_sensitivity).
                                                      neighbor_radius(neighbor_radius).
                                                      number_of_output_points(number_of_output_points));

    positionsOut.clear();
    normalsOut.clear();
    for (auto pos: points)
    {
        Point p;
        p[0] = CGAL::to_double(pos.first.x());
        p[1] = CGAL::to_double(pos.first.y());
        p[2] = CGAL::to_double(pos.first.z());

        positionsOut.push_back(p);

        Point n;
        n[0] = CGAL::to_double(pos.second.x());
        n[1] = CGAL::to_double(pos.second.y());
        n[2] = CGAL::to_double(pos.second.z());

        normalsOut.push_back(n);
    }
}

} //cgal
 
