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

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace cgal
{

///
/// \brief The UpsamplePointCloud class generates a denser point cloud from an input point cloud
/// More info here: https://doc.cgal.org/latest/Point_set_processing_3/index.html#Point_set_processing_3Upsampling
///
class UpsamplePointCloud : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(UpsamplePointCloud,sofa::core::DataEngine);

    typedef typename sofa::defaulttype::Vec3Types::Real Real;
    typedef typename sofa::defaulttype::Vec3Types::Coord Point;
    typedef typename sofa::defaulttype::Vec3Types::Coord Coord;
    typedef typename sofa::defaulttype::Vec3Types::VecCoord VecCoord;

    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;

public:
    UpsamplePointCloud();
    virtual ~UpsamplePointCloud() { };

    void init() override;
    void doUpdate() override;

    //Inputs
    sofa::core::objectmodel::Data<VecCoord> d_positionsIn; ///< Input point cloud positions
    sofa::core::objectmodel::Data<VecCoord> d_normalsIn; ///< Input point cloud normals

    //Outputs
    sofa::core::objectmodel::Data<VecCoord> d_positionsOut; ///< Output denser point cloud positions
    sofa::core::objectmodel::Data<VecCoord> d_normalsOut; ///< Output normals of denser point cloud positions
};

} //cgal

