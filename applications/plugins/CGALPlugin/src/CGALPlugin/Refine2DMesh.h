/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace cgal
{

    using namespace sofa::core::objectmodel;

template <class DataTypes>
class Refine2DMesh : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(Refine2DMesh,DataTypes),sofa::core::DataEngine);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef sofa::helper::vector<Real> VecReal;

    typedef sofa::core::topology::BaseMeshTopology::PointID PointID;
    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::topology::BaseMeshTopology::Tetrahedron Tetra;

    typedef sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;


public:
	Refine2DMesh();
	virtual ~Refine2DMesh() { };

    void init() override;
    void reinit() override;

    void doUpdate() override;
    void draw(const sofa::core::visual::VisualParams*) override;


    //Inputs
    Data<VecCoord> d_points;
    Data<SeqEdges> d_edges;
    Data<VecReal> d_edgesData1;
    Data<VecReal> d_edgesData2;
    Data<VecCoord> d_seedPoints;
    Data<VecCoord> d_regionPoints;
    Data<bool> d_useInteriorPoints;

    //Outputs
    Data<VecCoord> d_newPoints;
    Data<SeqTriangles> d_newTriangles;
    Data<SeqEdges> d_newEdges;
    Data<VecReal> d_newEdgesData1;
    Data<VecReal> d_newEdgesData2;
	Data<sofa::helper::vector<int> > d_trianglesRegion;
    Data<sofa::helper::vector<PointID> > d_newBdPoints;

    //Parameters
	Data<double> p_shapeCriteria, p_sizeCriteria;
	Data<bool> p_viewSeedPoints;
	Data<bool> p_viewRegionPoints;


};

#if !defined(CGALPLUGIN_REFINE2DMESH_CPP)
template class SOFA_CGALPLUGIN_API Refine2DMesh<sofa::defaulttype::Vec3Types>;
#endif

} //cgal
