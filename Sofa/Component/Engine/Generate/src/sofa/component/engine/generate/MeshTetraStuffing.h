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

#include <string>
#include <sofa/type/fixed_array.h>
#include <sofa/type/vector.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/helper/map.h>

namespace sofa::component::engine::generate
{

/**
 *  \brief Create a tetrahedral volume mesh from a surface, using the algorithm from F. Labelle and J.R. Shewchuk, "Isosurface Stuffing: Fast Tetrahedral Meshes with Good Dihedral Angles", SIGGRAPH 2007.
 *
 */

class SOFA_COMPONENT_ENGINE_GENERATE_API MeshTetraStuffing : public core::DataEngine
{
public:
    SOFA_CLASS(MeshTetraStuffing,core::objectmodel::BaseObject);

    typedef defaulttype::Vec3Types::Real Real;
    typedef defaulttype::Vec3Types::Coord Point;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::Tetra Tetra;

    typedef defaulttype::Vec3Types::VecCoord SeqPoints;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef sofa::core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;

protected:
    MeshTetraStuffing();

    ~MeshTetraStuffing() override;

public:

    void init() override;

    void draw(const core::visual::VisualParams* vparams) override;

    void doUpdate() override;


    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data< type::fixed_array<Point,2> > vbbox;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data<SeqPoints> inputPoints;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data<SeqTriangles> inputTriangles;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data<SeqQuads> inputQuads;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data<SeqPoints> outputPoints;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data<SeqTetrahedra> outputTetrahedra;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data< Real > alphaLong;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data<Real> alphaShort;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data<bool> bSnapPoints;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data< bool > bSplitTetrahedra;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data< bool > bDraw;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_GENERATE()
    Data< Real > size;



    Data< type::fixed_array<Point,2> > d_vbbox; ///< BBox to restrict the volume to
    Data< Real > d_size; ///< Size of the generate tetrahedra. If negative, number of grid cells in the largest bbox dimension
    Data<SeqPoints> d_inputPoints; ///< Input surface mesh points
    Data<SeqTriangles> d_inputTriangles; ///< Input surface mesh triangles
    Data<SeqQuads> d_inputQuads; ///< Input surface mesh quads
    Data<SeqPoints> d_outputPoints; ///< Output volume mesh points
    Data<SeqTetrahedra> d_outputTetrahedra; ///< Output volume mesh tetrahedra

    Data< Real > d_alphaLong; ///< Minimum alpha values on long edges when snapping points
    Data< Real > d_alphaShort; ///< Minimum alpha values on short edges when snapping points
    Data< bool > d_bSnapPoints; ///< Snap points to the surface if intersections on edges are closed to given alpha values
    Data< bool > d_bSplitTetrahedra; ///< Split tetrahedra crossing the surface
    Data< bool > d_bDraw; ///< Activate rendering of internal datasets

    Real cellsize;
    int gsize[3];
    Point g0;
    int ph0;
    int hsize[3];
    Point h0;

    enum { EDGESHELL = (6+8) };
    int getEdgePoint2(int p, int e);
    int getEdgeSize2(int e);
    Point getEdgeDir(int e);

    type::vector<int> pInside;
    type::vector< type::fixed_array<Real,EDGESHELL> > eBDist;
    std::map<std::pair<int,int>, int> splitPoints;

    SeqPoints rays;
    SeqPoints intersections;
    SeqPoints insides;
    SeqPoints snaps;
    SeqPoints diags;

    void addTetra(SeqTetrahedra& outT, SeqPoints& outP, int p1, int p2, int p3, int p4, int line=0);
    void addFinalTetra(SeqTetrahedra& outT, SeqPoints& outP, int p1, int p2, int p3, int p4, bool flip=false, int line=0);
    int getSplitPoint(int from, int to);

    /// Should the diagonal of abcd should be bd instead of ac ?
    bool flipDiag(const SeqPoints& outP, int a, int b, int c, int d, int e=-1);

    /// Is the two given vertices order flipped
    bool needFlip(int p1, int p2, int p3, int p4, int q1, int q2, int q3, int q4);

};

} // namespace sofa::component::engine::generate
