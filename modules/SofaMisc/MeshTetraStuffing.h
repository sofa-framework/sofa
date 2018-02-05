/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MISC_MESHTETRASTUFFING_H
#define SOFA_COMPONENT_MISC_MESHTETRASTUFFING_H
#include <SofaMisc/config.h>

#include <string>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/helper/map.h>

namespace sofa
{

namespace component
{

namespace misc
{

/**
 *  \brief Create a tetrahedral volume mesh from a surface, using the algorithm from F. Labelle and J.R. Shewchuk, "Isosurface Stuffing: Fast Tetrahedral Meshes with Good Dihedral Angles", SIGGRAPH 2007.
 *
 */

class SOFA_MISC_API MeshTetraStuffing : public core::objectmodel::BaseObject
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

    virtual ~MeshTetraStuffing();

public:

    virtual void init() override;

    virtual void draw(const core::visual::VisualParams* vparams) override;

    Data< helper::fixed_array<Point,2> > vbbox;
    Data< Real > size;
    Data<SeqPoints> inputPoints;
    Data<SeqTriangles> inputTriangles;
    Data<SeqQuads> inputQuads;
    Data<SeqPoints> outputPoints;
    Data<SeqTetrahedra> outputTetrahedra;

    Data< Real > alphaLong;
    Data< Real > alphaShort;
    Data< bool > bSnapPoints;
    Data< bool > bSplitTetrahedra;
    Data< bool > bDraw;

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

    helper::vector<int> pInside;
    helper::vector< helper::fixed_array<Real,EDGESHELL> > eBDist;
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

}

} // namespace component

} // namespace sofa

#endif
