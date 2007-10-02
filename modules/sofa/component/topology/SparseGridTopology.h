/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_SPARSEGRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_SPARSEGRIDTOPOLOGY_H

#include <sofa/component/topology/MeshTopology.h>
#include <sofa/defaulttype/Vec.h>
#include "RegularGridTopology.h"


namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


/** A sparse grid topology. Like a sparse FFD building from the bounding box of the object. Starting from a RegularGrid, only valid cells containing matter (ie intersecting the original surface mesh or totally inside the object) are considered.
Valid cells are tagged by a Type BOUNDARY or INSIDE
 */
class SparseGridTopology : public MeshTopology
{
public:

    typedef Vec3d Vec3;
    typedef fixed_array<Vec3d,8> CubeCorners;
    typedef enum {OUTSIDE,INSIDE,BOUNDARY} Type;



    SparseGridTopology();

    SparseGridTopology(int nx, int ny, int nz);


    bool load(const char* filename);
    void init();


    int getNx() const { return nx.getValue(); }
    int getNy() const { return ny.getValue(); }
    int getNz() const { return nz.getValue(); }

    void setNx(int n) { nx.setValue(n); }
    void setNy(int n) { ny.setValue(n); }
    void setNz(int n) { nz.setValue(n); }

    int getNbCubes();
    int getNbQuads();

    bool hasPos()  const { return true; }

protected:

    DataField<int> nx;
    DataField<int> ny;
    DataField<int> nz;

    /// bounding box positions, by default the real bounding box is used
    DataField<double> xmin;
    DataField<double> ymin;
    DataField<double> zmin;
    DataField<double> xmax;
    DataField<double> ymax;
    DataField<double> zmax;

    DataField< std::string > filename;  ///< Mesh used to initialize filled portion of the grid

    void updateLines();
    void updateQuads();
    void updateCubes();





    std::vector<Type> _types; ///< BOUNDARY or FULL filled cells
    /// start from a seed cell (i,j,k) the OUTSIDE filling is propagated to neighboor cells until meet a BOUNDARY cell (this function is called from all border cells of the RegularGrid)
    void propagateFrom( const int i, const int j, const int k,  RegularGridTopology& regularGrid, vector<Type>& regularGridTypes, vector<bool>& alreadyTested  );

    /// to compute valid cubes (intersection between mesh segments and cubes)
    typedef struct segmentForIntersection
    {
        Vec3 center;
        Vec3 dir;
        float norm;
        segmentForIntersection(const Vec3& s0, const Vec3& s1)
        {
            center = (s0+s1)*.5;
            dir = center-s0;
            norm = dir.norm();
            dir /= norm;
        };
    } SegmentForIntersection;
    struct ltSegmentForIntersection // for set of SegmentForIntersection
    {
        bool operator()(const SegmentForIntersection& s0, const SegmentForIntersection& s1) const
        {
            return s0.center < s1.center || s0.norm < s1.norm;
        }
    };
    typedef struct cubeForIntersection
    {
        Vec3 center;
        fixed_array<Vec3,3> dir;
        Vec3 norm;
        cubeForIntersection( const CubeCorners&  corners )
        {
            center = (corners[7] + corners[0]) * .5;

            norm[0] = (center[0] - corners[0][0]);
            dir[0] = Vec3(1,0,0);

            norm[1] = (center[1] - corners[0][1]);
            dir[1] = Vec3(0,1,0);

            norm[2] = (center[2] - corners[0][2]);
            dir[2] = Vec3(0,0,1);
        }
    } CubeForIntersection;
    /// return true if there is an intersection between a SegmentForIntersection and a CubeForIntersection
    bool SparseGridTopology::intersectionSegmentBox( const SegmentForIntersection& seg, const CubeForIntersection& cube  );


};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
