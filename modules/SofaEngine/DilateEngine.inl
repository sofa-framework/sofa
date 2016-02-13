/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ENGINE_DILATEENGINE_INL
#define SOFA_COMPONENT_ENGINE_DILATEENGINE_INL

#include <SofaEngine/DilateEngine.h>
#include <SofaMeshCollision/TriangleOctree.h>
#include <SofaMeshCollision/RayTriangleIntersection.h>
#include <sofa/helper/rmath.h> //M_PI

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
DilateEngine<DataTypes>::DilateEngine()
    : f_inputX ( initData (&f_inputX, "input_position", "input array of 3d points") )
    , f_outputX( initData (&f_outputX, "output_position", "output array of 3d points") )
    , f_triangles( initData (&f_triangles, "triangles", "input mesh triangles") )
    , f_quads( initData (&f_quads, "quads", "input mesh quads") )
    , f_normals( initData (&f_normals, "normal", "point normals") )
    , f_thickness( initData (&f_thickness, "thickness", "point thickness") )
    , f_distance( initData (&f_distance, (Real)0, "distance", "distance to move the points (positive for dilatation, negative for erosion)") )
    , f_minThickness( initData (&f_minThickness, (Real)0, "minThickness", "minimal thickness to enforce") )
{
    addAlias(&f_inputX,"position");
}


template <class DataTypes>
void DilateEngine<DataTypes>::init()
{
    addInput(&f_inputX);
    addInput(&f_triangles);
    addInput(&f_quads);
    addInput(&f_distance);
    addInput(&f_minThickness);
    addOutput(&f_outputX);
    addOutput(&f_normals);
    addOutput(&f_thickness);
    setDirtyValue();
}

template <class DataTypes>
void DilateEngine<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void DilateEngine<DataTypes>::update()
{
    helper::ReadAccessor<Data<VecCoord> > in = f_inputX;
    helper::ReadAccessor<Data<SeqTriangles> > triangles = f_triangles;
    helper::ReadAccessor<Data<SeqQuads> > quads = f_quads;    
    const Real distance = f_distance.getValue();
    const Real minThickness = f_minThickness.getValue();

    cleanDirty();

    helper::WriteOnlyAccessor<Data<VecCoord> > out = f_outputX;

    const int nbp = in.size();
    const int nbt = triangles.size();
    const int nbq = quads.size();

    helper::WriteOnlyAccessor<Data<VecCoord> > normals = f_normals;
    normals.resize(nbp);
    for (int i=0; i<nbp; ++i)
        normals[i].clear();
    for (int i=0; i<nbt; ++i)
    {
        Coord n = cross(in[triangles[i][1]] - in[triangles[i][0]], in[triangles[i][2]] - in[triangles[i][0]]);
        normals[triangles[i][0]] += n;
        normals[triangles[i][1]] += n;
        normals[triangles[i][2]] += n;
    }
    for (int i=0; i<nbq; ++i)
    {
        normals[quads[i][0]] += cross(in[quads[i][1]] - in[quads[i][0]], in[quads[i][3]] - in[quads[i][0]]);
        normals[quads[i][1]] += cross(in[quads[i][2]] - in[quads[i][1]], in[quads[i][0]] - in[quads[i][1]]);
        normals[quads[i][2]] += cross(in[quads[i][3]] - in[quads[i][2]], in[quads[i][1]] - in[quads[i][2]]);
        normals[quads[i][3]] += cross(in[quads[i][0]] - in[quads[i][3]], in[quads[i][2]] - in[quads[i][3]]);
    }
    for (int i=0; i<nbp; ++i)
        normals[i].normalize();

    if (minThickness != 0)
    {
        collision::TriangleOctreeRoot octree;
        SeqTriangles alltri = triangles.ref();
        for(int i=0; i<nbq; ++i)
        {
            alltri.push_back(Triangle(quads[i][0], quads[i][1], quads[i][2]));
            alltri.push_back(Triangle(quads[i][0], quads[i][2], quads[i][3]));
        }
        helper::WriteOnlyAccessor<Data<helper::vector<Real> > > thickness = f_thickness;
        thickness.resize(nbp);
        octree.buildOctree(&alltri, &(in.ref()));
        for (int ip=0; ip<nbp; ++ip)
        {
            Coord origin = in[ip];
            Coord direction = -normals[ip];
            Real mindist = -1.0f;
            helper::vector< collision::TriangleOctree::traceResult > results;
            octree.octreeRoot->traceAll(origin, direction, results);
            for (unsigned int i=0; i<results.size(); ++i)
            {
                int t = results[i].tid;
                if ((int)alltri[t][0] == ip || (int)alltri[t][1] == ip || (int)alltri[t][2] == ip) continue;
                Real dist = results[i].t;
                if (dist > 0 && (dist < mindist || mindist < 0))
                    mindist = dist;
            }
            if (mindist < 0) mindist = 0;
            thickness[ip] = mindist;
        }
    }

    //Set Output
    out.resize(nbp);
    helper::ReadAccessor<Data<helper::vector<Real> > > thickness = f_thickness;
    for (int i=0; i<nbp; ++i)
    {
        Real d = distance;
        if (minThickness > 0)
        {
            Real t = thickness[i];
            if (t < minThickness)
                d += (minThickness-t) * 0.5f;
        }
        out[i] = in[i] + normals[i] * d;
    }

}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
