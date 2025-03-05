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

#include <sofa/component/engine/transform/DilateEngine.h>
#include <sofa/helper/TriangleOctree.h>
#include <sofa/helper/rmath.h> //M_PI
#include <sofa/helper/logging/Messaging.h>

namespace sofa::component::engine::transform
{

using helper::ReadAccessor;
using helper::WriteOnlyAccessor;
using helper::TriangleOctreeRoot;
using type::vector;

template <class DataTypes>
DilateEngine<DataTypes>::DilateEngine()
    : d_inputX ( initData (&d_inputX, "input_position", "input array of 3d points") )
    , d_outputX( initData (&d_outputX, "output_position", "output array of 3d points") )
    , d_triangles( initData (&d_triangles, "triangles", "input mesh triangles") )
    , d_quads( initData (&d_quads, "quads", "input mesh quads") )
    , d_normals( initData (&d_normals, "normal", "point normals") )
    , d_thickness( initData (&d_thickness, "thickness", "point thickness") )
    , d_distance( initData (&d_distance, (Real)0, "distance", "distance to move the points (positive for dilatation, negative for erosion)") )
    , d_minThickness( initData (&d_minThickness, (Real)0, "minThickness", "minimal thickness to enforce") )
{
    addAlias(&d_inputX,"position");

    addInput(&d_inputX);
    addInput(&d_triangles);
    addInput(&d_quads);
    addInput(&d_distance);
    addInput(&d_minThickness);
    addOutput(&d_outputX);
    addOutput(&d_normals);
    addOutput(&d_thickness);
}


template <class DataTypes>
void DilateEngine<DataTypes>::init()
{
    setDirtyValue();
}


template <class DataTypes>
void DilateEngine<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void DilateEngine<DataTypes>::doUpdate()
{
    ReadAccessor<Data<VecCoord> > in = d_inputX;
    const ReadAccessor<Data<SeqTriangles> > triangles = d_triangles;
    const ReadAccessor<Data<SeqQuads> > quads = d_quads;
    const Real distance = d_distance.getValue();
    const Real minThickness = d_minThickness.getValue();

    WriteOnlyAccessor<Data<VecCoord> > out = d_outputX;
    WriteOnlyAccessor<Data<VecCoord> > normals = d_normals;

    const int nbp = in.size();
    const int nbt = triangles.size();
    const int nbq = quads.size();

    if(nbp == 0)
    {
        msg_warning() << "No input position";
        d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
        return;
    }

    if(nbt == 0 && nbq == 0)
    {
        msg_warning() << "No input mesh (neither triangle or quad)";
        d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
        return;
    }

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
        TriangleOctreeRoot octree;
        SeqTriangles alltri = triangles.ref();
        for(int i=0; i<nbq; ++i)
        {
            alltri.push_back(Triangle(quads[i][0], quads[i][1], quads[i][2]));
            alltri.push_back(Triangle(quads[i][0], quads[i][2], quads[i][3]));
        }
        WriteOnlyAccessor<Data<vector<Real> > > thickness = d_thickness;
        thickness.resize(nbp);
        octree.buildOctree(&alltri, &(in.ref()));
        for (int ip=0; ip<nbp; ++ip)
        {
            Coord origin = in[ip];
            Coord direction = -normals[ip];
            Real mindist = -1.0f;
            vector< helper::TriangleOctree::traceResult > results;
            octree.octreeRoot->traceAll(origin, direction, results);
            for (unsigned int i=0; i<results.size(); ++i)
            {
                const int t = results[i].tid;
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
    ReadAccessor<Data<vector<Real> > > thickness = d_thickness;
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


} //namespace sofa::component::engine::transform
