/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/misc/MeshTetraStuffing.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/DrawManager.h>
#include <sofa/component/collision/TriangleOctree.h>
#include <sofa/component/collision/RayTriangleIntersection.h>

#include <iostream>
#include <algorithm>

#define USE_OCTREE

namespace sofa
{

namespace component
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MeshTetraStuffing)

int MeshLoaderClass = core::RegisterObject("Create a regular tetrahedra mesh inside a surface mesh.")
        .add< MeshTetraStuffing >()
        ;

MeshTetraStuffing::MeshTetraStuffing()
    : bbox(initData(&bbox,"bbox","BBox to restrict the volume to"))
    , size(initData(&size,(Real)-8.0,"size","Size of the generate tetrahedra. If negative, number of grid cells in the largest bbox dimension"))
    , inputPoints(initData(&inputPoints,"inputPoints","Input surface mesh points"))
    , inputTriangles(initData(&inputTriangles,"inputTriangles","Input surface mesh triangles"))
    , outputPoints(initData(&outputPoints,"outputPoints","Output volume mesh points"))
    , outputTetras(initData(&outputTetras,"outputTetras","Output volume mesh tetras"))
{
}

MeshTetraStuffing::~MeshTetraStuffing()
{
}

void MeshTetraStuffing::init()
{
    const SeqPoints& inP = inputPoints.getValue();
    const SeqTriangles& inT = inputTriangles.getValue();
    if (inP.empty() || inT.empty())
    {
        serr << "Empty input mesh. Use data dependency to link them to a loaded Topology or MeshLoader" << sendl;
        return;
    }
    SeqPoints inTN;
    helper::fixed_array<Point,2> inputBBox;
    {
        inputBBox[0] = inP[0];
        inputBBox[1] = inP[0];
        int nbp = inP.size();
        int nbt = inT.size();
        for (int p=1; p<nbp; ++p)
        {
            for (int c=0; c<3; ++c)
                if (inP[p][c] < inputBBox[0][c]) inputBBox[0][c] = inP[p][c];
                else if (inP[p][c] > inputBBox[1][c]) inputBBox[1][c] = inP[p][c];
        }
        inTN.resize(nbt);
        for (int t=0; t<nbt; ++t)
        {
            inTN[t] = (inP[inT[t][1]]-inP[inT[t][0]]).cross(inP[inT[t][2]]-inP[inT[t][0]]);
        }
    }
    helper::fixed_array<Point,2>& bb = *bbox.beginEdit();
    if (bb[0][0] >= bb[1][0])
    {
        bb = inputBBox;
    }
    bbox.endEdit();

    cellsize = size.getValue();
    if (cellsize < 0)
    {
        Point b = bb[1]-bb[0];
        Real bsize = b[0];
        if (b[1]>bsize) bsize = b[1];
        if (b[2]>bsize) bsize = b[2];
        cellsize = bsize / -cellsize;
    }

    sout << "bbox = " << bb << " cell size = " << cellsize << sendl;

#ifdef USE_OCTREE
    collision::TriangleOctreeRoot octree;
    octree.buildOctree(&inT, &inP);
#else
    collision::RayTriangleIntersection raytri;
#endif

    Point p0 = (bb[0] + bb[1])/2;

    int c0[3], c1[3];
    for (int c=0; c<3; ++c)
    {
        c0[c] = - (int((p0[c] - bb[0][c]) / cellsize) + 1);
        c1[c] =   (int((bb[1][c] - p0[c]) / cellsize) + 1);
        gsize[c] = c1[c]-c0[c]+1;
        g0[c] = p0[c] + c0[c]*cellsize;
        hsize[c] = gsize[c]+1;
        h0[c] = g0[c] - cellsize/2;
    }

    sout << "Grid <"<<c0[0]<<","<<c0[1]<<","<<c0[2]<<">-<"<<c1[0]<<","<<c1[1]<<","<<c1[2]<<">" << sendl;

    SeqPoints& outP = *outputPoints.beginEdit();
    SeqTetras& outT = *outputTetras.beginEdit();

    outP.resize(gsize[0]*gsize[1]*gsize[2] + hsize[0]*hsize[1]*hsize[2]);

    ph0 = gsize[0]*gsize[1]*gsize[2];
    int p = 0;
    for (int z=0; z<gsize[2]; ++z)
        for (int y=0; y<gsize[1]; ++y)
            for (int x=0; x<gsize[0]; ++x,++p)
                outP[p] = g0 + Point(x*cellsize,y*cellsize,z*cellsize);
    p = ph0;
    for (int z=0; z<hsize[2]; ++z)
        for (int y=0; y<hsize[1]; ++y)
            for (int x=0; x<hsize[0]; ++x,++p)
                outP[p] = h0 + Point(x*cellsize,y*cellsize,z*cellsize);

    int nbp = outP.size();

    pInside.resize(nbp);
    eBDist.resize(nbp);
    for (int p=0; p<nbp; ++p)
    {
        // find edges where we are the first point
        for (int e=0; e<EDGESHELL; e+=2)
        {
            int p1 = getEdgePoint2(p,e);
            int p2 = getEdgePoint2(p,e+1);
            if (p1 != -1 && p2 == -1)
            {
                Point origin = outP[p];
                Point direction = outP[p1] - origin; //getEdgeDir(e);
                //rays.push_back(origin);
                //rays.push_back(origin+direction);
                helper::vector< collision::TriangleOctree::traceResult > results;
#ifdef USE_OCTREE
                octree.octreeRoot->traceAll(origin, direction, results);
#else
                for (unsigned int t=0; t<inT.size(); ++t)
                {
                    collision::TriangleOctree::traceResult r;
                    r.tid = t;
                    if (raytri.NewComputation(inP[inT[t][0]],inP[inT[t][1]],inP[inT[t][2]],origin,direction,r.t,r.u,r.v))
                        results.push_back(r);
                }
#endif
                if (!results.empty())
                {
                    //std::cout << "Point " << p << " edge " << e << " : " << results.size() << " intersections." << std::endl;
                    std::sort(results.begin(), results.end());
                    int n = results.size();
                    for (int i=0; i<n; ++i)
                    {
                        results[i].tid = dot(inTN[results[i].tid],direction) < 0 ? -1 : 1;
                        intersections.push_back(origin + direction * results[i].t);
                    }
                    for (int i=0; i<n-1; i++)
                    {
                        if (results[i].tid == -1 && results[i+1].tid == 1)
                        {
                            rays.push_back(origin+direction*results[i].t);
                            rays.push_back(origin+direction*results[i+1].t);
                        }
                    }
                    int d = 0;
                    int i = 0;
                    p1 = p;
                    while (p1 != -1)
                    {
                        while (i < n && results[i].t < d)
                            ++i;
                        // update inside point flags
                        if (i >= n)
                            pInside[p1] -= results[n-1].tid;
                        else if (i == 0)
                            pInside[p1] += results[0].tid;
                        else if ((d - results[i-1].t) < (results[i].t - d))
                            pInside[p1] += results[i].tid;
                        else
                            pInside[p1] -= results[i-1].tid;

                        // update edge border distances
                        if (i < n)
                        {
                            eBDist[p1][e] = results[i].t - d;
                        }
                        if (i > 0)
                        {
                            eBDist[p1][e+1] = d - results[i-1].t;
                        }

                        p1 = getEdgePoint2(p1,e);
                        ++d;
                    }
                }
            }
        }
    }

    for (int p=0; p<nbp; ++p)
    {
        if (pInside[p] == 0) --pInside[p]; // by default uncertain points are outside
        if (pInside[p] > 0) insides.push_back(outP[p]);
    }

    // Create tetrahedra inside or crossing the mesh
    const int gsize01 = gsize[0]*gsize[1];
    const int hsize01 = hsize[0]*hsize[1];
    for (int p=0, ph=ph0, z=0; z<gsize[2]; ++z,ph+=hsize[0])
        for (int y=0; y<gsize[1]; ++y,++ph)
            for (int x=0; x<gsize[0]; ++x,++p,++ph)
            {
                if (x > 0)
                {
                    // edge in X axis
                    int p2 = p - 1;
                    int hshell[4] = {ph, ph + hsize[0], ph + hsize[0] + hsize01, ph + hsize01};
                    for (int i=0; i<4; ++i)
                    {
                        int p3 = hshell[i];
                        int p4 = hshell[(i+1)%4];
                        if (pInside[p]>0 || pInside[p2]>0 || pInside[p3]>0 || pInside[p4]>0)
                            outT.push_back(Tetra(p,p2,p3,p4));
                    }
                }
                if (y > 0)
                {
                    // edge in Y axis
                    int p2 = p - gsize[0];
                    int hshell[4] = {ph, ph + hsize01, ph + 1 + hsize01, ph + 1};
                    for (int i=0; i<4; ++i)
                    {
                        int p3 = hshell[i];
                        int p4 = hshell[(i+1)%4];
                        if (pInside[p]>0 || pInside[p2]>0 || pInside[p3]>0 || pInside[p4]>0)
                            outT.push_back(Tetra(p,p2,p3,p4));
                    }
                }
                if (z > 0)
                {
                    // edge in X axis
                    int p2 = p - gsize01;
                    int hshell[4] = {ph, ph + 1, ph + 1 + hsize[0], ph + hsize[0]};
                    for (int i=0; i<4; ++i)
                    {
                        int p3 = hshell[i];
                        int p4 = hshell[(i+1)%4];
                        if (pInside[p]>0 || pInside[p2]>0 || pInside[p3]>0 || pInside[p4]>0)
                            outT.push_back(Tetra(p,p2,p3,p4));
                    }
                }
            }

    // compress output points to remove unused ones
    vector<int> newPid;
    newPid.resize(outP.size());
    for (unsigned int t=0; t<outT.size(); ++t)
        for (int i=0; i<4; ++i)
            newPid[outT[t][i]] = 1;
    nbp = 0;
    for (unsigned int p=0; p<newPid.size(); ++p)
    {
        if (newPid[p] == 0) newPid[p] = -1;
        else
        {
            newPid[p] = nbp++;
            if (newPid[p] != (int)p)
            {
                outP[newPid[p]] = outP[p];
                pInside[newPid[p]] = pInside[p];
            }
        }
    }
    outP.resize(nbp);
    pInside.resize(nbp);
    eBDist.clear();

    for (unsigned int t=0; t<outT.size(); ++t)
        for (int i=0; i<4; ++i)
            outT[t][i] = newPid[outT[t][i]];

    outputPoints.endEdit();
    outputTetras.endEdit();
}

int MeshTetraStuffing::getEdgePoint2(int p, int e)
{
    if (p < ph0)
    {
        int c[3];
        c[0] = p % gsize[0]; p /= gsize[0];
        c[1] = p % gsize[1]; p /= gsize[1];
        c[2] = p; // % gsize[2]; p /= gsize[2];
        if (e < 6)
        {
            // same grid
            if (e&1)
            {
                ++c[e>>1];
                if (c[e>>1] >= gsize[e>>1]) return -1;
            }
            else
            {
                --c[e>>1];
                if (c[e>>1] < 0) return -1;
            }
            return c[0]+gsize[0]*(c[1]+gsize[1]*c[2]);
        }
        else
        {
            e-=6;
            c[0] += (e>>0)&1;
            c[1] += ((e>>1)^e)&1;
            c[2] += ((e>>2)^e)&1;
            return ph0 + c[0]+hsize[0]*(c[1]+hsize[1]*c[2]);
        }
    }
    else
    {
        p -= ph0;
        int c[3];
        c[0] = p % hsize[0]; p /= hsize[0];
        c[1] = p % hsize[1]; p /= hsize[1];
        c[2] = p; // % hsize[2]; p /= hsize[2];
        if (e < 6)
        {
            // same grid
            if (e&1)
            {
                ++c[e>>1];
                if (c[e>>1] >= hsize[e>>1]) return -1;
            }
            else
            {
                --c[e>>1];
                if (c[e>>1] < 0) return -1;
            }
            return ph0 + c[0]+hsize[0]*(c[1]+hsize[1]*c[2]);
        }
        else
        {
            e-=6;
            c[0] += ((e>>0)&1) - 1;
            c[1] += (((e>>1)^e)&1) - 1;
            c[2] += (((e>>2)^e)&1) - 1;
            for (int i=0; i<3; ++i)
                if ((unsigned)c[i] >= (unsigned)gsize[i]) return -1;
            return c[0]+gsize[0]*(c[1]+gsize[1]*c[2]);
        }
    }
}

int MeshTetraStuffing::getEdgeSize2(int e)
{
    if (e < 6) return 4;
    else return 3;
}

MeshTetraStuffing::Point MeshTetraStuffing::getEdgeDir(int e)
{
    Point p;
    if (e < 6)
    {
        // same grid
        if (e&1)
            p[e>>1] += cellsize;
        else
            p[e>>1] -= cellsize;
    }
    else
    {
        p[0] += ((e>>0)&1) ? -cellsize/2 : cellsize/2;
        p[1] += (((e>>1)^e)&1) ? -cellsize/2 : cellsize/2;
        p[2] += (((e>>2)^e)&1) ? -cellsize/2 : cellsize/2;
    }
    return p;
}

void MeshTetraStuffing::draw()
{
    const SeqPoints& inP = inputPoints.getValue();
    const SeqTriangles& inT = inputTriangles.getValue();
    const SeqPoints& outP = outputPoints.getValue();
    const SeqTetras& outT = outputTetras.getValue();

    //simulation::getSimulation()->DrawUtility.drawPoints(inP, 1, Vec<4,float>(1,0,0,1));
    simulation::getSimulation()->DrawUtility.drawPoints(intersections, 2, Vec<4,float>(1,0,0,1));
    //simulation::getSimulation()->DrawUtility.drawPoints(insides, 1, Vec<4,float>(0,1,0,1));
    //simulation::getSimulation()->DrawUtility.drawLines(rays, 1, Vec<4,float>(1,1,0,1));
    simulation::getSimulation()->DrawUtility.drawPoints(outP, 1, Vec<4,float>(0,1,0,1));
}

} // namespace component

} // namespace sofa
