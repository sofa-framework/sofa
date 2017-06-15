/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <SofaMisc/MeshTetraStuffing.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Simulation.h>
#include <SofaGeneralMeshCollision/TriangleOctree.h>
#include <SofaMeshCollision/RayTriangleIntersection.h>

#include <iostream>
#include <algorithm>

#define USE_OCTREE

namespace sofa
{

namespace component
{

namespace misc
{

using namespace sofa::defaulttype;
using helper::vector;

SOFA_DECL_CLASS(MeshTetraStuffing)

int MeshTetraStuffingClass = core::RegisterObject("Create a tetrahedral volume mesh from a surface, using the algorithm from F. Labelle and J.R. Shewchuk, \"Isosurface Stuffing: Fast Tetrahedral Meshes with Good Dihedral Angles\", SIGGRAPH 2007.")
        .add< MeshTetraStuffing >()
        ;

MeshTetraStuffing::MeshTetraStuffing()
    : vbbox(initData(&vbbox,"vbbox","BBox to restrict the volume to"))
    , size(initData(&size,(Real)-8.0,"size","Size of the generate tetrahedra. If negative, number of grid cells in the largest bbox dimension"))
    , inputPoints(initData(&inputPoints,"inputPoints","Input surface mesh points"))
    , inputTriangles(initData(&inputTriangles,"inputTriangles","Input surface mesh triangles"))
    , inputQuads(initData(&inputQuads,"inputQuads","Input surface mesh quads"))
    , outputPoints(initData(&outputPoints,"outputPoints","Output volume mesh points"))
    , outputTetrahedra(initData(&outputTetrahedra,"outputTetrahedra","Output volume mesh tetrahedra"))
    , alphaLong(initData(&alphaLong,(Real)0.24999,"alphaLong","Minimum alpha values on long edges when snapping points"))
    , alphaShort(initData(&alphaShort,(Real)0.42978,"alphaShort","Minimum alpha values on short edges when snapping points"))
    , bSnapPoints(initData(&bSnapPoints,false,"snapPoints","Snap points to the surface if intersections on edges are closed to given alpha values"))
    , bSplitTetrahedra(initData(&bSplitTetrahedra,false,"splitTetrahedra","Split tetrahedra crossing the surface"))
    , bDraw(initData(&bDraw,false,"draw","Activate rendering of internal datasets"))
{
    addAlias(&outputTetrahedra,"outputTetras");
    addAlias(&bSplitTetrahedra,"splitTetras");
}

MeshTetraStuffing::~MeshTetraStuffing()
{
}

void MeshTetraStuffing::init()
{
    const SeqPoints& inP = inputPoints.getValue();
    SeqTriangles inT = inputTriangles.getValue();
    const SeqQuads& inQ = inputQuads.getValue();
    if (inP.empty() || (inT.empty() && inQ.empty()))
    {
        serr << "Empty input mesh. Use data dependency to link them to a loaded Topology or MeshLoader" << sendl;
        return;
    }
    if (!inQ.empty())
    {
        // triangulate quads
        inT.reserve(inQ.size()*2);
        for (unsigned int i=0; i < inQ.size(); ++i)
        {
            inT.push_back(Triangle(inQ[i][0], inQ[i][1], inQ[i][2]));
            inT.push_back(Triangle(inQ[i][0], inQ[i][2], inQ[i][3]));
        }
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
    helper::fixed_array<Point,2>& bb = *vbbox.beginEdit();
    if (bb[0][0] >= bb[1][0])
    {
        bb[0] = inputBBox[0] - (inputBBox[1]-inputBBox[0])*0.01;
        bb[1] = inputBBox[1] + (inputBBox[1]-inputBBox[0])*0.01;
    }
    vbbox.endEdit();

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
        c0[c] = - (int((p0[c] - bb[0][c]) / cellsize) + 2);
        c1[c] =   (int((bb[1][c] - p0[c]) / cellsize) + 2);
        gsize[c] = c1[c]-c0[c]+1;
        g0[c] = p0[c] + c0[c]*cellsize;
        hsize[c] = gsize[c]+1;
        h0[c] = g0[c] - cellsize/2;
    }

    sout << "Grid <"<<c0[0]<<","<<c0[1]<<","<<c0[2]<<">-<"<<c1[0]<<","<<c1[1]<<","<<c1[2]<<">" << sendl;

    SeqPoints& outP = *outputPoints.beginEdit();
    SeqTetrahedra& outT = *outputTetrahedra.beginEdit();

    outP.resize(gsize[0]*gsize[1]*gsize[2] + hsize[0]*hsize[1]*hsize[2]);

    sout << "Grid 1 size " << gsize[0] << "x" << gsize[1] << "x" << gsize[2] << ", total " << gsize[0]*gsize[1]*gsize[2] << sendl;
    sout << "Grid 2 size " << hsize[0] << "x" << hsize[1] << "x" << hsize[2] << ", total " << hsize[0]*hsize[1]*hsize[2] << sendl;

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
    sout << "nbp = " << nbp << sendl;

    pInside.resize(nbp);
    eBDist.resize(nbp);
    for (int p=0; p<nbp; ++p)
    {
        pInside[p] = 0;
        // find edges where we are the first point
        for (int e=0; e<EDGESHELL; ++e)
        {
            eBDist[p][e] = 0.0;
        }
    }
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
                std::set< int > tris;
                for (unsigned int i=0; i<results.size(); ++i)
                {
                    if (tris.find(results[i].tid) != tris.end())
                    {
                        if (i < results.size()-1)
                            results[i] = results[results.size()-1];
                        results.resize(results.size()-1);
                        --i;
                    }
                    else tris.insert(results[i].tid);
                }
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
                            if (eBDist[p1][e] < 0.00001)
                                sout << "WARNING: point " << p1 << " is too close to the surface on edge " << e << " : alpha = " << eBDist[p1][e]<<sendl;
                        }
                        if (i > 0)
                        {
                            eBDist[p1][e+1] = d - results[i-1].t;
                            if (eBDist[p1][e+1] < 0.00001)
                                sout << "WARNING: point " << p1 << " is too close to the surface on edge " << e+1 << " : alpha = " << eBDist[p1][e+1]<<sendl;
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

    if (bSnapPoints.getValue())
    {
        const Real alphaLong = this->alphaLong.getValue();
        const Real alphaShort = this->alphaShort.getValue();
        vector<bool> pViolated;
        pViolated.resize(nbp);
        std::set<int> violatedInsidePoints;
        std::set<int> violatedOutsidePoints;
        for (int p=0; p<nbp; ++p)
        {
            bool in1 = (pInside[p] > 0);
            bool violated = false;
            for (int e=0; e<EDGESHELL && !violated; ++e)
            {
                int p2 = getEdgePoint2(p,e);
                if (p2 == -1) continue;
                bool in2 = (pInside[p2] > 0);
                if (!(in1 ^ in2)) continue;

                Real alpha;
                if (e<6) // long edges
                    alpha = alphaLong;
                else
                    alpha = alphaShort;
                if (eBDist[p][e] != 0.0 && eBDist[p][e] < alpha)
                {
                    //sout << "point " << p << " violated on edge " << e << " by alpha " << eBDist[p][e] << sendl;
                    violated = true;
                }
            }
            pViolated[p] = violated;
            if (violated)
            {
                if (in1)
                    violatedInsidePoints.insert(p);
                else
                    violatedOutsidePoints.insert(p);
            }
        }
        // ordered wrapping : first try to move violated exterior points toward an unviolated interior point
        int nwraps = 0;
        do
        {
            nwraps = 0;
            for (std::set<int>::const_iterator it = violatedOutsidePoints.begin(), itend = violatedOutsidePoints.end(); it != itend; ++it)
            {
                int p = *it;
                if (pInside[p] == 0) continue; // point already wrapped
                Real minDist = 0;
                int minEdge = -1;
                for (int e=0; e<EDGESHELL; ++e)
                {
                    int p2 = getEdgePoint2(p,e);
                    if (p2 == -1) continue;
                    bool in2 = (pInside[p2] > 0);
                    if (in2 || pViolated[p2]) continue; // only move towards unviolated points
                    Real alpha;
                    if (e<6) // long edges
                        alpha = alphaLong;
                    else
                        alpha = alphaShort;
                    if (eBDist[p][e] == 0.0 || eBDist[p][e] >= alpha) continue; // this edge is not violated
                    Real dist = eBDist[p][e] * eBDist[p][e] * getEdgeSize2(e);
                    if (minEdge == -1 || dist < minDist)
                    {
                        minEdge = e;
                        minDist = dist;
                    }
                }
                if (minEdge == -1) continue; // no violated edge toward an unviolated point
                int e = minEdge;
                int p2 = getEdgePoint2(p,e);
                // Wrap p toward p2
                sout << "Wrapping outside point " << p << " toward " << p2 << " by " << eBDist[p][e] << sendl;
                outP[p] += (outP[p2]-outP[p]) * (eBDist[p][e]);
                snaps.push_back(outP[p]);
                ++nwraps;
                pInside[p] = 0; // p is now on the surface
                pViolated[p] = false; // and now longer violated
                for (int e=0; e<EDGESHELL; ++e)
                {
                    eBDist[p][e] = 0; // remove all cut points from p
                    int p2 = getEdgePoint2(p,e);
                    if (p2 == -1) continue;
                    eBDist[p2][e^1] = 0; // remove all cut points toward p
                    if (pViolated[p2] && pInside[p2] > 0)
                    {
                        // check to see if p2 is still violated
                        bool violated = false;
                        for (int e2=0; e2<EDGESHELL && !violated; ++e2)
                        {
                            int p3 = getEdgePoint2(p2,e2);
                            if (p3 == -1) continue;
                            if (pInside[p3] >= 0) continue;
                            Real alpha;
                            if (e<6) // long edges
                                alpha = alphaLong;
                            else
                                alpha = alphaShort;
                            if (eBDist[p2][e2] != 0.0 && eBDist[p2][e2] < alpha)
                                violated = true;
                        }
                        if (!violated)
                        {
                            // p2 is no longer violated
                            pViolated[p2] = false;
                            violatedInsidePoints.erase(p2);
                        }
                    }
                }
            }
        }
        while (nwraps > 0);
        // then order remaining violated inside points
        for (std::set<int>::const_iterator it = violatedInsidePoints.begin(), itend = violatedInsidePoints.end(); it != itend; ++it)
        {
            int p = *it;
            if (pInside[p] == 0)
            {
                serr << "ERROR: inside point " << p << " already wrapped." << sendl;
                continue;
            }
            Real minDist = 0;
            int minEdge = -1;
            for (int e=0; e<EDGESHELL; ++e)
            {
                int p2 = getEdgePoint2(p,e);
                if (p2 == -1) continue;
                if (pInside[p2] >= 0) continue; // only move towards outside points
                Real alpha;
                if (e<6) // long edges
                    alpha = alphaLong;
                else
                    alpha = alphaShort;
                if (eBDist[p][e] == 0.0 || eBDist[p][e] >= alpha) continue; // this edges is not violated
                Real dist = eBDist[p][e] * eBDist[p][e] * getEdgeSize2(e);
                if (minEdge == -1 || dist < minDist)
                {
                    minEdge = e;
                    minDist = dist;
                }
            }
            if (minEdge == -1) // no violated edge
            {
                serr << "ERROR: inside point " << p << " has no violated edges." << sendl;
                continue;
            }
            int e = minEdge;
            int p2 = getEdgePoint2(p,e);
            // Wrap p toward p2
            sout << "Wrapping inside point " << p << " toward " << p2 << " by " << eBDist[p][e] << sendl;
            outP[p] += (outP[p2]-outP[p]) * (eBDist[p][e]);
            snaps.push_back(outP[p]);
            pInside[p] = 0; // p is now on the surface
            pViolated[p] = false; // and now longer violated
            for (int e=0; e<EDGESHELL; ++e)
            {
                eBDist[p][e] = 0; // remove all cut points from p
                int p2 = getEdgePoint2(p,e);
                if (p2 == -1) continue;
                eBDist[p2][e^1] = 0; // remove all cut points toward p
            }
        }
    }

    // Create cut points
    if (bSplitTetrahedra.getValue())
    {
        for (int p=0; p<nbp; ++p)
        {
            if (pInside[p] >= 0) continue; // look for outside points
            for (int e=0; e<EDGESHELL; ++e)
            {
                int p2 = getEdgePoint2(p,e);
                if (p2 == -1) continue;
                if (pInside[p2] <= 0) continue; // look for inside points
                int newP = outP.size();
                sout << "Creating cut point " << newP << " between " << p << " and " << p2 << " at " << eBDist[p][e] << sendl;
                outP.push_back( outP[p] + (outP[p2]-outP[p]) * (eBDist[p][e]) );
                splitPoints[std::make_pair(p,p2)]=newP;
            }
        }
    }

    // Create tetrahedra inside or crossing the mesh
    const int gsize01 = gsize[0]*gsize[1];
    const int hsize01 = hsize[0]*hsize[1];
    for (int p=0, ph=ph0, z=0; z<gsize[2]; ++z,ph+=hsize[0])
        for (int y=0; y<gsize[1]; ++y,++ph)
            for (int x=0; x<gsize[0]; ++x,++p,++ph)
            {
                //p = x + gsize[0] * (y + gsize[1] * (z));
                //ph = ph0 + x + hsize[0] * (y + hsize[1] * (z));
                if (x > 0)
                {
                    // edge in X axis
                    int p2 = p - 1;
                    int hshell[4] = {ph, ph + hsize[0], ph + hsize[0] + hsize01, ph + hsize01};
                    for (int i=0; i<4; ++i)
                    {
                        int p3 = hshell[i];
                        int p4 = hshell[(i+1)%4];
                        addTetra(outT, outP, p,p2,p4,p3, __LINE__);
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
                        addTetra(outT, outP, p,p2,p4,p3, __LINE__);
                    }
                }
                if (z > 0)
                {
                    // edge in Z axis
                    int p2 = p - gsize01;
                    int hshell[4] = {ph, ph + 1, ph + 1 + hsize[0], ph + hsize[0]};
                    for (int i=0; i<4; ++i)
                    {
                        int p3 = hshell[i];
                        int p4 = hshell[(i+1)%4];
                        addTetra(outT, outP, p,p2,p4,p3, __LINE__);
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
                if (pInside.size() > p)
                {
                    pInside[newPid[p]] = pInside[p];
                }
            }
        }
    }
    outP.resize(nbp);
    pInside.resize(nbp);
    eBDist.clear();

    for (unsigned int t=0; t<outT.size(); ++t)
        for (int i=0; i<4; ++i)
            outT[t][i] = newPid[outT[t][i]];

    std::set<Triangle> triSet;

    // Check if all tetrahedra volumes are positive
    for (unsigned int t=0; t<outT.size(); ++t)
    {
        Point a = outP[outT[t][1]] - outP[outT[t][0]];
        Point b = outP[outT[t][2]] - outP[outT[t][0]];
        Point c = outP[outT[t][3]] - outP[outT[t][0]];
        Real vol6 = a*(b.cross(c));
        if (vol6 < 0)
        {
            sout << "WARNING: tetra " << t << " is inverted." << sendl;
            int tmp = outT[t][2]; outT[t][2] = outT[t][3]; outT[t][3] = tmp;
        }
        for (int i=0; i<4; ++i)
        {
            int i0 = 0;
            if (outT[t][(i+2)%4] < outT[t][(i+i0)%4]) i0 = 1;
            if (outT[t][(i+3)%4] < outT[t][(i+i0)%4]) i0 = 2;
            Triangle tr(outT[t][(i+1+((i0+0)%3))%4],outT[t][(i+1+((i0+1)%3))%4],outT[t][(i+1+((i0+2)%3))%4]);
            if (i%2) { int tmp = tr[1]; tr[1] = tr[2]; tr[2] = tmp; }
            if (!triSet.insert(tr).second)
            {
                serr << "ERROR: duplicate triangle " << tr << " in tetra " << t <<" : " << outT[t] << sendl;
            }
        }
    }

    sout << "Final mesh: " << outP.size() << " points, "<<outT.size() << " tetrahedra." << sendl;

    outputPoints.endEdit();
    outputTetrahedra.endEdit();
}

void MeshTetraStuffing::addFinalTetra(SeqTetrahedra& outT, SeqPoints& outP, int p1, int p2, int p3, int p4, bool flip, int line)
{
    if (flip)
    {
        int tmp = p3; p3 = p4; p4 = tmp;
    }
    Point a = outP[p2] - outP[p1];
    Point b = outP[p3] - outP[p1];
    Point c = outP[p4] - outP[p1];
    Real vol6 = a*(b.cross(c));
    if (vol6 < 0)
    {
        sout << __FILE__ << "(" << line << "): WARNING: final tetra " << p1 << " " << p2 << " " << p3 << " " << p4 << " is inverted." << sendl;
        int tmp = p3; p3 = p4; p4 = tmp;
    }
    outT.push_back(Tetra(p1,p2,p3,p4));
}

bool MeshTetraStuffing::needFlip(int p1, int p2, int p3, int p4, int q1, int q2, int q3, int q4)
{
    bool flip = false;
    // make the smallest indice the first vertex
    while(p1 > p2 || p1 > p3 || p1 > p4)
    {
        int tmp = p1; p1 = p2; p2 = p3; p3 = p4; p4 = tmp; flip = !flip;
    }
    while(q1 > q2 || q1 > q3 || q1 > q4)
    {
        int tmp = q1; q1 = q2; q2 = q3; q3 = q4; q4 = tmp; flip = !flip;
    }

    // make the second smallest indice the second vertex
    while(p2 > p3 || p2 > p4)
    {
        int tmp = p2; p2 = p3; p3 = p4; p4 = tmp; //flip = !flip;
    }
    while(q2 > q3 || q2 > q4)
    {
        int tmp = q2; q2 = q3; q3 = q4; q4 = tmp; //flip = !flip;
    }

    // the tetrahedra are flipped if the last edge is flipped
    if (p3 == q4) flip = !flip;
    return flip;
}

void MeshTetraStuffing::addTetra(SeqTetrahedra& outT, SeqPoints& outP, int p1, int p2, int p3, int p4, int line)
{
    {
        Point a = outP[p2] - outP[p1];
        Point b = outP[p3] - outP[p1];
        Point c = outP[p4] - outP[p1];
        Real vol6 = a*(b.cross(c));
        if (vol6 < 0)
        {
            sout << "MeshTetraStuffing("<<line<<"): WARNING: grid tetra " << p1 << " " << p2 << " " << p3 << " " << p4 << " is inverted." << sendl;
            int tmp = p3; p3 = p4; p4 = tmp;
        }
    }
    int in1 = pInside[p1];
    int in2 = pInside[p2];
    int in3 = pInside[p3];
    int in4 = pInside[p4];
    int nneg = 0, npos = 0, nzero = 0;
    int pneg[4];
    int ppos[4];
    int pzero[4];
    if (in1 < 0) pneg[nneg++]=p1; else if (in1 > 0) ppos[npos++]=p1; else pzero[nzero++]=p1;
    if (in2 < 0) pneg[nneg++]=p2; else if (in2 > 0) ppos[npos++]=p2; else pzero[nzero++]=p2;
    if (in3 < 0) pneg[nneg++]=p3; else if (in3 > 0) ppos[npos++]=p3; else pzero[nzero++]=p3;
    if (in4 < 0) pneg[nneg++]=p4; else if (in4 > 0) ppos[npos++]=p4; else pzero[nzero++]=p4;
    if (npos == 0 && nneg > 0) return ; // no tetra
    if (nneg == 0 || !bSplitTetrahedra.getValue()) // full tetra
    {
        addFinalTetra(outT,outP, p1,p2,p3,p4, false,__LINE__);
    }
    else if (npos == 1)
    {
        // only one tetra, move the negative points
        int p0 = ppos[0];
        if (in1 < 0) p1 = getSplitPoint(p1,p0);
        if (in2 < 0) p2 = getSplitPoint(p2,p0);
        if (in3 < 0) p3 = getSplitPoint(p3,p0);
        if (in4 < 0) p4 = getSplitPoint(p4,p0);
        addFinalTetra(outT,outP, p1,p2,p3,p4, false,__LINE__);
    }
    else if (npos == 2 && nneg == 1)
    {
        // two tetrahedra
        int p0 = pzero[0];
        int cut1 = getSplitPoint(pneg[0],ppos[0]);
        int cut2 = getSplitPoint(pneg[0],ppos[1]);
        bool flipD = flipDiag(outP, ppos[0],ppos[1],cut2,cut1,pneg[0]);
        bool flipT = needFlip(p0, ppos[0], ppos[1], pneg[0], p1,p2,p3,p4);
        if (!flipD)
        {
            addFinalTetra(outT,outP, p0,ppos[0],ppos[1],cut2, flipT,__LINE__);
            addFinalTetra(outT,outP, p0,ppos[0],cut2,cut1, flipT,__LINE__);
        }
        else
        {
            addFinalTetra(outT,outP, p0,ppos[0],ppos[1],cut1, flipT,__LINE__);
            addFinalTetra(outT,outP, p0,ppos[1],cut2,cut1, flipT,__LINE__);
        }
    }
    else if (npos == 2 && nneg == 2)
    {
        int cut1 = getSplitPoint(pneg[0],ppos[0]);
        int cut2 = getSplitPoint(pneg[0],ppos[1]);
        int cut3 = getSplitPoint(pneg[1],ppos[0]);
        int cut4 = getSplitPoint(pneg[1],ppos[1]);
        bool flipA = flipDiag(outP, ppos[0],ppos[1],cut2,cut1,pneg[0]);
        bool flipB = flipDiag(outP, ppos[0],ppos[1],cut4,cut3,pneg[1]);
        bool flipT = needFlip(ppos[0],ppos[1],pneg[0],pneg[1], p1,p2,p3,p4);
        if (!flipA && flipB)
        {
            addFinalTetra(outT,outP, ppos[0],ppos[1],cut2,cut3, flipT,__LINE__);
            addFinalTetra(outT,outP, ppos[0],cut2,cut1,cut3, flipT,__LINE__);
            addFinalTetra(outT,outP, ppos[1],cut2,cut3,cut4, flipT,__LINE__);
        }
        else if (flipA && !flipB)
        {
            //bool flipT = needFlip(ppos[1],pneg[0],ppos[0],pneg[1], p1,p2,p3,p4);
            addFinalTetra(outT,outP, ppos[0],ppos[1],cut1,cut4, flipT,__LINE__);
            addFinalTetra(outT,outP, ppos[1],cut1,cut4,cut2, flipT,__LINE__);
            addFinalTetra(outT,outP, ppos[0],cut1,cut3,cut4, flipT,__LINE__);
        }
        else
        {
            serr << "Invalid tetra split: flipA = " << flipA << "   flipB = " << flipB << sendl;
        }
    }
    else // npos == 3 && nneg == 1
    {
        int cut1 = getSplitPoint(pneg[0],ppos[0]);
        int cut2 = getSplitPoint(pneg[0],ppos[1]);
        int cut3 = getSplitPoint(pneg[0],ppos[2]);
        bool flip1 = flipDiag(outP, ppos[0],ppos[1],cut2,cut1,pneg[0]);
        bool flip2 = flipDiag(outP, ppos[1],ppos[2],cut3,cut2,pneg[0]);
        bool flip3 = flipDiag(outP, ppos[2],ppos[0],cut1,cut3,pneg[0]);
        if (flip1 == flip2 && flip2 == flip3)
        {
            serr << "Invalid tetra split" << sendl;
            flip3 = !flip1;
        }
        int pp0;
        if (flip1 && !flip2)    pp0 = ppos[1];
        else if (flip2 && !flip3)    pp0 = ppos[2];
        else /* (flip3 && !flip1) */ pp0 = ppos[0];
        int pc0;
        if (!flip1 && flip2)    pc0 = cut2;
        else if (!flip2 && flip3)    pc0 = cut3;
        else /* (!flip3 && flip1) */ pc0 = cut1;
        bool flipT = needFlip(pneg[0],ppos[0],ppos[1],ppos[2], p1,p2,p3,p4);
        addFinalTetra(outT,outP, pp0, cut1, cut3, cut2, flipT,__LINE__);
        addFinalTetra(outT,outP, pc0, ppos[0], ppos[1], ppos[2], flipT,__LINE__);
        if (flip1 == flip2)    addFinalTetra(outT,outP, ppos[1], cut2, pp0, pc0, needFlip(ppos[1],pneg[0],pp0,(pp0 == ppos[0] ? ppos[2] : ppos[0]), p1,p2,p3,p4),__LINE__);
        else if (flip2 == flip3)    addFinalTetra(outT,outP, ppos[2], cut3, pp0, pc0, needFlip(ppos[2],pneg[0],pp0,(pp0 == ppos[0] ? ppos[1] : ppos[0]), p1,p2,p3,p4),__LINE__);
        else /* (flip2 == flip3) */ addFinalTetra(outT,outP, ppos[0], cut1, pp0, pc0, needFlip(ppos[0],pneg[0],pp0,(pp0 == ppos[1] ? ppos[2] : ppos[1]), p1,p2,p3,p4),__LINE__);
    }
}

/// Should the diagonal of abcd should be bd instead of ac ?
bool MeshTetraStuffing::flipDiag(const SeqPoints& outP, int a, int b, int c, int d, int e)
{
    bool done = false;
    bool flip = false;
    // choose the cut point along the long edge if any
    if (e < ph0)
    {
        if (a < ph0)
        {
            flip = true;
            done = true;
        }
        else if (b < ph0)
        {
            flip = false;
            done = true;
        }
    }
    else
    {
        if (a >= ph0)
        {
            flip = true;
            done = true;
        }
        else if (b >= ph0)
        {
            flip = false;
            done = true;
        }
    }
    if (!done)
    {
        // long edge is ab
        int nsup = 0;
        for (int i=0; i<3; ++i)
            if (outP[a][i] > outP[c][i]) ++nsup;
        // we choose ac if a has an odd number of coordinates greater than c's coordinates,
        // but reverse the condition if ab is on the second grid
        if ((nsup&1) == (a<ph0 ? 1 : 0))
            flip = false;
        else
            flip = true;
    }

    if (!flip)
    {
        diags.push_back(outP[a]); //*0.95f + outP[c]*0.05f);
        diags.push_back(outP[c]); //*0.95f + outP[a]*0.05f);
    }
    else
    {
        diags.push_back(outP[b]); //*0.95f + outP[d]*0.05f);
        diags.push_back(outP[d]); //*0.95f + outP[b]*0.05f);
    }
    return flip;
}

int MeshTetraStuffing::getSplitPoint(int from, int to)
{
    std::map<std::pair<int,int>, int>::const_iterator it = splitPoints.find(std::make_pair(from,to));
    if (it != splitPoints.end()) return it->second;
    it = splitPoints.find(std::make_pair(to, from));
    if (it != splitPoints.end()) return it->second;
    serr << "ERROR: cut point between " << from << " and " << to << " not found." << sendl;
    return from;
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

void MeshTetraStuffing::draw(const core::visual::VisualParams* vparams)
{
    if (!bDraw.getValue())
        return;

    const SeqPoints& outP = outputPoints.getValue();
    vparams->drawTool()->drawPoints(intersections, 2, Vec<4,float>(1,0,0,1));
    vparams->drawTool()->drawPoints(outP, 1, Vec<4,float>(0,1,0,1));
    if (!diags.empty())
        vparams->drawTool()->drawLines(diags, 1, Vec<4,float>(0,1,1,1));

    if (!snaps.empty())
        vparams->drawTool()->drawPoints(snaps, 4, Vec<4,float>(0,0,1,1));
}

}

} // namespace component

} // namespace sofa
