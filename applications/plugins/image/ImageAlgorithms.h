/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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

#ifndef IMAGE_IMAGEALGORITHMS_H
#define IMAGE_IMAGEALGORITHMS_H

#include "ImageTypes.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/rmath.h>
#include <set>
#include <vector>

#include <omp.h>

using namespace cimg_library;

/**
*  Move points in position data to the centroid of their voronoi region
*  centroid are computed given a (biased) distance measure d as $p + (b)/N \sum_i d(p,p_i)*2/((b)+(b_i))*(p_i-p)/|p_i-p|$
*  with no bias, we obtain the classical mean $1/N \sum_i p_i$
* returns true if points have moved
*/

template<typename real,typename T>
bool Lloyd (std::vector<sofa::defaulttype::Vec<3,real> >& pos, CImg<real>& distances, CImg<unsigned int>& voronoi, const sofa::defaulttype::ImageLPTransform<real>& transform,  const CImg<T>* biasFactor=NULL)
{
    typedef sofa::defaulttype::Vec<3,real> Coord;
    unsigned int nbp=pos.size();
    bool moved=false;

    // get rounded point coordinates in image (to check that points do not share the same voxels)
    std::vector<Coord> P;
    P.resize(nbp);
    for (unsigned int i=0; i<nbp; i++) { Coord p = transform.toImage(pos[i]);  for (unsigned int j=0; j<3; j++)  P[i][j]=round(p[j]); }

    #pragma omp parallel for
    for (unsigned int i=0; i<nbp; i++)
    {
        // compute centroid
        Coord c,p,u;
        unsigned int count=0;
        bool valid=true;
        cimg_forXYZ(voronoi,x,y,z) if (voronoi(x,y,z)==i+1)
        {
            p=transform.fromImage(Coord(x,y,z));
            u=p-pos[i]; u.normalize();
            c+=u*distances(x,y,z);
            count++;
        }
        if(!count) goto stop;

        c/=(real)count;

        if (biasFactor)
        {
            //                Real stiff=getStiffness(grid.data()[indices[i]]);
            //                if(biasFactor!=(Real)1.) stiff=(Real)pow(stiff,biasFactor);
            //                c*=stiff;
        }

        c+=pos[i];

        // check validity
        p = transform.toImage(c); for (unsigned int j=0; j<3; j++) p[j]=round(p[j]);
        if (distances(p[0],p[1],p[2])==-1) valid=false; // out of object
        else { for (unsigned int j=0; j<nbp; j++) if(i!=j) if(P[j][0]==p[0]) if(P[j][1]==p[1]) if(P[j][2]==p[2]) valid=false; } // check occupancy

        while(!valid)  // get closest unoccupied point in voronoi
        {
            real dmin=cimg::type<real>::max();
            cimg_forXYZ(voronoi,x,y,z) if (voronoi(x,y,z)==i+1)
            {
                Coord pi=transform.fromImage(Coord(x,y,z));
                real d2=(c-pi).norm2();
                if(dmin>d2) { dmin=d2; p=Coord(x,y,z); }
            }
            if(dmin==cimg::type<real>::max()) goto stop;// no point found
            bool val2=true; for (unsigned int j=0; j<nbp; j++) if(i!=j) if(P[j][0]==p[0]) if(P[j][1]==p[1]) if(P[j][2]==p[2]) val2=false; // check occupancy
            if(val2) valid=true;
            else voronoi(p[0],p[1],p[2])=0;
        }

        if(P[i][0]!=p[0] || P[i][1]!=p[1] || P[i][2]!=p[2]) // set new position if different
        {
            pos[i] = transform.fromImage(p);
            for (unsigned int j=0; j<3; j++) P[i][j]=p[j];
            moved=true;
        }
stop: ;
    }

    return moved;
}


/**
* Computes geodesic distances in the image from @param pos up to @param distMax, given a bias distance function b(x).
* This is equivalent to solve for the eikonal equation || grad d(x) || = 1/b(x) with d(p)=0 at @param pos
* using fast marching method presented from sethian http://math.berkeley.edu/~sethian/2006/Publications/Book/2006/
* distances should be intialized (-1 outside the object, and >0 inside)
* returns @param voronoi and @param distances
*/

template<typename real,typename T>
void fastMarching (CImg<real>& distances, CImg<unsigned int>& voronoi, const std::vector<sofa::defaulttype::Vec<3,real> >& pos,  const sofa::defaulttype::ImageLPTransform<real>& transform, const CImg<T>* biasFactor=NULL, const real distMax=cimg::type<real>::max())
{
    /*
        typedef sofa::defaulttype::Vec<3,real> Coord;
        typedef sofa::defaulttype::Vec<3,int> iCoord;

        unsigned int nbp=pos.size();

        // get rounded point coordinates in image
        std::vector<iCoord> P;
        P.resize(nbp);
        for (unsigned int i=0;i<nbp;i++)  { Coord p = transform.toImage(pos[i]);  for (unsigned int j=0;j<3;j++)  P[i][j]=round(p[j]); }

        // init
        //    CImg_3x3x3(D,real); // cimg neighborhood for distances
        //    CImg_3x3x3(V,unsigned int); // cimg neighborhood for voronoi
        //    sofa::defaulttype::Vec<27,  iCoord > offset; // image coord offsets related to neighbors
        //    int count=0; for (int k=-1;k<=1;k++) for (int j=-1;j<=1;j++) for (int i=-1;i<=1;i++) offset[count++]= iCoord(i,j,k);

        const Coord &voxelsize=transform.getScale();

        // add samples t the queue
        typedef std::pair<real,iCoord > DistanceToPoint;
        typedef typename std::set<DistanceToPoint>::iterator IT;
        std::set<DistanceToPoint> q; // priority queue
        for (unsigned int i=0;i<nbp;i++)
        {
            if(distances.containsXYZC(P[i][0],P[i][1],P[i][2]))
                if(distances(P[i][0],P[i][1],P[i][2])!=-1)
                {
                    q.insert( DistanceToPoint(0.,P[i]) );
                    distances(P[i][0],P[i][1],P[i][2])=0;
                    voronoi(P[i][0],P[i][1],P[i][2])=i+1;
                }
        }

        // dijkstra
        while( !q.empty() )
        {
            DistanceToPoint top = *q.begin();
            q.erase(q.begin());
            iCoord v = top.second;

            int x = v[0] ,y = v[1] ,z = v[2];
            const int _p1x = x?x-1:x, _p1y = y?y-1:y, _p1z = z?z-1:z, _n1x = x<distances.width()-1?x+1:x, _n1y = y<distances.height()-1?y+1:y, _n1z = z<distances.depth()-1?z+1:z;    // boundary conditions for cimg neighborood manipulation macros

            cimg_get3x3x3(distances,x,y,z,0,D,real); // get distances in neighborhood
            cimg_get3x3x3(voronoi,x,y,z,0,V,unsigned int); // get voronoi in neighborhood

            if(biasFactor) { }  // TO DO!!!   define lD for biased distances

            for (unsigned int i=0;i<27;i++)
            {
                real d = Dccc + lD[i];
                if(D[i] > d )
                {
                    iCoord v2 = v + offset[i];
                    if(distances.containsXYZC(v2[0],v2[1],v2[2]))
                    {
                        if(D[i] < distMax) { IT it=q.find(DistanceToPoint(D[i],v2)); if(it!=q.end()) q.erase(it); }
                        voronoi(v2[0],v2[1],v2[2]) = Vccc;
                        distances(v2[0],v2[1],v2[2]) = d;
                        q.insert( DistanceToPoint(d,v2) );
                    }
                }
            }
        }
        */
}


/**
* solve the local eikonal system:  || grad d || = 1/b given the 6 neihborood values of d at [-dx,+dx,-dy,+dy,-dz,+dz] where [dx,dy,dz] is the voxel size
* using upwind first order approximation (http://math.berkeley.edu/~sethian/2006/Publications/Book/2006/)
* if values at [-2dx,+2dx,-2dy,+2dy,-2dz,+2dz] (@param d2) are provided (!=-1), use upwind second order approximation ("A second-order fast marching eikonal solver", James Rickett and Sergey Fomel , 1999)
* if(d<0) these values are not used (untreated voxels in the fast marching algorithm)
*/
template<typename real>
real Eikonal(const sofa::defaulttype::Vec<6,real>& d,const sofa::defaulttype::Vec<6,real>& d2,const sofa::defaulttype::Vec<3,real>& voxelsize, const real b=(real)1.0)
{
    // get minimum distance in each direction and some precomputations
    unsigned int nbValid=3;
    sofa::defaulttype::Vec<6,real> D(-1,-1,-1),D2;
    sofa::defaulttype::Vec<3,real> S;
    real B2=(real)1./(b*b);
    for (unsigned int i=0; i<3; i++)
    {
        if(d[2*i]!=-1 && d2[2*i]!=-1 && d[2*i]<d[2*i+1]) { D[i]=(4.0*d[2*i]-d2[2*i])/3.0;  S[i]=9.0/(4.0*voxelsize[i]*voxelsize[i]); }
        else if(d[2*i+1]!=-1 && d2[2*i+1]!=-1)           { D[i]=(4.0*d[2*i+1]-d2[2*i+1])/3.0;    S[i]=9.0/(4.0*voxelsize[i]*voxelsize[i]); }
        else if(d[2*i]!=-1 && d[2*i]<d[2*i+1])           { D[i]=d[2*i];   S[i]=1.0/(voxelsize[i]*voxelsize[i]); }
        else if(d[2*i+1]!=-1)                            { D[i]=d[2*i+1];  S[i]=1.0/(voxelsize[i]*voxelsize[i]); }
        else nbValid--;
        D2[i]=D[i]*D[i];
    }

    // solve sum S_i*(U-D_i)^2  = 1/b^2
    while(0)
    {
        if(nbValid==0) return -1; // no valid neighbor
        else if(nbValid==1) { for (unsigned int i=0; i<3; i++) if(D[i]!=-1.) return (D[i]+voxelsize[i]/b); } // one valid neighbor -> simple 1D propagation
        else // two or three valid neighbors -> quadratic equation
        {
            real A=(real)0,B=(real)0,C=-B2;
            for (unsigned int i=0; i<3; i++) if(D[i]!=-1.) { A+=S[i]; B+=D[i]*S[i];  C+=D2[i]*S[i]; } B*=(real)-2.;
            real Delta=B*B-4.0*A*C;
            if(Delta<0) { if(D[0]>D[1]) { if(D[0]>D[2]) D[0]=-1; else D[2]=-1; } else { if(D[1]>D[2]) D[1]=-1; else D[2]=-1; }  nbValid--; }
            else
            {
                real U=0.5*(sqrt(Delta)-B)/A;  // largest root since A>0
                if(U>D[0]) if(U>D[1]) if(U>D[2]) return U;
                // one entry should be canceled
                if(D[0]>D[1]) { if(D[0]>D[2]) D[0]=-1; else D[2]=-1; }
                else { if(D[1]>D[2]) D[1]=-1; else D[2]=-1; }  nbValid--;
            }
        }
    }
}



/**
* Computes geodesic distances in the image from @param pos up to @param distMax, given a bias distance function b(x).
* This is equivalent to solve for the eikonal equation || grad d(x) || = 1/b(x) with d(p)=0 at @param pos
* using dijkstra minimum path algorithm
* distances should be intialized (-1 outside the object, and >0 inside)
* returns @param voronoi and @param distances
*/

template<typename real,typename T>
void dijkstra (CImg<real>& distances, CImg<unsigned int>& voronoi, const std::vector<sofa::defaulttype::Vec<3,real> >& pos,  const sofa::defaulttype::ImageLPTransform<real>& transform, const CImg<T>* biasFactor=NULL, const real distMax=cimg::type<real>::max())
{
    typedef sofa::defaulttype::Vec<3,real> Coord;
    typedef sofa::defaulttype::Vec<3,int> iCoord;

    unsigned int nbp=pos.size();

    // get rounded point coordinates in image
    std::vector<iCoord> P;
    P.resize(nbp);
    for (unsigned int i=0; i<nbp; i++)  { Coord p = transform.toImage(pos[i]);  for (unsigned int j=0; j<3; j++)  P[i][j]=round(p[j]); }

    // init
    CImg_3x3x3(D,real); // cimg neighborhood for distances
    CImg_3x3x3(V,unsigned int); // cimg neighborhood for voronoi
    sofa::defaulttype::Vec<27,  iCoord > offset; // image coord offsets related to neighbors
    int count=0; for (int k=-1; k<=1; k++) for (int j=-1; j<=1; j++) for (int i=-1; i<=1; i++) offset[count++]= iCoord(i,j,k);

    CImg<real> lD(3,3,3);  // local distances
    if(!biasFactor) // precompute local distances (supposing that the transformation is not projective)
    {
        lD(1,1,1)=0;
        lD(2,1,1) = lD(0,1,1) = transform.getScale()[0];
        lD(1,2,1) = lD(1,0,1) = transform.getScale()[1];
        lD(1,1,2) = lD(1,1,0) = transform.getScale()[2];
        lD(2,2,1) = lD(0,2,1) = lD(2,0,1) = lD(0,0,1) = sqrt(transform.getScale()[0]*transform.getScale()[0] + transform.getScale()[1]*transform.getScale()[1]);
        lD(2,1,2) = lD(0,1,2) = lD(2,1,0) = lD(0,1,0) = sqrt(transform.getScale()[0]*transform.getScale()[0] + transform.getScale()[2]*transform.getScale()[2]);
        lD(1,2,2) = lD(1,0,2) = lD(1,2,0) = lD(1,0,0) = sqrt(transform.getScale()[2]*transform.getScale()[2] + transform.getScale()[1]*transform.getScale()[1]);
        lD(2,2,2) = lD(2,0,2) = lD(2,2,0) = lD(2,0,0) = lD(0,2,2) = lD(0,0,2) = lD(0,2,0) = lD(0,0,0) = sqrt(transform.getScale()[0]*transform.getScale()[0] + transform.getScale()[1]*transform.getScale()[1] + transform.getScale()[2]*transform.getScale()[2]);
    }

    // add samples t the queue
    typedef std::pair<real,iCoord > DistanceToPoint;
    typedef typename std::set<DistanceToPoint>::iterator IT;
    std::set<DistanceToPoint> q; // priority queue
    for (unsigned int i=0; i<nbp; i++)
    {
        if(distances.containsXYZC(P[i][0],P[i][1],P[i][2]))
            if(distances(P[i][0],P[i][1],P[i][2])!=-1)
            {
                q.insert( DistanceToPoint(0.,P[i]) );
                distances(P[i][0],P[i][1],P[i][2])=0;
                voronoi(P[i][0],P[i][1],P[i][2])=i+1;
            }
    }

    // dijkstra
    while( !q.empty() )
    {
        DistanceToPoint top = *q.begin();
        q.erase(q.begin());
        iCoord v = top.second;

        int x = v[0] ,y = v[1] ,z = v[2];
        const int _p1x = x?x-1:x, _p1y = y?y-1:y, _p1z = z?z-1:z, _n1x = x<distances.width()-1?x+1:x, _n1y = y<distances.height()-1?y+1:y, _n1z = z<distances.depth()-1?z+1:z;    // boundary conditions for cimg neighborood manipulation macros

        cimg_get3x3x3(distances,x,y,z,0,D,real); // get distances in neighborhood
        cimg_get3x3x3(voronoi,x,y,z,0,V,unsigned int); // get voronoi in neighborhood

        if(biasFactor) { }  // TO DO!!!   define lD for biased distances

        for (unsigned int i=0; i<27; i++)
        {
            real d = Dccc + lD[i];
            if(D[i] > d )
            {
                iCoord v2 = v + offset[i];
                if(distances.containsXYZC(v2[0],v2[1],v2[2]))
                {
                    if(D[i] < distMax) { IT it=q.find(DistanceToPoint(D[i],v2)); if(it!=q.end()) q.erase(it); }
                    voronoi(v2[0],v2[1],v2[2]) = Vccc;
                    distances(v2[0],v2[1],v2[2]) = d;
                    q.insert( DistanceToPoint(d,v2) );
                }
            }
        }
    }
}

#endif // IMAGEALGORITHMS_H
