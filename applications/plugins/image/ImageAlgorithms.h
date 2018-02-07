/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef IMAGE_IMAGEALGORITHMS_H
#define IMAGE_IMAGEALGORITHMS_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/rmath.h>
#include <sofa/defaulttype/Mat.h>
#include <set>
#include <vector>

#if (defined(WIN32) || defined (_XBOX)) && (_MSC_VER < 1800) // for all version anterior to Visual Studio 2013
# include <float.h>
# define isnan(x)  (_isnan(x))
#else
# include <cmath>
# define isnan(x) (std::isnan(x))
#endif

#include "ImageTypes.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/**
*  Move points to the centroid of their voronoi region
*  returns true if points have moved
*/

template<typename real>
bool Lloyd (std::vector<sofa::defaulttype::Vec<3,real> >& pos,const std::vector<unsigned int>& voronoiIndex, cimg_library::CImg<unsigned int>& voronoi)
{
    typedef sofa::defaulttype::Vec<3,real> Coord;
    unsigned int nbp=pos.size();
    bool moved=false;

#ifdef _OPENMP
#pragma omp parallel for
#endif
#ifdef WIN32
	for(long int i=0; i<nbp; i++)
#else
    for (unsigned int i=0; i<nbp; i++)
#endif
    {
        // compute centroid
        Coord C,p;
        unsigned int count=0;
        bool valid=true;

        cimg_forXYZ(voronoi,x,y,z) if (voronoi(x,y,z)==voronoiIndex[i])
        {
            C+=Coord(x,y,z);
            count++;
        }
        if(!count) goto stop;
        C/=(real)count;

        // check validity
        for (unsigned int j=0; j<3; j++) p[j]=sofa::helper::round(C[j]);
        if (voronoi(p[0],p[1],p[2])!=voronoiIndex[i]) valid=false; // out of voronoi
        else { for (unsigned int j=0; j<nbp; j++) if(i!=j) if(sofa::helper::round(pos[j][0])==p[0]) if(sofa::helper::round(pos[j][1])==p[1]) if(sofa::helper::round(pos[j][2])==p[2]) valid=false; } // check occupancy

        while(!valid)  // get closest unoccupied point in voronoi
        {
            real dmin=cimg_library::cimg::type<real>::max();
            cimg_forXYZ(voronoi,x,y,z) if (voronoi(x,y,z)==voronoiIndex[i])
            {
                real d2=(C-Coord(x,y,z)).norm2();
                if(dmin>d2) { dmin=d2; p=Coord(x,y,z); }
            }
            if(dmin==cimg_library::cimg::type<real>::max()) goto stop;// no point found
            bool val2=true; for (unsigned int j=0; j<nbp; j++) if(i!=j) if(sofa::helper::round(pos[j][0])==p[0]) if(sofa::helper::round(pos[j][1])==p[1]) if(sofa::helper::round(pos[j][2])==p[2]) val2=false; // check occupancy
            if(val2) valid=true;
            else voronoi(p[0],p[1],p[2])=0;
        }

        if(pos[i][0]!=p[0] || pos[i][1]!=p[1] || pos[i][2]!=p[2]) // set new position if different
        {
            pos[i] = p;
            moved=true;
        }
stop: ;
    }

    return moved;
}


/**
* solve the local eikonal system:  || grad d || = 1/b given the 6 neihborood values of d at [-dx,+dx,-dy,+dy,-dz,+dz] where [dx,dy,dz] is the voxel size
* using upwind first order approximation (http://math.berkeley.edu/~sethian/2006/Publications/Book/2006/)
* if(d<0) these values are not used (untreated voxels in the fast marching algorithm)
* if values at [-2dx,+2dx,-2dy,+2dy,-2dz,+2dz] (@param d2) are provided (>0), use more accurate second order approximation (cf "A second-order fast marching eikonal solver", James Rickett and Sergey Fomel , 1999)
*/
template<typename real>
real Eikonal(const sofa::defaulttype::Vec<6,real>& d,const sofa::defaulttype::Vec<6,real>& d2,const sofa::defaulttype::Vec<3,real>& voxelsize, const real b=(real)1.0)
{
    // get minimum distance in each direction and some precomputations
    unsigned int nbValid=3;
    sofa::defaulttype::Vec<3,real> D(-1,-1,-1),D2,S;
    real B2=(real)1./(b*b),val;
    for (unsigned int i=0; i<3; i++)
    {
        if(d[2*i]>=0 && d2[2*i]>=0)  { val=(4.0*d[2*i]-d2[2*i])/3.0;  if(val<D[i] || D[i]==-1) { D[i]=val; S[i]=9.0/(4.0*voxelsize[i]*voxelsize[i]); } }
        else if(d[2*i+1]>=0 && d2[2*i+1]>=0)  { val=(4.0*d[2*i+1]-d2[2*i+1])/3.0;  if(val<D[i] || D[i]==-1) { D[i]=val; S[i]=9.0/(4.0*voxelsize[i]*voxelsize[i]); } }
        else if(d[2*i]>=0)  { val=d[2*i];  if(val<D[i] || D[i]==-1) { D[i]=val;  S[i]=1.0/(voxelsize[i]*voxelsize[i]); } }
        else if(d[2*i+1]>=0)  { val=d[2*i+1];  if(val<D[i] || D[i]==-1) { D[i]=val;  S[i]=1.0/(voxelsize[i]*voxelsize[i]); } }
        else nbValid--;
        D2[i]=D[i]*D[i];
    }
    // solve sum S_i*(U-D_i)^2  = 1/b^2
    while(1)
    {
        if(nbValid==0) return -1; // no valid neighbor
        else if(nbValid==1) { for (unsigned int i=0; i<3; i++) if(D[i]>=0.) return (D[i]+voxelsize[i]/b); } // one valid neighbor -> simple 1D propagation
        else // two or three valid neighbors -> quadratic equation
        {
            real A=(real)0,B=(real)0,C=-B2;
            for (unsigned int i=0; i<3; i++) if(D[i]>=0.) { A+=S[i]; B+=D[i]*S[i];  C+=D2[i]*S[i]; } B*=(real)-2.;
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
    return -1;
}

/**
* Update geodesic distances in the image, given a bias distance function b(x).
* This is equivalent to solve for the eikonal equation || grad d(x) || = 1/b(x) with d(p)=0 at @param pos
* using fast marching method presented from sethian http://math.berkeley.edu/~sethian/2006/Publications/Book/2006/
* distances should be intialized (<0 outside the object, >=0 inside, and = 0 for seeds)
* returns @param voronoi and @param distances
*/


template<typename real,typename T>
void fastMarching (std::set<std::pair<real,sofa::defaulttype::Vec<3,int> > > &trial,cimg_library::CImg<real>& distances, cimg_library::CImg<unsigned int>& voronoi,
                   const sofa::defaulttype::Vec<3,real>& voxelsize, const cimg_library::CImg<T>* biasFactor=NULL)
{
    typedef sofa::defaulttype::Vec<3,int> iCoord;
    typedef sofa::defaulttype::Vec<6,real> Dist;
    typedef std::pair<real,iCoord > DistanceToPoint;
    const iCoord dim(distances.width(),distances.height(),distances.depth());

    // init
    sofa::defaulttype::Vec<6,  iCoord > offset; // image coord offsets related to 6 neighbors
    for (unsigned int i=0; i<3; i++) { offset[2*i][i]=-1; offset[2*i+1][i]=1;}
    unsigned int nbOffset=offset.size();
    cimg_library::CImg<bool> alive(dim[0],dim[1],dim[2]); alive.fill(false);

    // FMM
    while( !trial.empty() )
    {
        DistanceToPoint top = *trial.begin();
        trial.erase(trial.begin());
        iCoord v = top.second;
        alive(v[0],v[1],v[2])=true;

        unsigned int vor = voronoi(v[0],v[1],v[2]);
        real b1; if(biasFactor) b1=(real)(*biasFactor)(v[0],v[1],v[2]); else  b1=1.0;

        // update neighbors
        for (unsigned int i=0; i<nbOffset; i++)
        {
            // update distance on neighbors using their own neighbors
            iCoord v2 = v + offset[i];
            if(v2[0]>=0) if(v2[1]>=0) if(v2[2]>=0) if(v2[0]<dim[0]) if(v2[1]<dim[1]) if(v2[2]<dim[2])
                if(!alive(v2[0],v2[1],v2[2]))
                {
                    // get neighboring alive values
                    iCoord v3=v2;
                    Dist d,d2;
                    for (unsigned int j=0; j<3; j++)
                    {
                        v3[j]--;  if(v3[j]>=0 && alive(v3[0],v3[1],v3[2])) d[2*j]= distances(v3[0],v3[1],v3[2]); else d[2*j]=-1;
                        v3[j]--;  if(v3[j]>=0 && alive(v3[0],v3[1],v3[2])) d2[2*j]= distances(v3[0],v3[1],v3[2]); else d2[2*j]=-1;
                        v3[j]+=3; if(v3[j]<dim[j] && alive(v3[0],v3[1],v3[2])) d[2*j+1]= distances(v3[0],v3[1],v3[2]); else d[2*j+1]=-1;
                        v3[j]++;  if(v3[j]<dim[j] && alive(v3[0],v3[1],v3[2])) d2[2*j+1]= distances(v3[0],v3[1],v3[2]); else d2[2*j+1]=-1;
                        v3[j]-=2;
                    }
                    real b2; if(biasFactor) b2=(real)(*biasFactor)(v2[0],v2[1],v2[2]); else  b2=1.0;
                    real newDist = Eikonal<real>(d,d2,voxelsize,sofa::helper::rmin(b1,b2));
                    real oldDist = distances(v2[0],v2[1],v2[2]);
                    if(oldDist>newDist)
                    {
                        typename std::set<DistanceToPoint>::iterator it=trial.find(DistanceToPoint(oldDist,v2)); if(it!=trial.end()) trial.erase(it);
                        voronoi(v2[0],v2[1],v2[2])=vor;
                        distances(v2[0],v2[1],v2[2])=newDist;
                        trial.insert( DistanceToPoint(newDist,v2) );
                    }
                }
        }
    }
}


/**
* Update geodesic distances in the image given a bias distance function b(x).
* This is equivalent to solve for the eikonal equation || grad d(x) || = 1/b(x) with d(p)=0 at @param pos
* using dijkstra minimum path algorithm
* distances should be intialized (<0 outside the object, >=0 inside, and = 0 for seeds)
* returns @param voronoi and @param distances
*/



template<typename real,typename T>
void dijkstra (std::set<std::pair<real,sofa::defaulttype::Vec<3,int> > > &trial, cimg_library::CImg<real>& distances, cimg_library::CImg<unsigned int>& voronoi,
               const sofa::defaulttype::Vec<3,real>& voxelsize, const cimg_library::CImg<T>* biasFactor=NULL)
{
    typedef sofa::defaulttype::Vec<3,int> iCoord;
    typedef std::pair<real,iCoord > DistanceToPoint;
    const iCoord dim(distances.width(),distances.height(),distances.depth());

    //CImg<bool> alive(dim[0],dim[1],dim[2]); alive.fill(false);

    // init
    sofa::defaulttype::Vec<27,  iCoord > offset; // image coord offsets related to neighbors
    sofa::defaulttype::Vec<27,  real > lD;      // precomputed local distances (supposing that the transformation is linear)
    int count=0; for (int k=-1; k<=1; k++) for (int j=-1; j<=1; j++) for (int i=-1; i<=1; i++)
    {
        offset[count]= iCoord(i,j,k);
        lD[count]= (voxelsize.linearProduct(offset[count])).norm();
        count++;
    }
    unsigned int nbOffset=offset.size();

    // dijkstra
    while( !trial.empty() )
    {
        DistanceToPoint top = *trial.begin();
        trial.erase(trial.begin());
        iCoord v = top.second;
        //alive(v[0],v[1],v[2])=true;

        unsigned int vor = voronoi(v[0],v[1],v[2]);
        real b1; if(biasFactor) b1=(real)(*biasFactor)(v[0],v[1],v[2]); else  b1=1.0;

        for (unsigned int i=0; i<nbOffset; i++)
        {
            iCoord v2 = v + offset[i];
            if(v2[0]>=0) if(v2[1]>=0) if(v2[2]>=0) if(v2[0]<dim[0]) if(v2[1]<dim[1]) if(v2[2]<dim[2])
                //if(!alive(v2[0],v2[1],v2[2]))
            {
                real b2; if(biasFactor) b2=(real)(*biasFactor)(v2[0],v2[1],v2[2]); else  b2=1.0;
                real newDist = distances(v[0],v[1],v[2]) + lD[i]*1.0/sofa::helper::rmin(b1,b2);
                real oldDist = distances(v2[0],v2[1],v2[2]);
                if(oldDist>newDist)
                {
                    typename std::set<DistanceToPoint>::iterator it=trial.find(DistanceToPoint(oldDist,v2)); if(it!=trial.end()) trial.erase(it);
                    voronoi(v2[0],v2[1],v2[2]) = vor;
                    distances(v2[0],v2[1],v2[2]) = newDist;
                    trial.insert( DistanceToPoint(newDist,v2) );
                }
            }
        }
    }
}

///@brief Compute norm L2 of a pixel in a CImg
template<typename real>
real norm(cimg_library::CImg<real>& distances, sofa::helper::fixed_array<int, 3>& coord)
{
    return sqrt(pow(distances(coord[0],coord[1],coord[2],0),2) +
            pow(distances(coord[0],coord[1],coord[2],1),2) +
            pow(distances(coord[0],coord[1],coord[2],2),2));
}

/// @brief Replace value at oldCoord with a combinaison of value at newCoord, a offset and a bias if provided
template<typename real,typename T>
void replace(cimg_library::CImg<unsigned int>& voronoi, cimg_library::CImg<real>& distances, sofa::helper::fixed_array<int, 3>& oldCoord, sofa::helper::fixed_array<int, 3>& newCoord,
             sofa::helper::fixed_array<real, 3>& offset, const sofa::helper::fixed_array<real, 3>& voxelSize, const cimg_library::CImg<T>* bias)
{
    real b=1.0;
    if(bias)
        b=std::min((*bias)(oldCoord[0], oldCoord[1], oldCoord[2]), (*bias)(newCoord[0], newCoord[1], newCoord[2]));
    distances(oldCoord[0], oldCoord[1], oldCoord[2], 0) = distances(newCoord[0], newCoord[1], newCoord[2], 0) +  offset[0]*voxelSize[0]/b;
    distances(oldCoord[0], oldCoord[1], oldCoord[2], 1) = distances(newCoord[0], newCoord[1], newCoord[2], 1) +  offset[1]*voxelSize[1]/b;
    distances(oldCoord[0], oldCoord[1], oldCoord[2], 2) = distances(newCoord[0], newCoord[1], newCoord[2], 2) +  offset[2]*voxelSize[2]/b;
    voronoi(oldCoord[0], oldCoord[1], oldCoord[2]) = voronoi(newCoord[0], newCoord[1], newCoord[2]);
}

/// @brief Update value of the pixel of an image after comparing it with its neighbor
template<typename real,typename T>
void update(cimg_library::CImg<real>& distances, cimg_library::CImg<unsigned int>& voronoi, sofa::helper::fixed_array< sofa::helper::fixed_array<int, 3>, 10 >& coord, sofa::helper::fixed_array< sofa::helper::fixed_array<real, 3>, 10 >& offset, const sofa::helper::fixed_array<real, 3>& voxelSize, const cimg_library::CImg<T>* bias)
{
    real l_curr=norm(distances,coord[0]);
    for(int l=1; l<=9; ++l)
    {
        real l_neigh=norm(distances,coord[l]);
        if(l_neigh<l_curr){ replace(voronoi,distances,coord[0],coord[l],offset[l], voxelSize, bias); l_curr=l_neigh; }
    }
}

/// @brief Compare two images, pixel per pixel
/// @return true if for each pixel the error is bounded by a threshold, false otherwise.
template<typename real>
bool hasConverged(cimg_library::CImg<real>& previous, cimg_library::CImg<real>& current, SReal tolerance)
{
    bool result=true;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<previous.width(); ++i) for(int j=0; j<previous.height(); ++j) for(int k=0; k<previous.depth(); ++k)
    {
        if( !isnan(previous(i,j,k,0)) && !isnan(current(i,j,k,0)) )
        {
            SReal error = sqrt( pow(previous(i,j,k,0)-current(i,j,k,0),2) +
                                pow(previous(i,j,k,1)-current(i,j,k,1),2) +
                                pow(previous(i,j,k,2)-current(i,j,k,2),2));
            if(error>tolerance)
                result = false;
        }
    }
    return result;
}

/// @brief Perform a raster scan from left to right to update distances
template<typename real,typename T>
void left(cimg_library::CImg<unsigned int>& v, cimg_library::CImg<real>& d, const sofa::helper::fixed_array<real, 3>& vx, const cimg_library::CImg<T>* bias)
{
    for(int i=d.width()-2; i>=0; --i)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int j=d.height()-2; j>=1; --j)
        {
            for(int k=d.depth()-2; k>=1; --k)
            {
                sofa::helper::fixed_array< sofa::helper::fixed_array<int, 3>, 10 > c;
                sofa::helper::fixed_array< sofa::helper::fixed_array<real, 3>, 10 > o;
                c[0] = sofa::helper::fixed_array<int, 3>(i,j,k); o[0] = sofa::helper::fixed_array<real, 3>(0,0,0); int count=1;
                for(int y=-1;y<=1; ++y) for(int z=-1; z<=1; z++)
                {
                    c[count] = sofa::helper::fixed_array<int, 3>(i+1,j+y,k+z);
                    o[count] = sofa::helper::fixed_array<real, 3>(1,std::abs(y),std::abs(z)); count++;
                }
                update(d,v,c,o,vx, bias);
            }
        }
    }
}

/// @brief Perform a raster scan from right to left to update distances
template<typename real,typename T>
void right(cimg_library::CImg<unsigned int>& v, cimg_library::CImg<real>& d, const sofa::helper::fixed_array<real, 3>& vx, const cimg_library::CImg<T>* bias)
{
    for(int i=1; i<d.width(); ++i)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int j=1; j<d.height()-1; ++j)
        {
            for(int k=1; k<d.depth()-1; ++k)
            {
                sofa::helper::fixed_array< sofa::helper::fixed_array<int, 3>, 10 > c;
                sofa::helper::fixed_array< sofa::helper::fixed_array<real, 3>, 10 > o;
                c[0] = sofa::helper::fixed_array<int, 3>(i,j,k); o[0] = sofa::helper::fixed_array<real, 3>(0,0,0); int count=1;
                for(int y=-1;y<=1; ++y) for(int z=-1; z<=1; z++)
                {
                    c[count] = sofa::helper::fixed_array<int, 3>(i-1,j+y,k+z);
                    o[count] = sofa::helper::fixed_array<real, 3>(1,std::abs(y),std::abs(z)); count++;
                }
                update(d,v,c,o,vx, bias);
            }
        }
    }
}

/// @brief Perform a raster scan from down to up to update distances
template<typename real,typename T>
void down(cimg_library::CImg<unsigned int>& v, cimg_library::CImg<real>& d, const sofa::helper::fixed_array<real, 3>& vx, const cimg_library::CImg<T>* bias)
{
    for(int j=d.height()-2; j>=0; --j)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=d.width()-2; i>=1; --i)
        {
            for(int k=d.depth()-2; k>=1; --k)
            {
                sofa::helper::fixed_array< sofa::helper::fixed_array<int, 3>, 10 > c;
                sofa::helper::fixed_array< sofa::helper::fixed_array<real, 3>, 10 > o;
                c[0] = sofa::helper::fixed_array<int, 3>(i,j,k); o[0] = sofa::helper::fixed_array<real, 3>(0,0,0); int count=1;
                for(int x=-1;x<=1; ++x) for(int z=-1; z<=1; z++)
                {
                    c[count] = sofa::helper::fixed_array<int, 3>(i+x,j+1,k+z);
                    o[count] = sofa::helper::fixed_array<real, 3>(std::abs(x),1,std::abs(z)); count++;
                }
                update(d,v,c,o,vx, bias);
            }
        }
    }
}

/// @brief Perform a raster scan from up to down to update distances
template<typename real,typename T>
void up(cimg_library::CImg<unsigned int>& v, cimg_library::CImg<real>& d, const sofa::helper::fixed_array<real, 3>& vx, const cimg_library::CImg<T>* bias)
{
    for(int j=1; j<d.height(); ++j)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=1; i<d.width()-1; ++i)
        {
            for(int k=1; k<d.depth()-1; ++k)
            {
                sofa::helper::fixed_array< sofa::helper::fixed_array<int, 3>, 10 > c;
                sofa::helper::fixed_array< sofa::helper::fixed_array<real, 3>, 10 > o;
                c[0] = sofa::helper::fixed_array<int, 3>(i,j,k); o[0] = sofa::helper::fixed_array<real, 3>(0,0,0); int count=1;
                for(int x=-1;x<=1; ++x) for(int z=-1; z<=1; z++)
                {
                    c[count] = sofa::helper::fixed_array<int, 3>(i+x,j-1,k+z);
                    o[count] = sofa::helper::fixed_array<real, 3>(std::abs(x),1,std::abs(z)); count++;
                }
                update(d,v,c,o,vx, bias);
            }
        }
    }
}

/// @brief Perform a raster scan from backward to forward to update distances
template<typename real,typename T>
void backward(cimg_library::CImg<unsigned int>& v, cimg_library::CImg<real>& d, const sofa::helper::fixed_array<real, 3>& vx, const cimg_library::CImg<T>* bias)
{
    for(int k=d.depth()-2; k>=0; --k)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=d.width()-2; i>=1; --i)
        {
            for(int j=d.height()-2; j>=1; --j)
            {
                sofa::helper::fixed_array< sofa::helper::fixed_array<int, 3>, 10 > c;
                sofa::helper::fixed_array< sofa::helper::fixed_array<real, 3>, 10 > o;
                c[0] = sofa::helper::fixed_array<int, 3>(i,j,k); o[0] = sofa::helper::fixed_array<real, 3>(0,0,0); int count=1;
                for(int x=-1;x<=1; ++x) for(int y=-1; y<=1; y++)
                {
                    c[count] = sofa::helper::fixed_array<int, 3>(i+x,j+y,k+1);
                    o[count] = sofa::helper::fixed_array<real, 3>(std::abs(x),std::abs(y),1); count++;
                }
                update(d,v,c,o,vx,bias);
            }
        }
    }
}

/// @brief Perform a raster scan from forward to backward to update distances
template<typename real,typename T>
void forward(cimg_library::CImg<unsigned int>& v, cimg_library::CImg<real>& d, const sofa::helper::fixed_array<real, 3>& vx, const cimg_library::CImg<T>* bias)
{
    for(int k=1; k<d.depth(); ++k)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=1; i<d.width()-1; ++i)
        {
            for(int j=1; j<d.height()-1; ++j)
            {
                sofa::helper::fixed_array< sofa::helper::fixed_array<int, 3>, 10 > c;
                sofa::helper::fixed_array< sofa::helper::fixed_array<real, 3>, 10 > o;
                c[0] = sofa::helper::fixed_array<int, 3>(i,j,k); o[0] = sofa::helper::fixed_array<real, 3>(0,0,0); int count=1;
                for(int x=-1;x<=1; ++x) for(int y=-1; y<=1; y++)
                {
                    c[count] = sofa::helper::fixed_array<int, 3>(i+x,j+y,k-1);
                    o[count] = sofa::helper::fixed_array<real, 3>(std::abs(x),std::abs(y),1); count++;
                }
                update(d,v,c,o,vx,bias);
            }
        }
    }
}

/// @brief Perform 6 raster scan of an image to fully cover it.
template<typename real,typename T>
void rasterScan(cimg_library::CImg<unsigned int>& voronoi, cimg_library::CImg<real>& distances, const sofa::helper::fixed_array<real, 3>& voxelSize, const cimg_library::CImg<T>* biasFactor=NULL)
{
    right(voronoi, distances, voxelSize, biasFactor);
    left(voronoi, distances, voxelSize, biasFactor);
    down(voronoi, distances, voxelSize,biasFactor);
    up(voronoi, distances, voxelSize,biasFactor);
    forward(voronoi, distances, voxelSize,biasFactor);
    backward(voronoi, distances, voxelSize,biasFactor);
}

/// @brief Update geodesic distances in the image given a bias distance function b(x).
/// using Parallel Marching Method (PMM) from Ofir Weber & .al (https://ssl.lu.usi.ch/entityws/Allegati/pdf_pub5153.pdf).
/// The implementation works with openMP. Due to data dependency it may quite slow compared to a sequential algorithm because it requires many iterations to converge.
/// In specific cases it can be very efficient (convex domain) because only one iteration is required. A GPU implementation is possible and is on the todo list.
/// @param maxIter should be carefully chosen to minimize computation time.
/// @param tolerance should be carefully chosen to minimize computation time.
/// @returns @param voronoi and @param distances
template<typename real,typename T>
void parallelMarching(cimg_library::CImg<real>& distances, cimg_library::CImg<unsigned int>& voronoi, const sofa::helper::fixed_array<real, 3>& voxelSize, const unsigned int maxIter=std::numeric_limits<unsigned int>::max(), const SReal tolerance=10, const cimg_library::CImg<T>* biasFactor=NULL)
{
    if(distances.width()<3 || distances.height()<3 || distances.depth()<3)
    {
        std::cerr << "ImageAlgorithms::parallelMarching : Boundary conditions are not treated so size (width,height,depth) should be >=3. (Work in Progress)" << std::endl;
        return;
    }
    //Build a new distance image from distances.
    cimg_library::CImg<real> v_distances(distances.width(), distances.height(), distances.depth(), 3, std::numeric_limits<real>::max());
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<distances.width(); ++i) for(int j=0; j<distances.height(); ++j) for(int k=0; k<distances.depth(); ++k)
    {
        if( distances(i,j,k,0) < 0 )
            v_distances(i,j,k,0) = v_distances(i,j,k,1) = v_distances(i,j,k,2) = std::numeric_limits<real>::signaling_NaN();
        else
            v_distances(i,j,k,0) = v_distances(i,j,k,1) = v_distances(i,j,k,2) = distances(i,j,k,0);
    }

    //Perform raster scan until convergence
    bool converged = false; unsigned int iter_count = 0; cimg_library::CImg<real> prev_distances;
    while( (converged==false) || (iter_count<maxIter) )
    {
        prev_distances = v_distances; iter_count++;
        rasterScan(voronoi, v_distances, voxelSize, biasFactor);
        converged = hasConverged(prev_distances, v_distances, tolerance);
    }

    //Update distances with v_distances
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<distances.width(); ++i) for(int j=0; j<distances.height(); ++j) for(int k=0; k<distances.depth(); ++k)
    {
        if( isnan(v_distances(i,j,k,0)) )
            distances(i,j,k,0) = -1.0;
        else
            distances(i,j,k,0) = std::sqrt( std::pow(v_distances(i,j,k,0),2) + std::pow(v_distances(i,j,k,1),2) + std::pow(v_distances(i,j,k,2),2) );
    }
}

/**
* Initialize null distances and voronoi value (=point index) from a position in image coordinates
* and returns list of seed (=trial) points to be used in dijkstra or fast marching algorithms
*/

template<typename real>
void AddSeedPoint (std::set<std::pair<real,sofa::defaulttype::Vec<3,int> > >& trial, cimg_library::CImg<real>& distances, cimg_library::CImg<unsigned int>& voronoi,
                   const sofa::defaulttype::Vec<3,real>& pos,  const unsigned int index)
{
    typedef sofa::defaulttype::Vec<3,int> iCoord;
    typedef std::pair<real,iCoord > DistanceToPoint;

    iCoord P;  for (unsigned int j=0; j<3; j++)  P[j]=sofa::helper::round(pos[j]);
    if(distances.containsXYZC(P[0],P[1],P[2]))
        if(distances(P[0],P[1],P[2])>=0)
        {
            distances(P[0],P[1],P[2])=0;
            voronoi(P[0],P[1],P[2])=index;
            trial.insert( DistanceToPoint(0.,iCoord(P[0],P[1],P[2])) );
        }
}


#undef isnan

#endif // IMAGEALGORITHMS_H
