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
#ifndef FLEXIBLE_VoronoiShapeFunction_H
#define FLEXIBLE_VoronoiShapeFunction_H

#include "../initFlexible.h"
#include "../shapeFunction/BaseShapeFunction.h"
#include "../shapeFunction/BaseImageShapeFunction.h"
#include "../types/PolynomialBasis.h"

#include <image/ImageTypes.h>
#include <image/ImageAlgorithms.h>
#include <image/BranchingImage.h>

#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>

#define DISTANCE 0
#define LAPLACE 1
#define SIBSON 2

namespace sofa
{
namespace component
{
namespace shapefunction
{

using sofa::helper::round;
using core::behavior::BaseShapeFunction;
using defaulttype::Mat;
using defaulttype::Vec;

/**
Voronoi shape functions are natural neighbor interpolants
there are computed from an image (typically a rasterized object)
  */



/// Default implementation does not compile
template <int imageTypeLabel>
struct VoronoiShapeFunctionSpecialization
{
};

template<class Real>
struct NaturalNeighborData
{
    Real vol;
    Real surf;
    Real dist;
    NaturalNeighborData():vol(0),surf(0),dist(std::numeric_limits<Real>::max()) {}
    typedef std::map<unsigned int, NaturalNeighborData<Real> > Map;
};


/// Specialization for regular Image
template <>
struct VoronoiShapeFunctionSpecialization<defaulttype::IMAGELABEL_IMAGE>
{
    template<class VoronoiShapeFunction>
    static void init(VoronoiShapeFunction* This)
    {
        typedef typename VoronoiShapeFunction::ImageTypes ImageTypes;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        // retrieve data
        raImage in(This->image);
        if(in->isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        const typename ImageTypes::CImgT& inimg = in->getCImg(0);  // suppose time=0

        // init voronoi and distances
        typename VoronoiShapeFunction::imCoord dim = in->getDimensions(); dim[ImageTypes::DIMENSION_S]=dim[ImageTypes::DIMENSION_T]=1;

        waInd vorData(This->f_voronoi); vorData->setDimensions(dim);
        typename IndTypes::CImgT& voronoi = vorData->getCImg(); voronoi.fill(0);

        waDist distData(This->f_distances);         distData->setDimensions(dim);
        typename DistTypes::CImgT& dist = distData->getCImg(); dist.fill(-1);
        cimg_forXYZC(inimg,x,y,z,c) if(inimg(x,y,z,c)) dist(x,y,z)=cimg::type<DistT>::max();

        // init indices and weights images
        unsigned int nbref=This->f_nbRef.getValue();        dim[ImageTypes::DIMENSION_S]=nbref;

        waInd indData(This->f_index); indData->setDimensions(dim);
        typename IndTypes::CImgT& indices = indData->getCImg(); indices.fill(0);

        waDist weightData(This->f_w);         weightData->setDimensions(dim);
        typename DistTypes::CImgT& weights = weightData->getCImg(); weights.fill(0);
    }


    template<class VoronoiShapeFunction>
    static void computeVoronoi(VoronoiShapeFunction* This)
    {
        typedef typename VoronoiShapeFunction::ImageTypes ImageTypes;
        typedef typename VoronoiShapeFunction::T T;
        typedef typename VoronoiShapeFunction::Coord Coord;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::raTransform raTransform;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        typedef Vec<3,int> iCoord;
        typedef std::pair<DistT,iCoord > DistanceToPoint;

        // retrieve data
        raImage in(This->image);
        raTransform inT(This->transform);
        if(in->isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        const typename ImageTypes::CImgT& inimg = in->getCImg(0);  // suppose time=0
        const typename ImageTypes::CImgT* biasFactor=This->biasDistances.getValue()?&inimg:NULL;

        waInd vorData(This->f_voronoi);         typename IndTypes::CImgT& voronoi = vorData->getCImg();
        waDist distData(This->f_distances);     typename DistTypes::CImgT& dist = distData->getCImg();

        helper::ReadAccessor<Data<vector<Coord> > > parent(This->f_position);
        if(!parent.size()) { This->serr<<"Parent nodes not found"<<This->sendl; return; }
        vector<iCoord> parentiCoord;        for(unsigned int i=0; i<parent.size(); i++) { Coord p = inT->toImageInt(parent[i]);  parentiCoord.push_back(iCoord(p[0],p[1],p[2])); }

        // compute voronoi and distances based on nodes
        std::set<DistanceToPoint> trial;                // list of seed points
        for(unsigned int i=0; i<parent.size(); i++) AddSeedPoint<DistT>(trial,dist,voronoi, parentiCoord[i],i+1);

        if(This->useDijkstra.getValue()) dijkstra<DistT,T>(trial,dist, voronoi, inT->getScale() , biasFactor);
        else fastMarching<DistT,T>(trial,dist, voronoi, inT->getScale() ,biasFactor );
    }


    template<class VoronoiShapeFunction>
    static void ComputeWeigths_DistanceRatio(VoronoiShapeFunction* This)
    {
        typedef typename VoronoiShapeFunction::ImageTypes ImageTypes;
        typedef typename VoronoiShapeFunction::T T;
        typedef typename VoronoiShapeFunction::Coord Coord;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::raTransform raTransform;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        typedef Vec<3,int> iCoord;
        typedef std::pair<DistT,iCoord > DistanceToPoint;

        // retrieve data
        raImage in(This->image);
        raTransform inT(This->transform);
        if(in->isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        const typename ImageTypes::CImgT& inimg = in->getCImg(0);  // suppose time=0
        const typename ImageTypes::CImgT* biasFactor=This->biasDistances.getValue()?&inimg:NULL;

        waInd vorData(This->f_voronoi);         typename IndTypes::CImgT& voronoi = vorData->getCImg();
        waDist distData(This->f_distances);     typename DistTypes::CImgT& dist = distData->getCImg();
        waInd indData(This->f_index);           typename IndTypes::CImgT& indices = indData->getCImg();
        waDist weightData(This->f_w);           typename DistTypes::CImgT& weights = weightData->getCImg();

        helper::ReadAccessor<Data<vector<Coord> > > parent(This->f_position);
        if(!parent.size()) { This->serr<<"Parent nodes not found"<<This->sendl; return; }
        vector<iCoord> parentiCoord;        for(unsigned int i=0; i<parent.size(); i++) { Coord p = inT->toImageInt(parent[i]);  parentiCoord.push_back(iCoord(p[0],p[1],p[2])); }

        unsigned int nbref=This->f_nbRef.getValue();

        // compute weight of each parent
        for(unsigned int i=0; i<parentiCoord.size(); i++)
        {
            std::set<DistanceToPoint> trial;                // list of seed points
            DistT dmax=0;
            cimg_forXYZ(voronoi,x,y,z) if(voronoi(x,y,z)==i+1) if(dmax<dist(x,y,z)) dmax=dist(x,y,z);

            // distances from voronoi border
            typename DistTypes::CImgT distB=dist;  cimg_foroff(distB,off) if(distB[off]!=-1) distB[off]=dmax;
            typename IndTypes::CImgT voronoiP=voronoi;
            cimg_forXYZ(voronoi,x,y,z) if(voronoi(x,y,z)==i+1)
            {
                bool border=false;  if(x!=0 && voronoi(x-1,y,z)!=i+1  && voronoi(x-1,y,z)!=0) border=true; else if(x!=voronoi.width()-1 && voronoi(x+1,y,z)!=i+1 && voronoi(x+1,y,z)!=0) border=true; else if(y!=0 && voronoi(x,y-1,z)!=i+1 && voronoi(x,y-1,z)!=0) border=true; else if(y!=voronoi.height()-1 && voronoi(x,y+1,z)!=i+1 && voronoi(x,y+1,z)!=0) border=true; else if(z!=0 && voronoi(x,y,z-1)!=i+1 && voronoi(x,y,z-1)!=0) border=true; else if(z!=voronoi.depth()-1 && voronoi(x,y,z+1)!=i+1 && voronoi(x,y,z+1)!=0) border=true;
                if(border)
                {
                    distB(x,y,z)=0;
                    trial.insert( DistanceToPoint(0.,iCoord(x,y,z)) );
                }
            }
            if(This->useDijkstra.getValue()) dijkstra<DistT,T>(trial,distB, voronoiP, inT->getScale() , biasFactor); else fastMarching<DistT,T>(trial,distB, voronoiP, inT->getScale() ,biasFactor );

            // extend voronoi to 2*dmax
            dmax*=(DistT)2.;
            typename DistTypes::CImgT distP=dist;  cimg_foroff(distP,off) if(distP[off]!=-1) distP[off]=dmax;

            AddSeedPoint<DistT>(trial,distP,voronoiP, parentiCoord[i],i+1);
            if(This->useDijkstra.getValue()) dijkstra<DistT,T>(trial,distP, voronoiP, inT->getScale() , biasFactor); else fastMarching<DistT,T>(trial,distP, voronoiP, inT->getScale() ,biasFactor );

            // compute weight as distance ratio
            cimg_forXYZ(voronoiP,x,y,z) if(voronoiP(x,y,z)==i+1)
            {
                DistT w;
                DistT db=distB(x,y,z),dp=distP(x,y,z);

                if(dp==0) w=(DistT)1.;
                else if(voronoi(x,y,z)==i+1) w=(DistT)0.5*((DistT)1. + db/(dp+db)); // inside voronoi: dist(frame,closestVoronoiBorder)=d+disttovoronoi
                else if(dp==db) w=(DistT)0.;
                else w=(DistT)0.5*((DistT)1. - db/(dp-db)); // outside voronoi: dist(frame,closestVoronoiBorder)=d-disttovoronoi
                if(w<0) w=0; else if(w>(DistT)1.) w=(DistT)1.;

                // insert in weights
                unsigned int j=0;
                while(j!=nbref && weights(x,y,z,j)>=w) j++;
                if(j!=nbref)
                {
                    if(j!=nbref-1) for(unsigned int k=nbref-1; k>j; k--) { weights(x,y,z,k)=weights(x,y,z,k-1); indices(x,y,z,k)=indices(x,y,z,k-1); }
                    weights(x,y,z,j)=w;
                    indices(x,y,z,j)=i+1;
                }
            }
        }
        // normalize
        cimg_forXYZ(voronoi,x,y,z) if(voronoi(x,y,z))
        {
            DistT totW=0;
            cimg_forC(weights,c) totW+=weights(x,y,z,c);
            if(totW) cimg_forC(weights,c) weights(x,y,z,c)/=totW;
        }
    }

    /**
    * returns Natural Neighbor Interpolant coordinates of a point @param index: http://dilbert.engr.ucdavis.edu/~suku/nem/
    * from :
    *   - initial nodal voronoi regions (@param voronoi and @param distances)
    *   - the updated voronoi including the point (@param voronoiPt and @param distancesPt)
    * returns volume, area and distance associated to each natural neighbor (indexed in @param ref)
    */

    template<class VoronoiShapeFunction>
    static void ComputeWeigths_NaturalNeighbors(VoronoiShapeFunction* This)
    {
        typedef typename VoronoiShapeFunction::ImageTypes ImageTypes;
        typedef typename VoronoiShapeFunction::T T;
        typedef typename VoronoiShapeFunction::Real Real;
        typedef typename VoronoiShapeFunction::Coord Coord;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::raTransform raTransform;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        typedef Vec<3,int> iCoord;
        typedef std::pair<DistT,iCoord > DistanceToPoint;

        typedef NaturalNeighborData<Real> NNData;
        typedef typename NNData::Map NNMap;

        // retrieve data
        raImage in(This->image);
        raTransform inT(This->transform);
        if(in->isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        const typename ImageTypes::CImgT& inimg = in->getCImg(0);  // suppose time=0
        const typename ImageTypes::CImgT* biasFactor=This->biasDistances.getValue()?&inimg:NULL;

        waInd vorData(This->f_voronoi);         typename IndTypes::CImgT& voronoi = vorData->getCImg();
        waDist distData(This->f_distances);     typename DistTypes::CImgT& dist = distData->getCImg();
        waInd indData(This->f_index);           typename IndTypes::CImgT& indices = indData->getCImg();
        waDist weightData(This->f_w);           typename DistTypes::CImgT& weights = weightData->getCImg();

        unsigned int nbref=This->f_nbRef.getValue();

        Coord voxelsize(inT->getScale());
        Real pixelvol=voxelsize[0]*voxelsize[1]*voxelsize[2];
        Vec<3,Real> pixelsurf(voxelsize[1]*voxelsize[2],voxelsize[0]*voxelsize[2],voxelsize[0]*voxelsize[1]);
        unsigned int indexPt=This->f_position.getValue().size()+1; // voronoi index of points that will be added to compute NNI

        // compute weights voxel-by-voxel
        cimg_forXYZ(voronoi,xi,yi,zi) if(voronoi(xi,yi,zi))
        {
            // compute updated voronoi including voxel (xi,yi,iz)
            std::set<DistanceToPoint> trial;                // list of seed points

            typename IndTypes::CImgT voronoiPt=voronoi;
            typename DistTypes::CImgT distPt=dist;

            AddSeedPoint<DistT>(trial,distPt,voronoiPt, iCoord(xi,yi,zi),indexPt);

            if(This->useDijkstra.getValue()) dijkstra<DistT,T>(trial,distPt, voronoiPt, voxelsize , biasFactor); else fastMarching<DistT,T>(trial,distPt, voronoiPt, voxelsize,biasFactor );

            // compute Natural Neighbor Data based on neighboring voronoi cells
            NNMap dat;
            //bool border;
            cimg_forXYZ(voronoiPt,x,y,z) if(voronoiPt(x,y,z)==indexPt)
            {
                unsigned int node=voronoi(x,y,z);
                if(!dat.count(node)) dat[node]=NNData();
                dat[node].vol+=pixelvol;
                if(x!=0)                    if(voronoiPt(x-1,y,z)!=indexPt) dat[node].surf+=pixelsurf[0];
                if(x!=voronoiPt.width()-1)  if(voronoiPt(x+1,y,z)!=indexPt) dat[node].surf+=pixelsurf[0];
                if(y!=0)                    if(voronoiPt(x,y-1,z)!=indexPt) dat[node].surf+=pixelsurf[1];
                if(y!=voronoiPt.height()-1) if(voronoiPt(x,y+1,z)!=indexPt) dat[node].surf+=pixelsurf[1];
                if(z!=0)                    if(voronoiPt(x,y,z-1)!=indexPt) dat[node].surf+=pixelsurf[2];
                if(z!=voronoiPt.depth()-1)  if(voronoiPt(x,y,z+1)!=indexPt) dat[node].surf+=pixelsurf[2];
                if(distPt(x,y,z)+dist(x,y,z)<dat[node].dist) dat[node].dist=distPt(x,y,z)+dist(x,y,z);
            }

            if (This->method.getValue().getSelectedId() == LAPLACE)   // replace vol (SIBSON) by surf/dist coordinates (LAPLACE)
            {
                for ( typename NNMap::iterator it=dat.begin() ; it != dat.end(); it++ )
                    if((*it).second.dist==0) (*it).second.vol=std::numeric_limits<Real>::max();
                    else (*it).second.vol=(*it).second.surf/(*it).second.dist;
            }

            // prune to nbref if necessary (nb of natural neighbors >nbref)
            while(dat.size()>nbref)
            {
                Real vmin=std::numeric_limits<Real>::max(); unsigned int key=0;
                for ( typename NNMap::iterator it=dat.begin() ; it != dat.end(); it++ ) if((*it).second.vol<vmin) key=(*it).first;
                dat.erase(key);
            }

            // compute weights based on Natural Neighbor coordinates
            Real total=0;
            for ( typename NNMap::iterator it=dat.begin() ; it != dat.end(); it++ ) total+=(*it).second.vol;
            if(total)
            {
                int count=0;
                for ( typename NNMap::iterator it=dat.begin() ; it != dat.end(); it++ )
                {
                    weights(xi,yi,zi,count)=(*it).second.vol/total;
                    indices(xi,yi,zi,count)=(*it).first;
                    count++;
                }
            }
        }
    }

};






/// Specialization for branching Image
template <>
struct VoronoiShapeFunctionSpecialization<defaulttype::IMAGELABEL_BRANCHINGIMAGE>
{
    template<class VoronoiShapeFunction>
    static void init(VoronoiShapeFunction* This)
    {
        typedef typename VoronoiShapeFunction::ImageTypes ImageTypes;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        // retrieve data
        raImage inData(This->image);    const ImageTypes& in = inData.ref();
        if(in.isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }

        // init voronoi and distances
        typename VoronoiShapeFunction::imCoord dim = in.getDimensions(); dim[ImageTypes::DIMENSION_S]=dim[ImageTypes::DIMENSION_T]=1;

        waInd vorData(This->f_voronoi);  IndTypes& voronoi = vorData.wref();
        voronoi.setDimensions(dim);
        voronoi.cloneTopology (in,0);

        waDist distData(This->f_distances);        DistTypes& dist = distData.wref();
        dist.setDimensions(dim);
        dist.cloneTopology (in,-1.0);
        bimg_forCVoffT(in,c,v,off1D,t) if(t==0 && c==0) if(in(off1D,v,c,t)) dist(off1D,v,c,0)=cimg::type<DistT>::max();

        // init indices and weights images
        unsigned int nbref=This->f_nbRef.getValue();        dim[ImageTypes::DIMENSION_S]=nbref;

        waInd indData(This->f_index); IndTypes& indices = indData.wref();
        indices.setDimensions(dim);
        indices.cloneTopology (in,0);

        waDist weightData(This->f_w);    DistTypes& weights = weightData.wref();
        weights.setDimensions(dim);
        weights.cloneTopology (in,0);
    }


    template<class VoronoiShapeFunction>
    static void computeVoronoi(VoronoiShapeFunction* This)
    {
        typedef typename VoronoiShapeFunction::ImageTypes ImageTypes;
        typedef typename VoronoiShapeFunction::T T;
        typedef typename VoronoiShapeFunction::Coord Coord;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::raTransform raTransform;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        typedef typename ImageTypes::VoxelIndex VoxelIndex;
        typedef std::pair<DistT,VoxelIndex > DistanceToPoint;

        // retrieve data
        raImage inData(This->image);    const ImageTypes& in = inData.ref();
        raTransform inT(This->transform);
        if(in.isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        const ImageTypes* biasFactor=This->biasDistances.getValue()?&in:NULL;

        waInd vorData(This->f_voronoi);            IndTypes& voronoi = vorData.wref();
        waDist distData(This->f_distances);        DistTypes& dist = distData.wref();

        helper::ReadAccessor<Data<vector<Coord> > > parent(This->f_position);
        if(!parent.size()) { This->serr<<"Parent nodes not found"<<This->sendl; return; }
        vector<VoxelIndex> parentiCoord;
        for(unsigned int i=0; i<parent.size(); i++)
        {
            Coord p = inT->toImageInt(parent[i]);
            parentiCoord.push_back(VoxelIndex(in.index3Dto1D(p[0],p[1],p[2]),0));
            if(This->f_cell.getValue().size()>i) if(This->f_cell.getValue()[i]>0) parentiCoord.back().offset=This->f_cell.getValue()[i]-1;
        }

        // compute voronoi and distances based on nodes
        std::set<DistanceToPoint> trial;                // list of seed points
        for(unsigned int i=0; i<parent.size(); i++)            AddSeedPoint<DistT>(trial,dist,voronoi, parentiCoord[i],i+1);

        if(This->useDijkstra.getValue()) dijkstra<DistT,T>(trial,dist, voronoi, inT->getScale() , biasFactor); else fastMarching<DistT,T>(trial,dist, voronoi, inT->getScale() ,biasFactor );
    }


    template<class VoronoiShapeFunction>
    static void ComputeWeigths_DistanceRatio(VoronoiShapeFunction* This)
    {
        typedef typename VoronoiShapeFunction::ImageTypes ImageTypes;
        typedef typename VoronoiShapeFunction::T T;
        typedef typename VoronoiShapeFunction::Coord Coord;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::raTransform raTransform;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        typedef typename ImageTypes::VoxelIndex VoxelIndex;
        typedef typename ImageTypes::Neighbours Neighbours;
        typedef typename ImageTypes::NeighbourOffset NeighbourOffset;
        typedef std::pair<DistT,VoxelIndex > DistanceToPoint;

        // retrieve data
        raImage inData(This->image);    const ImageTypes& in = inData.ref();
        raTransform inT(This->transform);
        if(in.isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        const ImageTypes* biasFactor=This->biasDistances.getValue()?&in:NULL;

        waInd vorData(This->f_voronoi);            IndTypes& voronoi = vorData.wref();
        waDist distData(This->f_distances);        DistTypes& dist = distData.wref();
        waInd indData(This->f_index);              IndTypes& indices = indData.wref();
        waDist weightData(This->f_w);              DistTypes& weights = weightData.wref();

        helper::ReadAccessor<Data<vector<Coord> > > parent(This->f_position);
        if(!parent.size()) { This->serr<<"Parent nodes not found"<<This->sendl; return; }
        vector<VoxelIndex> parentiCoord;
        for(unsigned int i=0; i<parent.size(); i++)
        {
            Coord p = inT->toImageInt(parent[i]);
            parentiCoord.push_back(VoxelIndex(in.index3Dto1D(p[0],p[1],p[2]),0));
            if(This->f_cell.getValue().size()>i) if(This->f_cell.getValue()[i]>0) parentiCoord.back().offset=This->f_cell.getValue()[i]-1;
        }

        unsigned int nbref=This->f_nbRef.getValue();

        // compute weight of each parent
        for(unsigned int i=0; i<parentiCoord.size(); i++)
        {
            std::set<DistanceToPoint> trial;                // list of seed points
            DistT dmax=0;
            bimg_forCVoffT(voronoi,c,v,off1D,t) if(voronoi(off1D,v,c,t)==i+1) if(dmax<dist(off1D,v,c,t)) dmax=dist(off1D,v,c,t);

            // distances from voronoi border
            DistTypes distB(dist,false);  bimg_forCVoffT(distB,c,v,off1D,t) if(distB(off1D,v,c,t)!=-1)  distB(off1D,v,c,t)=dmax;
            IndTypes voronoiP(voronoi,false);
            bimg_forCVoffT(voronoi,c,v,off1D,t) if(voronoi(off1D,v,c,t)==i+1)
            {
                Neighbours neighbours = voronoi.getNeighbours(VoxelIndex(off1D,v));
                bool border=false;   for (unsigned int n=0; n<neighbours.size(); n++)
                    //if(voronoi.getDirection( off1D,neighbours[n].index1d ).connectionType()==NeighbourOffset::FACE)
                    if(voronoi(neighbours[n],c,t)!=0)  border=true;
                if(border)
                {
                    distB(off1D,v,c,t)=0;
                    trial.insert( DistanceToPoint(0.,VoxelIndex(off1D,v) ) );
                }
            }
            if(This->useDijkstra.getValue()) dijkstra<DistT,T>(trial,distB, voronoiP, inT->getScale() , biasFactor);            else fastMarching<DistT,T>(trial,distB, voronoiP, inT->getScale() ,biasFactor );

            // extend voronoi to 2*dmax
            dmax*=(DistT)2.;
            DistTypes distP(dist,false);  bimg_forCVoffT(distP,c,v,off1D,t) if(distP(off1D,v,c,t)!=-1)  distP(off1D,v,c,t)=dmax;

            AddSeedPoint<DistT>(trial,distP,voronoiP, parentiCoord[i],i+1);
            if(This->useDijkstra.getValue()) dijkstra<DistT,T>(trial,distP, voronoiP, inT->getScale() , biasFactor);            else fastMarching<DistT,T>(trial,distP, voronoiP, inT->getScale() ,biasFactor );

            // compute weight as distance ratio
            bimg_forCVoffT(voronoiP,c,v,off1D,t) if(voronoiP(off1D,v,c,t)==i+1)
            {
                DistT w;
                DistT db=distB(off1D,v,c,t),dp=distP(off1D,v,c,t);

                if(dp==0) w=(DistT)1.;
                else if(voronoi(off1D,v,c,t)==i+1) w=(DistT)0.5*((DistT)1. + db/(dp+db)); // inside voronoi: dist(frame,closestVoronoiBorder)=d+disttovoronoi
                else if(dp==db) w=(DistT)0.;
                else w=(DistT)0.5*((DistT)1. - db/(dp-db)); // outside voronoi: dist(frame,closestVoronoiBorder)=d-disttovoronoi
                if(w<0) w=0; else if(w>(DistT)1.) w=(DistT)1.;

                // insert in weights
                unsigned int j=0;
                while(j!=nbref && weights(off1D,v,j,t)>=w) j++;
                if(j!=nbref)
                {
                    if(j!=nbref-1) for(unsigned int k=nbref-1; k>j; k--) { weights(off1D,v,k,t)=weights(off1D,v,k-1,t); indices(off1D,v,k,t)=indices(off1D,v,k-1,t); }
                    weights(off1D,v,j,t)=w;
                    indices(off1D,v,j,t)=i+1;
                }
            }
        }
        // normalize
        bimg_forCVoffT(voronoi,c,v,off1D,t) if(voronoi(off1D,v,c,t))
        {
            DistT totW=0;
            bimg_forC(weights,c) totW+=weights(off1D,v,c,t);
            if(totW) bimg_forC(weights,c) weights(off1D,v,c,t)/=totW;
        }
    }


    /**
    * returns Natural Neighbor Interpolant coordinates of a point @param index: http://dilbert.engr.ucdavis.edu/~suku/nem/
    * from :
    *   - initial nodal voronoi regions (@param voronoi and @param distances)
    *   - the updated voronoi including the point (@param voronoiPt and @param distancesPt)
    * returns volume, area and distance associated to each natural neighbor (indexed in @param ref)
    */

    template<class VoronoiShapeFunction>
    static void ComputeWeigths_NaturalNeighbors(VoronoiShapeFunction* This)
    {
        typedef typename VoronoiShapeFunction::ImageTypes ImageTypes;
        typedef typename VoronoiShapeFunction::T T;
        typedef typename VoronoiShapeFunction::Real Real;
        typedef typename VoronoiShapeFunction::Coord Coord;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::raTransform raTransform;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        typedef typename ImageTypes::VoxelIndex VoxelIndex;
        typedef typename ImageTypes::Neighbours Neighbours;
        typedef typename ImageTypes::NeighbourOffset NeighbourOffset;
        typedef std::pair<DistT,VoxelIndex > DistanceToPoint;

        typedef NaturalNeighborData<Real> NNData;
        typedef typename NNData::Map NNMap;

        // retrieve data
        raImage inData(This->image);    const ImageTypes& in = inData.ref();
        raTransform inT(This->transform);
        if(in.isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        const ImageTypes* biasFactor=This->biasDistances.getValue()?&in:NULL;

        waInd vorData(This->f_voronoi);            IndTypes& voronoi = vorData.wref();
        waDist distData(This->f_distances);        DistTypes& dist = distData.wref();
        waInd indData(This->f_index);              IndTypes& indices = indData.wref();
        waDist weightData(This->f_w);              DistTypes& weights = weightData.wref();

        unsigned int nbref=This->f_nbRef.getValue();

        Coord voxelsize(inT->getScale());
        Real pixelvol=voxelsize[0]*voxelsize[1]*voxelsize[2];
        Vec<3,Real> pixelsurf(voxelsize[1]*voxelsize[2],voxelsize[0]*voxelsize[2],voxelsize[0]*voxelsize[1]);
        unsigned int indexPt=This->f_position.getValue().size()+1; // voronoi index of points that will be added to compute NNI

        IndTypes voronoiPt(voronoi,false);
        DistTypes distPt(dist,false);

        // compute weights voxel-by-voxel
        bimg_forCVoffT(voronoi,ci,vi,off1Di,ti) if(voronoi(off1Di,vi,ci,ti))
        {
            // compute updated voronoi including voxel (xi,yi,iz)
            std::set<DistanceToPoint> trial;                // list of seed points

            // copy
            bimg_forCVoffT(voronoiPt,c,v,off1D,t) voronoiPt(off1D,v,c,t)=voronoi(off1D,v,c,t);
            bimg_forCVoffT(distPt,c,v,off1D,t) distPt(off1D,v,c,t)=dist(off1D,v,c,t);

            AddSeedPoint<DistT>(trial,distPt,voronoiPt, VoxelIndex(off1Di,vi),indexPt);

            if(This->useDijkstra.getValue()) dijkstra<DistT,T>(trial,distPt, voronoiPt, voxelsize , biasFactor); else fastMarching<DistT,T>(trial,distPt, voronoiPt, voxelsize,biasFactor );

            // compute Natural Neighbor Data based on neighboring voronoi cells
            NNMap dat;
            bimg_forCVoffT(voronoiPt,c,v,off1D,t) if(voronoiPt(off1D,v,c,t)==indexPt)
            {
                unsigned int node=voronoi(off1D,v,c,t);
                if(!dat.count(node)) dat[node]=NNData();
                dat[node].vol+=pixelvol;
                Neighbours neighbours = voronoiPt.getNeighbours(VoxelIndex(off1D,v));
                for (unsigned int n=0; n<neighbours.size(); n++)
                    if(voronoiPt(neighbours[n],c,t)!=indexPt)
                    {
                        NeighbourOffset of = voronoiPt.getDirection( off1D,neighbours[n].index1d ) ;
                        if(of.connectionType()==NeighbourOffset::FACE)
                        {
                            if(of[0]) dat[node].surf+=pixelsurf[0];
                            else if(of[1]) dat[node].surf+=pixelsurf[1];
                            else if(of[2]) dat[node].surf+=pixelsurf[2];
                        }
                    }
                if(distPt(off1D,v,c,t)+dist(off1D,v,c,t)<dat[node].dist) dat[node].dist=distPt(off1D,v,c,t)+dist(off1D,v,c,t);
            }

            if (This->method.getValue().getSelectedId() == LAPLACE)   // replace vol (SIBSON) by surf/dist coordinates (LAPLACE)
            {
                for ( typename NNMap::iterator it=dat.begin() ; it != dat.end(); it++ )
                    if((*it).second.dist==0) (*it).second.vol=std::numeric_limits<Real>::max();
                    else (*it).second.vol=(*it).second.surf/(*it).second.dist;
            }

            // prune to nbref if necessary (nb of natural neighbors >nbref)
            while(dat.size()>nbref)
            {
                Real vmin=std::numeric_limits<Real>::max(); unsigned int key=0;
                for ( typename NNMap::iterator it=dat.begin() ; it != dat.end(); it++ ) if((*it).second.vol<vmin) key=(*it).first;
                dat.erase(key);
            }

            // compute weights based on Natural Neighbor coordinates
            Real total=0;
            for ( typename NNMap::iterator it=dat.begin() ; it != dat.end(); it++ ) total+=(*it).second.vol;
            if(total)
            {
                int count=0;
                for ( typename NNMap::iterator it=dat.begin() ; it != dat.end(); it++ )
                {
                    weights(off1Di,vi,count,ti)=(*it).second.vol/total;
                    indices(off1Di,vi,count,ti)=(*it).first;
                    count++;
                }
            }
        }
    }

};









template <class ShapeFunctionTypes_,class ImageTypes_>
class VoronoiShapeFunction : public BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_>
{
    friend struct VoronoiShapeFunctionSpecialization<defaulttype::IMAGELABEL_IMAGE>;
    friend struct VoronoiShapeFunctionSpecialization<defaulttype::IMAGELABEL_BRANCHINGIMAGE>;

public:
    SOFA_CLASS(SOFA_TEMPLATE2(VoronoiShapeFunction, ShapeFunctionTypes_,ImageTypes_) , SOFA_TEMPLATE2(BaseImageShapeFunction, ShapeFunctionTypes_,ImageTypes_));
    typedef BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_> Inherit;

    /** @name  Shape function types */
    //@{
    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    //@}

    /** @name  Image data */
    //@{
    typedef ImageTypes_ ImageTypes;
    typedef typename Inherit::T T;
    typedef typename Inherit::imCoord imCoord;
    typedef typename Inherit::raImage raImage;

    typedef typename Inherit::raTransform raTransform;

    typedef typename Inherit::DistT DistT;
    typedef typename Inherit::DistTypes DistTypes;
    typedef typename Inherit::waDist waDist;
    Data< DistTypes > f_distances;

    typedef typename Inherit::IndT IndT;
    typedef typename Inherit::IndTypes IndTypes;
    typedef typename Inherit::waInd waInd;
    Data< IndTypes > f_voronoi;

    typedef Vec<3,int> iCoord;
    typedef std::pair<Real,iCoord > DistanceToPoint;

    //@}

    /** @name  Options */
    //@{
    Data<bool> f_clearData;
    Data<helper::OptionsGroup> method;
    Data<bool> biasDistances;
    Data<bool> useDijkstra;
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const VoronoiShapeFunction<ShapeFunctionTypes_,ImageTypes_>* = NULL) { return ShapeFunctionTypes_::Name()+std::string(",")+ImageTypes_::Name(); }


    virtual void init()
    {
        Inherit::init();

        // init voronoi, distance, weight and indice image
        VoronoiShapeFunctionSpecialization<ImageTypes::label>::init( this );

        // compute voronoi based on node positions
        VoronoiShapeFunctionSpecialization<ImageTypes::label>::computeVoronoi( this );

        // compute weights from voronoi
        if(this->method.getValue().getSelectedId() == DISTANCE)  VoronoiShapeFunctionSpecialization<ImageTypes::label>::ComputeWeigths_DistanceRatio(this);
        else VoronoiShapeFunctionSpecialization<ImageTypes::label>::ComputeWeigths_NaturalNeighbors(this);

        // clear voronoi and distance image ?
        if(this->f_clearData.getValue())
        {
            waDist dist(this->f_distances); dist->clear();
            waInd vor(this->f_voronoi); vor->clear();
        }
    }


protected:
    VoronoiShapeFunction()
        :Inherit()
        , f_distances(initData(&f_distances,DistTypes(),"distances",""))
        , f_voronoi(initData(&f_voronoi,IndTypes(),"voronoi",""))
        , f_clearData(initData(&f_clearData,true,"clearData","clear voronoi and distance images after computation"))
        , method ( initData ( &method,"method","method (param)" ) )
        , biasDistances(initData(&biasDistances,false,"bias","Bias distances using inverse pixel values"))
        , useDijkstra(initData(&useDijkstra,true,"useDijkstra","Use Dijkstra for geodesic distance computation (use fastmarching otherwise)"))
    {
        helper::OptionsGroup methodOptions(3,"0 - Distance ratio"
                                           ,"1 - Laplace interpolant"
                                           ,"2 - Sibson interpolant"
                                           );
        methodOptions.setSelectedItem(DISTANCE);
        method.setValue(methodOptions);

        f_distances.setGroup("output");
        f_voronoi.setGroup("output");
        method.setGroup("parameters");
        biasDistances.setGroup("parameters");
        useDijkstra.setGroup("parameters");
    }

    virtual ~VoronoiShapeFunction()
    {

    }

};


}
}
}


#endif
