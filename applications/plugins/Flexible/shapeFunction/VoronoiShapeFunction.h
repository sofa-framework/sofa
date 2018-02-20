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
#ifndef FLEXIBLE_VoronoiShapeFunction_H
#define FLEXIBLE_VoronoiShapeFunction_H

#include <Flexible/config.h>
#include "../shapeFunction/BaseShapeFunction.h"
#include "../shapeFunction/BaseImageShapeFunction.h"
#include "../types/PolynomialBasis.h"

#include <image/ImageTypes.h>
#include <image/ImageAlgorithms.h>

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

/**
Voronoi shape functions are natural neighbor interpolants
there are computed from an image (typically a rasterized object)
  */



/// Default implementation does not compile
template <class ImageType>
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
template <class T>
struct VoronoiShapeFunctionSpecialization<defaulttype::Image<T>>
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
        cimg_forXYZC(inimg,x,y,z,c) if(inimg(x,y,z,c)) dist(x,y,z)=cimg_library::cimg::type<DistT>::max();

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
        typedef typename VoronoiShapeFunction::Coord Coord;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::raTransform raTransform;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        typedef defaulttype::Vec<3,int> iCoord;
        typedef std::pair<DistT,iCoord > DistanceToPoint;

        // retrieve data
        raImage in(This->image);
        raTransform inT(This->transform);
        if(in->isEmpty())  { This->serr<<"Image not found"<<This->sendl; return; }
        const typename ImageTypes::CImgT& inimg = in->getCImg(0);  // suppose time=0
        const typename ImageTypes::CImgT* biasFactor=This->biasDistances.getValue()?&inimg:NULL;

        waInd vorData(This->f_voronoi);         typename IndTypes::CImgT& voronoi = vorData->getCImg();
        waDist distData(This->f_distances);     typename DistTypes::CImgT& dist = distData->getCImg();

        helper::ReadAccessor<Data<helper::vector<Coord> > > parent(This->f_position);
        if(!parent.size()) { This->serr<<"Parent nodes not found"<<This->sendl; return; }
        helper::vector<iCoord> parentiCoord;        for(unsigned int i=0; i<parent.size(); i++) { Coord p = inT->toImageInt(parent[i]);  parentiCoord.push_back(iCoord(p[0],p[1],p[2])); }

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
        typedef typename VoronoiShapeFunction::Coord Coord;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::raTransform raTransform;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        typedef defaulttype::Vec<3,int> iCoord;
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

        helper::ReadAccessor<Data<helper::vector<Coord> > > parent(This->f_position);
        if(!parent.size()) { This->serr<<"Parent nodes not found"<<This->sendl; return; }
        helper::vector<iCoord> parentiCoord;        for(unsigned int i=0; i<parent.size(); i++) { Coord p = inT->toImageInt(parent[i]);  parentiCoord.push_back(iCoord(p[0],p[1],p[2])); }

        unsigned int nbref=This->f_nbRef.getValue();

        // compute weight of each parent
        for(unsigned int i=0; i<parentiCoord.size(); i++)
        {
            std::set<DistanceToPoint> trial;                // list of seed points

            // distance max to voronoi
            DistT dmax=0;
            cimg_forXYZ(voronoi,x,y,z) if(voronoi(x,y,z)==i+1)
            {
                if(dmax<dist(x,y,z)) dmax=dist(x,y,z);
                // check neighbors to retrieve upper distance bound
                if(x!=0                 && voronoi(x-1,y,z)!=i+1  && voronoi(x-1,y,z)!=0)   { if(dmax<dist(x-1,y,z)) dmax=dist(x-1,y,z); }
                if(x!=voronoi.width()-1 && voronoi(x+1,y,z)!=i+1  && voronoi(x+1,y,z)!=0)   { if(dmax<dist(x+1,y,z)) dmax=dist(x+1,y,z); }
                if(y!=0                 && voronoi(x,y-1,z)!=i+1  && voronoi(x,y-1,z)!=0)   { if(dmax<dist(x,y-1,z)) dmax=dist(x,y-1,z); }
                if(y!=voronoi.height()-1 && voronoi(x,y+1,z)!=i+1 && voronoi(x,y+1,z)!=0)   { if(dmax<dist(x,y+1,z)) dmax=dist(x,y+1,z); }
                if(z!=0                 && voronoi(x,y,z-1)!=i+1  && voronoi(x,y,z-1)!=0)   { if(dmax<dist(x,y,z-1)) dmax=dist(x,y,z-1); }
                if(z!=voronoi.depth()-1 && voronoi(x,y,z+1)!=i+1  && voronoi(x,y,z+1)!=0)   { if(dmax<dist(x,y,z+1)) dmax=dist(x,y,z+1); }
            }

            // extend voronoi to 2*dmax
            typename DistTypes::CImgT distP=dist;  cimg_foroff(distP,off) if(distP[off]!=-1) distP[off]=dmax*(DistT)2.;
            typename IndTypes::CImgT voronoiP=voronoi;
            AddSeedPoint<DistT>(trial,distP,voronoiP, parentiCoord[i],i+1);
            if(This->useDijkstra.getValue()) dijkstra<DistT,T>(trial,distP, voronoiP, inT->getScale() , biasFactor); else fastMarching<DistT,T>(trial,distP, voronoiP, inT->getScale() ,biasFactor );

            // distances from voronoi border
            typename DistTypes::CImgT distB=dist;  cimg_foroff(distB,off) if(distB[off]!=-1) distB[off]=dmax;
            typename IndTypes::CImgT voronoiB=voronoi;
            cimg_forXYZ(voronoi,x,y,z) if(voronoi(x,y,z)==i+1)
            {
                bool border=false;
                iCoord BP;
                // subpixel voronoi frontier localization
                BP.set(x-1,y,z); if(vorData->isInside((int)BP[0],(int)BP[1],(int)BP[2])) if(voronoi(BP[0],BP[1],BP[2])!=i+1  && voronoi(BP[0],BP[1],BP[2])!=0)                { border=true; distB(BP[0],BP[1],BP[2])= (DistT)0.5*( distP(BP[0],BP[1],BP[2]) - dist(BP[0],BP[1],BP[2]) );                    trial.insert( DistanceToPoint(distB(BP[0],BP[1],BP[2]),BP) ); DistT d = (DistT)0.5*(dist(BP[0],BP[1],BP[2]) + distP(BP[0],BP[1],BP[2])) - dist(x,y,z);  if(d<0) d=0;  if(d<distB(x,y,z)) distB(x,y,z) = d; }
                BP.set(x+1,y,z); if(vorData->isInside((int)BP[0],(int)BP[1],(int)BP[2])) if(voronoi(BP[0],BP[1],BP[2])!=i+1  && voronoi(BP[0],BP[1],BP[2])!=0)                { border=true; distB(BP[0],BP[1],BP[2])= (DistT)0.5*( distP(BP[0],BP[1],BP[2]) - dist(BP[0],BP[1],BP[2]) );                    trial.insert( DistanceToPoint(distB(BP[0],BP[1],BP[2]),BP) ); DistT d = (DistT)0.5*(dist(BP[0],BP[1],BP[2]) + distP(BP[0],BP[1],BP[2])) - dist(x,y,z);  if(d<0) d=0;  if(d<distB(x,y,z)) distB(x,y,z) = d; }
                BP.set(x,y-1,z); if(vorData->isInside((int)BP[0],(int)BP[1],(int)BP[2])) if(voronoi(BP[0],BP[1],BP[2])!=i+1  && voronoi(BP[0],BP[1],BP[2])!=0)                { border=true; distB(BP[0],BP[1],BP[2])= (DistT)0.5*( distP(BP[0],BP[1],BP[2]) - dist(BP[0],BP[1],BP[2]) );                    trial.insert( DistanceToPoint(distB(BP[0],BP[1],BP[2]),BP) ); DistT d = (DistT)0.5*(dist(BP[0],BP[1],BP[2]) + distP(BP[0],BP[1],BP[2])) - dist(x,y,z);  if(d<0) d=0;  if(d<distB(x,y,z)) distB(x,y,z) = d; }
                BP.set(x,y+1,z); if(vorData->isInside((int)BP[0],(int)BP[1],(int)BP[2])) if(voronoi(BP[0],BP[1],BP[2])!=i+1  && voronoi(BP[0],BP[1],BP[2])!=0)                { border=true; distB(BP[0],BP[1],BP[2])= (DistT)0.5*( distP(BP[0],BP[1],BP[2]) - dist(BP[0],BP[1],BP[2]) );                    trial.insert( DistanceToPoint(distB(BP[0],BP[1],BP[2]),BP) ); DistT d = (DistT)0.5*(dist(BP[0],BP[1],BP[2]) + distP(BP[0],BP[1],BP[2])) - dist(x,y,z);  if(d<0) d=0;  if(d<distB(x,y,z)) distB(x,y,z) = d; }
                BP.set(x,y,z-1); if(vorData->isInside((int)BP[0],(int)BP[1],(int)BP[2])) if(voronoi(BP[0],BP[1],BP[2])!=i+1  && voronoi(BP[0],BP[1],BP[2])!=0)                { border=true; distB(BP[0],BP[1],BP[2])= (DistT)0.5*( distP(BP[0],BP[1],BP[2]) - dist(BP[0],BP[1],BP[2]) );                    trial.insert( DistanceToPoint(distB(BP[0],BP[1],BP[2]),BP) ); DistT d = (DistT)0.5*(dist(BP[0],BP[1],BP[2]) + distP(BP[0],BP[1],BP[2])) - dist(x,y,z);  if(d<0) d=0;  if(d<distB(x,y,z)) distB(x,y,z) = d; }
                BP.set(x,y,z+1); if(vorData->isInside((int)BP[0],(int)BP[1],(int)BP[2])) if(voronoi(BP[0],BP[1],BP[2])!=i+1  && voronoi(BP[0],BP[1],BP[2])!=0)                { border=true; distB(BP[0],BP[1],BP[2])= (DistT)0.5*( distP(BP[0],BP[1],BP[2]) - dist(BP[0],BP[1],BP[2]) );                    trial.insert( DistanceToPoint(distB(BP[0],BP[1],BP[2]),BP) ); DistT d = (DistT)0.5*(dist(BP[0],BP[1],BP[2]) + distP(BP[0],BP[1],BP[2])) - dist(x,y,z);  if(d<0) d=0;  if(d<distB(x,y,z)) distB(x,y,z) = d; }
                if(border)  trial.insert( DistanceToPoint(distB(x,y,z),iCoord(x,y,z)) );
            }
            if(This->useDijkstra.getValue()) dijkstra<DistT,T>(trial,distB, voronoiB, inT->getScale() , biasFactor); else fastMarching<DistT,T>(trial,distB, voronoiB, inT->getScale() ,biasFactor );

            // compute weight as distance ratio
            DistT TOL = 1E-4; // warning: hard coded tolerance on the weights (to maximize sparsity)
            cimg_forXYZ(voronoiP,x,y,z) if(voronoiP(x,y,z)==i+1)
            {
                DistT w;
                DistT db=distB(x,y,z),dp=distP(x,y,z);

                if(dp==0) w=(DistT)1.;
                else if(voronoi(x,y,z)==i+1) w=(DistT)0.5*((DistT)1. + db/(dp+db)); // inside voronoi: dist(frame,closestVoronoiBorder)=d+disttovoronoi
                else if(dp==db) w=(DistT)0.;
                else w=(DistT)0.5*((DistT)1. - db/(dp-db)); // outside voronoi: dist(frame,closestVoronoiBorder)=d-disttovoronoi
                if(w<TOL) w=0; else if(w>(DistT)1.-TOL) w=(DistT)1.;
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
        typedef typename VoronoiShapeFunction::Real Real;
        typedef typename VoronoiShapeFunction::Coord Coord;
        typedef typename VoronoiShapeFunction::raImage raImage;
        typedef typename VoronoiShapeFunction::raTransform raTransform;
        typedef typename VoronoiShapeFunction::DistTypes DistTypes;
        typedef typename VoronoiShapeFunction::DistT DistT;
        typedef typename VoronoiShapeFunction::waDist waDist;
        typedef typename VoronoiShapeFunction::IndTypes IndTypes;
        typedef typename VoronoiShapeFunction::waInd waInd;

        typedef defaulttype::Vec<3,int> iCoord;
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
        defaulttype::Vec<3,Real> pixelsurf(voxelsize[1]*voxelsize[2],voxelsize[0]*voxelsize[2],voxelsize[0]*voxelsize[1]);
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






///Voronoi shape functions are natural neighbor interpolants there are computed from an image (typically a rasterized object)
template <class ShapeFunctionTypes_,class ImageTypes_>
class VoronoiShapeFunction : public BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_>
{
    friend struct VoronoiShapeFunctionSpecialization<ImageTypes_>;

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

    typedef defaulttype::Vec<3,int> iCoord;
    typedef std::pair<Real,iCoord > DistanceToPoint;

    //@}

    /** @name  Options */
    //@{
    Data<bool> f_clearData; ///< clear voronoi and distance images after computation
    Data<helper::OptionsGroup> method; ///< method (param)
    Data<bool> biasDistances; ///< Bias distances using inverse pixel values
    Data<bool> useDijkstra; ///< Use Dijkstra for geodesic distance computation (use fastmarching otherwise)
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const VoronoiShapeFunction<ShapeFunctionTypes_,ImageTypes_>* = NULL) { return ShapeFunctionTypes_::Name()+std::string(",")+ImageTypes_::Name(); }


    virtual void init()
    {
        Inherit::init();

        // init voronoi, distance, weight and indice image
        VoronoiShapeFunctionSpecialization<ImageTypes>::init( this );

        // compute voronoi based on node positions
        VoronoiShapeFunctionSpecialization<ImageTypes>::computeVoronoi( this );

        // compute weights from voronoi
        if(this->method.getValue().getSelectedId() == DISTANCE)  VoronoiShapeFunctionSpecialization<ImageTypes>::ComputeWeigths_DistanceRatio(this);
        else VoronoiShapeFunctionSpecialization<ImageTypes>::ComputeWeigths_NaturalNeighbors(this);

        // clear voronoi and distance image ?
        if(this->f_clearData.getValue())
        {
            waDist dist(this->f_distances); dist->clear();
            waInd vor(this->f_voronoi); vor->clear();
        }

        if(this->f_printLog.getValue())  std::cout<<this->getName()<<" shape function initialized"<<std::endl;
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
