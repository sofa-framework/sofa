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

using core::behavior::BaseShapeFunction;
using defaulttype::Mat;
using defaulttype::Vec;
/**
Voronoi shape functions are natural neighbor interpolants
there are computed from an image (typically a rasterized object)
  */

template <class ShapeFunctionTypes_,class ImageTypes_>
class VoronoiShapeFunction : public BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_>
{
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
    Data<helper::OptionsGroup> method;
    Data<bool> biasDistances;
    Data<bool> useDijkstra;
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const VoronoiShapeFunction<ShapeFunctionTypes_,ImageTypes_>* = NULL) { return ShapeFunctionTypes_::Name()+std::string(",")+ImageTypes_::Name(); }


    virtual void init()
    {
        Inherit::init();

        helper::ReadAccessor<Data<vector<Coord> > > parent(this->f_position);
        if(!parent.size()) { serr<<"Parent nodes not found"<<sendl; return; }

        // get tranform and image at time t
        raImage in(this->image);
        raTransform inT(this->transform);
        if(!in->getCImgList().size())  { serr<<"Image not found"<<sendl; return; }
        const CImg<T>& inimg = in->getCImg(0);  // suppose time=0
        const CImg<T>* biasFactor=biasDistances.getValue()?&inimg:NULL;
        const Vec<3,Real>& voxelsize=this->transform.getValue().getScale();

        // init voronoi and distances
        imCoord dim = in->getDimensions(); dim[3]=dim[4]=1;
        waInd vorData(this->f_voronoi); vorData->setDimensions(dim);
        CImg<IndT>& voronoi = vorData->getCImg(); voronoi.fill(0);

        waDist distData(this->f_distances);         distData->setDimensions(dim);
        CImg<DistT>& dist = distData->getCImg(); dist.fill(-1);
        cimg_forXYZC(inimg,x,y,z,c) if(inimg(x,y,z,c)) dist(x,y,z)=cimg::type<DistT>::max();

        // init indices and weights images
        unsigned int nbref=this->f_nbRef.getValue();
        dim[3]=nbref;
        waInd indData(this->f_index); indData->setDimensions(dim);
        CImg<IndT>& indices = indData->getCImg(); indices.fill(0);

        waDist weightData(this->f_w);         weightData->setDimensions(dim);
        CImg<DistT>& weights = weightData->getCImg(); weights.fill(0);

        vector<iCoord> parentiCoord;
        for(unsigned int i=0; i<parent.size(); i++)
        {
            Coord p = inT->toImage(parent[i]);
            parentiCoord.push_back(iCoord(round(p[0]),round(p[1]),round(p[2])));
        }

        // compute voronoi and distances based on nodes
        std::set<DistanceToPoint> trial;                // list of seed points
        for(unsigned int i=0; i<parent.size(); i++)
        {
            trial.insert( DistanceToPoint(0.,parentiCoord[i]) );
            voronoi(parentiCoord[i][0],parentiCoord[i][1],parentiCoord[i][2])=i+1;
            dist(parentiCoord[i][0],parentiCoord[i][1],parentiCoord[i][2])=0;
        }
        if(useDijkstra.getValue()) dijkstra<Real,T>(trial,dist, voronoi, voxelsize , biasFactor); else fastMarching<Real,T>(trial,dist, voronoi, voxelsize ,biasFactor );

        // compute weights from voronoi
        if(this->method.getValue().getSelectedId() == DISTANCE)  ComputeWeigths_DistanceRatio(indices,weights,voronoi,dist,biasFactor,parentiCoord);
        else ComputeWeigths_NaturalNeighbors(indices,weights,voronoi,dist,biasFactor);
    }


protected:
    VoronoiShapeFunction()
        :Inherit()
        , f_distances(initData(&f_distances,DistTypes(),"distances",""))
        , f_voronoi(initData(&f_voronoi,IndTypes(),"voronoi",""))
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



    void ComputeWeigths_DistanceRatio(CImg<IndT>& indices, CImg<DistT>& weights, const CImg<IndT>& voronoi, const CImg<DistT>& dist, const CImg<T>* biasFactor, const vector<iCoord> &parentiCoord )
    {
        // get data
        const unsigned int nbref=this->f_nbRef.getValue();
        const Vec<3,Real>& voxelsize=this->transform.getValue().getScale();

        // compute weight of each parent
        for(unsigned int i=0; i<parentiCoord.size(); i++)
        {
            std::set<DistanceToPoint> trial;                // list of seed points
            DistT dmax=0;
            cimg_forXYZ(voronoi,x,y,z) if(voronoi(x,y,z)==i+1) if(dmax<dist(x,y,z)) dmax=dist(x,y,z);

            // distances from voronoi border
            CImg<DistT> distB=dist;  cimg_foroff(distB,off) if(distB[off]!=-1) distB[off]=dmax;
            CImg<IndT> voronoiP=voronoi;
            cimg_forXYZ(voronoi,x,y,z) if(voronoi(x,y,z)==i+1)
            {
                bool border=false;  if(x!=0 && voronoi(x-1,y,z)!=i+1  && voronoi(x-1,y,z)!=0) border=true; else if(x!=voronoi.width()-1 && voronoi(x+1,y,z)!=i+1 && voronoi(x+1,y,z)!=0) border=true; else if(y!=0 && voronoi(x,y-1,z)!=i+1 && voronoi(x,y-1,z)!=0) border=true; else if(y!=voronoi.height()-1 && voronoi(x,y+1,z)!=i+1 && voronoi(x,y+1,z)!=0) border=true; else if(z!=0 && voronoi(x,y,z-1)!=i+1 && voronoi(x,y,z-1)!=0) border=true; else if(z!=voronoi.depth()-1 && voronoi(x,y,z+1)!=i+1 && voronoi(x,y,z+1)!=0) border=true;
                if(border)
                {
                    distB(x,y,z)=0;
                    trial.insert( DistanceToPoint(0.,iCoord(x,y,z)) );
                }
            }
            if(this->useDijkstra.getValue()) dijkstra<Real,T>(trial,distB, voronoiP, voxelsize , biasFactor); else fastMarching<Real,T>(trial,distB, voronoiP, voxelsize ,biasFactor );

            // extend voronoi to 2*dmax
            dmax*=(DistT)2.;
            CImg<DistT> distP=dist;  cimg_foroff(distP,off) if(distP[off]!=-1) distP[off]=dmax;

            voronoiP(parentiCoord[i][0],parentiCoord[i][1],parentiCoord[i][2])=i+1;
            distP(parentiCoord[i][0],parentiCoord[i][1],parentiCoord[i][2])=0;
            trial.insert( DistanceToPoint(0.,parentiCoord[i]) );

            if(this->useDijkstra.getValue()) dijkstra<Real,T>(trial,distP, voronoiP, voxelsize , biasFactor); else fastMarching<Real,T>(trial,distP, voronoiP, voxelsize ,biasFactor );

            // compute weight as distance ratio
            cimg_forXYZ(voronoiP,x,y,z) if(voronoiP(x,y,z)==i+1)
            {
                DistT w;
                if(voronoi(x,y,z)==i+1) w=(DistT)0.5*((DistT)1. + distB(x,y,z)/(distP(x,y,z)+distB(x,y,z))); // inside voronoi: dist(frame,closestVoronoiBorder)=d+disttovoronoi
                else w=(DistT)0.5*((DistT)1. - distB(x,y,z)/(distP(x,y,z)-distB(x,y,z))); // outside voronoi: dist(frame,closestVoronoiBorder)=d-disttovoronoi
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
        cimg_forXYZ(voronoi,x,y,z)
                if(voronoi(x,y,z))
        {
            DistT totW=0;
            cimg_forC(weights,c) totW+=weights(x,y,z,c);
            if(totW) cimg_forC(weights,c) weights(x,y,z,c)/=totW;
        }
    }


    void ComputeWeigths_NaturalNeighbors(CImg<IndT>& indices, CImg<DistT>& weights, const CImg<IndT>& voronoi, const CImg<DistT>& dist, const CImg<T>* biasFactor)
    {
        // get data
        const unsigned int nbref=this->f_nbRef.getValue();
        const Vec<3,Real>& voxelsize=this->transform.getValue().getScale();
        unsigned int indexPt=this->f_position.getValue().size()+1;

        cimg_forXYZ(voronoi,x,y,z)
                if(voronoi(x,y,z))
        {
            // compute updated voronoi including voxel (x,y,z)
            std::set<DistanceToPoint> trial;                // list of seed points
            trial.insert( DistanceToPoint(0.,iCoord(x,y,z)) );
            CImg<IndT> voronoiPt=voronoi;   voronoiPt(x,y,z)=indexPt;
            CImg<DistT> distPt=dist;        distPt(x,y,z)=0;

            if(this->useDijkstra.getValue()) dijkstra<Real,T>(trial,distPt, voronoiPt, voxelsize , biasFactor); else fastMarching<Real,T>(trial,distPt, voronoiPt, voxelsize,biasFactor );

            // compute Natural Neighbor Data based on neighboring voronoi cells
            NaturalNeighborDataMap dat=getNaturalNeighborData(indexPt,distPt,voronoiPt,dist,voronoi,voxelsize);

            if (this->method.getValue().getSelectedId() == LAPLACE)   // replace vol (SIBSON) by surf/dist coordinates (LAPLACE)
            {
                for ( typename NaturalNeighborDataMap::iterator it=dat.begin() ; it != dat.end(); it++ )
                    if((*it).second.dist==0) (*it).second.vol=std::numeric_limits<Real>::max();
                    else (*it).second.vol=(*it).second.surf/(*it).second.dist;
            }

            // prune to nbref if necessary (nb of natural neighbors >nbref)
            while(dat.size()>nbref)
            {
                Real vmin=std::numeric_limits<Real>::max(); unsigned int key=0;
                for ( typename NaturalNeighborDataMap::iterator it=dat.begin() ; it != dat.end(); it++ ) if((*it).second.vol<vmin) key=(*it).first;
                dat.erase(key);
            }

            // compute weights based on Natural Neighbor coordinates
            Real total=0;
            for ( typename NaturalNeighborDataMap::iterator it=dat.begin() ; it != dat.end(); it++ ) total+=(*it).second.vol;
            if(total)
            {
                int count=0;
                for ( typename NaturalNeighborDataMap::iterator it=dat.begin() ; it != dat.end(); it++ )
                {
                    weights(x,y,z,count)=(*it).second.vol/total;
                    indices(x,y,z,count)=(*it).first;
                    count++;
                }
            }
        }
    }

    /**
    * returns Natural Neighbor Interpolant coordinates of a point @param index: http://dilbert.engr.ucdavis.edu/~suku/nem/
    * from :
    *   - initial nodal voronoi regions (@param voronoi and @param distances)
    *   - the updated voronoi including the point (@param voronoiPt and @param distancesPt)
    * returns volume, area and distance associated to each natural neighbor (indexed in @param ref)
    */

    struct NaturalNeighborData
    {
        Real vol;
        Real surf;
        Real dist;
        NaturalNeighborData():vol(0),surf(0),dist(std::numeric_limits<Real>::max()) {}
    };
    typedef std::map<unsigned int, NaturalNeighborData> NaturalNeighborDataMap;

    NaturalNeighborDataMap getNaturalNeighborData(const unsigned int index, const CImg<DistT>& distancesPt,const CImg<IndT>& voronoiPt, const CImg<DistT>& distances,const CImg<IndT>& voronoi, const Vec<3,Real>& voxelsize)
    {
        NaturalNeighborDataMap data;
        Real pixelvol=voxelsize[0]*voxelsize[1]*voxelsize[2];
        Vec<3,Real> pixelsurf(voxelsize[1]*voxelsize[2],voxelsize[0]*voxelsize[2],voxelsize[0]*voxelsize[1]);
        //bool border;

        cimg_forXYZ(voronoiPt,x,y,z)
                if(voronoiPt(x,y,z)==index)
        {
            unsigned int node=voronoi(x,y,z);
            if(!data.count(node)) data[node]=NaturalNeighborData();
            data[node].vol+=pixelvol;
            //border=false;
            if(x!=0)                    if(voronoiPt(x-1,y,z)!=index) {data[node].surf+=pixelsurf[0];  /*border=true;*/}
            if(x!=voronoiPt.width()-1)  if(voronoiPt(x+1,y,z)!=index) {data[node].surf+=pixelsurf[0];  /*border=true;*/}
            if(y!=0)                    if(voronoiPt(x,y-1,z)!=index) {data[node].surf+=pixelsurf[1];  /*border=true;*/}
            if(y!=voronoiPt.height()-1) if(voronoiPt(x,y+1,z)!=index) {data[node].surf+=pixelsurf[1];  /*border=true;*/}
            if(z!=0)                    if(voronoiPt(x,y,z-1)!=index) {data[node].surf+=pixelsurf[2];  /*border=true;*/}
            if(z!=voronoiPt.depth()-1)  if(voronoiPt(x,y,z+1)!=index) {data[node].surf+=pixelsurf[2];  /*border=true;*/}
            if(distancesPt(x,y,z)+distances(x,y,z)<data[node].dist) data[node].dist=distancesPt(x,y,z)+distances(x,y,z);
        }
        return data;
    }


};


}
}
}


#endif
