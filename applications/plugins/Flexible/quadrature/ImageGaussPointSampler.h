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
#ifndef SOFA_ImageGaussPointSAMPLER_H
#define SOFA_ImageGaussPointSAMPLER_H

#include "../initFlexible.h"
#include "../quadrature/BaseGaussPointSampler.h"

#include "../types/PolynomialBasis.h"

#include "ImageTypes.h"
#include "ImageAlgorithms.h"

#include <set>
#include <map>

namespace sofa
{
namespace component
{
namespace engine
{

using helper::vector;

/**
 * This class samples an object represented by an image
 */


template <class ImageTypes_>
class ImageGaussPointSampler : public BaseGaussPointSampler
{
public:
    typedef BaseGaussPointSampler Inherit;
    SOFA_CLASS(SOFA_TEMPLATE(ImageGaussPointSampler,ImageTypes_),Inherit);

    /** @name  GaussPointSampler types */
    //@{
    typedef Inherit::Real Real;
    typedef Inherit::Coord Coord;
    typedef Inherit::SeqPositions SeqPositions;
    typedef Inherit::raPositions raPositions;
    typedef Inherit::waPositions waPositions;
    //@}

    /** @name  Image data */
    //@{
    typedef unsigned int IndT;
    typedef defaulttype::Image<IndT> IndTypes;
    typedef helper::ReadAccessor<Data< IndTypes > > raInd;
    typedef helper::WriteAccessor<Data< IndTypes > > waInd;
    Data< IndTypes > f_index;

    typedef ImageTypes_ DistTypes;
    typedef typename DistTypes::T DistT;
    typedef typename DistTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< DistTypes > > raDist;
    typedef helper::WriteAccessor<Data< DistTypes > > waDist;
    Data< DistTypes > f_w;

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > f_transform;
    //@}

    /** @name  region data */
    //@{
    Data< IndTypes > f_region;
    Data< DistTypes > f_error;
    //@}

    /** @name  Options */
    //@{
    Data<unsigned int> targetNumber;
    Data<bool> useDijkstra;
    Data<unsigned int> iterations;
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const ImageGaussPointSampler<ImageTypes_>* = NULL) { return ImageTypes_::Name(); }

    virtual const unsigned int* getRegion()
    {
        raInd rreg(this->f_region);
        const CImg<IndT>& regimg = rreg->getCImg(0);
        return &regimg(0);
    }

    virtual void init()
    {
        Inherit::init();

        addInput(&f_index);
        addInput(&f_w);
        addInput(&f_transform);
        addOutput(&f_region);
        addOutput(&f_error);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:
    ImageGaussPointSampler()    :   Inherit()
        , f_index(initData(&f_index,IndTypes(),"indices",""))
        , f_w(initData(&f_w,DistTypes(),"weights",""))
        , f_transform(initData(&f_transform,TransformType(),"transform",""))
        , f_region(initData(&f_region,IndTypes(),"region","sample region : labeled image with sample indices"))
        , f_error(initData(&f_error,DistTypes(),"error","weigth fitting error"))
        , targetNumber(initData(&targetNumber,(unsigned int)0,"targetNumber","target number of samples"))
        , useDijkstra(initData(&useDijkstra,true,"useDijkstra","Use Dijkstra for geodesic distance computation (use fastmarching otherwise)"))
        , iterations(initData(&iterations,(unsigned int)100,"iterations","maximum number of Lloyd iterations"))
    {
    }

    virtual ~ImageGaussPointSampler()
    {

    }

    typedef std::set<unsigned int> indList;  ///< list of parent indices

    struct regionData  ///< data related to each voronoi region
    {
        unsigned int nb;    // nb of voxels in the region
        unsigned int voronoiIndex;    // value corresponding to the value in the voronoi image
        Coord c;            // centroid
        indList indices;    // indices of parents of that region
        vector<vector<Real> > coeff;    // coeffs of weigh tfit
        Real err;           // errror of weight fit
        vector<Real> vol; // volume (and moments) of this region
        regionData(indList ind,unsigned int i):nb(1),voronoiIndex(i),c(Coord()),indices(ind),err(0) {}
    };

    virtual void update()
    {
        cleanDirty();

        // get tranform and images at time t
        raDist rweights(this->f_w);             if(!rweights->getCImgList().size())  { serr<<"Weights not found"<<sendl; return; }
        raInd rindices(this->f_index);          if(!rindices->getCImgList().size())  { serr<<"Indices not found"<<sendl; return; }
        raTransform transform(this->f_transform);
        const CImg<IndT>& indices = rindices->getCImg(0);  // suppose time=0

        imCoord dim = rweights->getDimensions();
        dim[3]=dim[4]=1; // remove nbchannels from dimensions (to allocate single channel images later)

        // get output data
        waPositions pos(this->f_position);
        waVolume vol(this->f_volume);
        pos.clear();

        // init voronoi (=region data) and distances (=error image)
        waDist werr(this->f_error);
        werr->setDimensions(dim);
        CImg<DistT>& dist = werr->getCImg(0);
        dist.fill((DistT)(-1));

        waInd wreg(this->f_region);
        wreg->setDimensions(dim);
        CImg<IndT>& regimg = wreg->getCImg(0);
        regimg.fill(0);

        // init regions
        vector<regionData> regions;

        if(this->f_order.getValue()==1) // midpoint integration : put samples uniformly and weight them by their volume
        {
            // identify regions with similar repartitions
            Cluster_SimilarIndices(regions);

            // init soft regions (=more than one parent) where uniform sampling will be done
            for(unsigned int i=0; i<regions.size(); i++)
            {
                if(regions[i].indices.size()>1)
                {
                    cimg_forXYZ(regimg,x,y,z)
                    if(regimg(x,y,z)==regions[i].voronoiIndex)
                    {
                        dist(x,y,z)=cimg::type<DistT>::max();
                        regimg(x,y,z)=0;
                    }
                    regions.erase (regions.begin()+i); i--;  //  erase region (soft regions will be generated after uniform sampling)
                }
            }


            // fixed points = points set by the user in soft regions.
            // Disabled for now since pos is cleared
            SeqPositions fpos;
            vector<unsigned int> fpos_voronoiIndex;
            for(unsigned int i=0; i<pos.size(); i++)
            {
                Coord p = transform->toImage(pos[i]);
                for (unsigned int j=0; j<3; j++)  p[j]=round(p[j]);
                if(indices.containsXYZC(p[0],p[1],p[2]))
                {
                    indList l;
                    cimg_forC(indices,v) if(indices(p[0],p[1],p[2],v)) l.insert(indices(p[0],p[1],p[2],v));
                    if(l.size()>1) { fpos.push_back(pos[i]); fpos_voronoiIndex.push_back(i+1); }
                }
            }

            // target nb of points
            unsigned int nbrigid = regions.size();
            unsigned int nb = (fpos.size()+nbrigid>targetNumber.getValue())?fpos.size()+nbrigid:targetNumber.getValue();
            unsigned int nbsoft = nb-nbrigid;
            if(this->f_printLog.getValue()) std::cout<<"GaussPointSampler: Number of rigid/soft regions : "<<nbrigid<<"/"<<nbsoft<< std::endl;

            // init seeds for uniform sampling
            std::set<std::pair<DistT,sofa::defaulttype::Vec<3,int> > > trial;

            // farthest point sampling using geodesic distances
            SeqPositions newpos;
            vector<unsigned int> newpos_voronoiIndex;

            for(unsigned int i=0; i<fpos.size(); i++) AddSeedPoint<DistT>(trial,dist,regimg, transform.ref(), fpos[i],fpos_voronoiIndex[i]);
            while(newpos.size()+fpos.size()<nbsoft)
            {
                DistT dmax=0;  Coord pmax;
                cimg_forXYZ(dist,x,y,z) if(dist(x,y,z)>dmax) { dmax=dist(x,y,z); pmax =Coord(x,y,z); }
                if(dmax)
                {
                    newpos.push_back(transform->fromImage(pmax));
                    newpos_voronoiIndex.push_back(fpos.size()+nbrigid+newpos.size());
                    AddSeedPoint<DistT>(trial,dist,regimg, transform.ref(), newpos.back(),newpos_voronoiIndex.back());
                    if(useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
                    else fastMarching<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
                }
                else break;
            }

            // Loyd
            unsigned int it=0;
            bool converged =(it>=iterations.getValue())?true:false;
            while(!converged)
            {
                converged=!(Lloyd<DistT,DistT>(newpos,newpos_voronoiIndex,dist,regimg,this->f_transform.getValue(),NULL));
                // recompute voronoi
                cimg_foroff(dist,off) if(dist[off]!=-1) dist[off]=cimg::type<DistT>::max();
                for(unsigned int i=0; i<fpos.size(); i++) AddSeedPoint<DistT>(trial,dist,regimg, this->f_transform.getValue(), fpos[i],fpos_voronoiIndex[i]);
                for(unsigned int i=0; i<newpos.size(); i++) AddSeedPoint<DistT>(trial,dist,regimg, this->f_transform.getValue(), newpos[i],newpos_voronoiIndex[i]);
                if(useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
                else fastMarching<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
                it++; if(it>=iterations.getValue()) converged=true;
            }

            if(this->f_printLog.getValue()) std::cout<<"GaussPointSampler: Completed in "<< it <<" Lloyd iterations"<<std::endl;

            // create soft regions and update teir data
            for(unsigned int i=0; i<fpos.size(); i++)           // Disabled for now since fpos is empty
            {
                Coord p = transform->toImage(fpos[i]);
                for (unsigned int j=0; j<3; j++)  p[j]=round(p[j]);
                if(indices.containsXYZC(p[0],p[1],p[2]))
                {
                    indList l; cimg_forC(indices,v) if(indices(p[0],p[1],p[2],v)) l.insert(indices(p[0],p[1],p[2],v));
                    regionData reg(l,fpos_voronoiIndex[i]);
                    reg.c=fpos[i];
                    regions.push_back(reg);
                }
            }
            for(unsigned int i=0; i<newpos.size(); i++)
            {
                Coord p = transform->toImage(newpos[i]);
                for (unsigned int j=0; j<3; j++)  p[j]=round(p[j]);
                if(indices.containsXYZC(p[0],p[1],p[2]))
                {
                    indList l; cimg_forC(indices,v) if(indices(p[0],p[1],p[2],v)) l.insert(indices(p[0],p[1],p[2],v));
                    regionData reg(l,newpos_voronoiIndex[i]);
                    reg.c=newpos[i];
                    regions.push_back(reg);
                }
            }

            // fit weights
            Real err=0;
            for(unsigned int i=0; i<regions.size(); i++)
            {
                regions[i].nb=0; cimg_foroff(regimg,off)  if(regimg(off) == regions[i].voronoiIndex) regions[i].nb++;

                computeVolumes(regions[i],0);
                fitWeights(regions[i],1,true);
                err+=regions[i].err;
                //if(this->f_printLog.getValue()) std::cout<<"GaussPointSampler: weight fitting error on sample "<<i<<" = "<<regions[i].err<< std::endl;
            }
            if(this->f_printLog.getValue()) std::cout<<"GaussPointSampler: total error = "<<err<<std::endl;
        }
        else
        {


            if(this->f_method.getValue().getSelectedId() == GAUSSLEGENDRE)
            {
                serr<<"GAUSSLEGENDRE quadrature not yet implemented"<<sendl;
            }
            else if(this->f_method.getValue().getSelectedId() == NEWTONCOTES)
            {
                serr<<"NEWTONCOTES quadrature not yet implemented"<<sendl;
            }
            else if(this->f_method.getValue().getSelectedId() == ELASTON)
            {
                // identify regions with similar repartitions
                Cluster_SimilarIndices(regions);

                // fit weights
                for(unsigned int i=pos.size(); i<regions.size(); i++) fitWeights(regions[i],1);

                // subdivide region with largest error until target number is reached
                while(regions.size()<targetNumber.getValue())
                {
                    Real maxerr=-1;
                    unsigned int maxindex=0;
                    for(unsigned int i=0; i<regions.size(); i++) if(maxerr<regions[i].err) {maxerr=regions[i].err; maxindex=i;}
                    if(maxerr==0) break;
                    subdivideRegion(regions,maxindex);
                    fitWeights(regions[maxindex],1);
                    fitWeights(regions.back(),1);
                }

                // fit weights
                Real err=0;
                for(unsigned int i=0; i<regions.size(); i++)
                {
                    computeVolumes(regions[i],4);       // compute volumes and moments up to order 4
                    fitWeights(regions[i],1,true);
                    err+=regions[i].err;
                    //if(this->f_printLog.getValue()) std::cout<<"GaussPointSampler: weight fitting error on sample "<<regions[i].voronoiIndex<<" = "<<regions[i].err<< std::endl;
                }
                if(this->f_printLog.getValue()) std::cout<<"GaussPointSampler: total error = "<<err<<std::endl;
            }

        }

        // create samples
        pos.resize ( regions.size() );
        vol.resize ( regions.size() );

        for(unsigned int i=0; i<regions.size(); i++)
        {
            pos[i]=regions[i].c;
            vol[i].assign(regions[i].vol.begin(),regions[i].vol.end());
        }

        cimg_forXYZ(dist,x,y,z) if(dist(x,y,z)==-1) dist(x,y,z)=0; // clean error output image (used as a container for distances)

        if(this->f_printLog.getValue()) if(pos.size())    std::cout<<"GaussPointSampler: "<< pos.size() <<" generated samples"<<std::endl;
    }


    virtual void draw(const core::visual::VisualParams* vparams)
    {
        Inherit::draw(vparams);
    }


protected:


    /// Identify regions sharing similar parents
    /// returns a list of region containing the parents, the number of voxels and center; and fill the voronoi image
    void Cluster_SimilarIndices(vector<regionData>& regions)
    {
        // get tranform and images at time t
        raInd rindices(this->f_index);          if(!rindices->getCImgList().size())  { serr<<"Indices not found"<<sendl; return; }
        raTransform transform(this->f_transform);
        const CImg<IndT>& indices = rindices->getCImg(0);  // suppose time=0

        // get regimg
        waInd wreg(this->f_region);
        CImg<IndT>& regimg = wreg->getCImg(0);

        // map to find repartitions-> region index
        typedef std::map<indList, unsigned int> indMap;
        indMap List;

        // allows user to fix points. Currently disabled since pos is cleared
        raPositions pos(this->f_position);
        const unsigned int initialPosSize=pos.size();
        for(unsigned int i=0; i<initialPosSize; i++)
        {
            Coord p = transform->toImage(pos[i]);
            for (unsigned int j=0; j<3; j++)  p[j]=round(p[j]);
            if(indices.containsXYZC(p[0],p[1],p[2]))
            {
                indList l;
                cimg_forC(indices,v) if(indices(p[0],p[1],p[2],v)) l.insert(indices(p[0],p[1],p[2],v));
                List[l]=i;
                regions.push_back(regionData(l,i+1));
                regimg(p[0],p[1],p[2])=regions.back().voronoiIndex;
            }
        }

        // traverse index image to identify regions with unique indices
        cimg_forXYZ(indices,x,y,z)
        if(indices(x,y,z))
        {
            indList l;
            cimg_forC(indices,v) if(indices(x,y,z,v)) l.insert(indices(x,y,z,v));
            indMap::iterator it=List.find(l);
            unsigned int index;
            if(it==List.end()) { index=List.size(); List[l]=index;  regions.push_back(regionData(l,index+1));}
            else { index=it->second; regions[index].nb++;}

            regions[index].c+=transform->fromImage(Coord(x,y,z));
            regimg(x,y,z)=regions[index].voronoiIndex;
        }

        // average to get centroid (may not be inside the region if not convex)
        for(unsigned int i=0; i<regions.size(); i++)
        {
            regions[i].c/=(Real)regions[i].nb;
        }
    }

    /// subdivide region[index] in two regions
    void subdivideRegion(vector<regionData>& regions, const unsigned int index)
    {
        raTransform transform(this->f_transform);
        raInd rindices(this->f_index);
        const CImg<IndT>& indices = rindices->getCImg(0);  // suppose time=0

        waInd wreg(this->f_region);
        CImg<IndT>& regimg = wreg->getCImg(0);

        waDist werr(this->f_error);
        CImg<DistT>& dist = werr->getCImg(0);

        vector<Coord> pos(2);
        vector<unsigned int> vorindex;
        vorindex.push_back(regions[index].voronoiIndex);
        vorindex.push_back(regions.size()+1);
        for(unsigned int i=0; i<regions.size(); i++) if(vorindex[1]==regions[i].voronoiIndex) vorindex[1]++; // check that the voronoi index is unique. not necessary in principle

        // get closest/farthest point from c and init distance image
        Real dmin=cimg::type<Real>::max(),dmax=0;
        cimg_forXYZ(regimg,x,y,z)
        if(regimg(x,y,z)==vorindex[0])
        {
            dist(x,y,z)=cimg::type<DistT>::max();
            Coord p = transform->fromImage(Coord(x,y,z));
            Real d = (p-regions[index].c).norm2();
            if(dmin>d) {dmin=d; pos[0]=p;}
            if(dmax<d) {dmax=d; pos[1]=p;}
        }
        else dist(x,y,z)=(DistT)(-1);

        // Loyd relaxation
        std::set<std::pair<DistT,sofa::defaulttype::Vec<3,int> > > trial;
        unsigned int it=0;
        bool converged =(it>=iterations.getValue())?true:false;

        for(unsigned int i=0; i<2; i++) AddSeedPoint<DistT>(trial,dist,regimg, this->f_transform.getValue(), pos[i],vorindex[i]);
        if(useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
        else fastMarching<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
        //dist.display();
        //regimg.display();
        while(!converged)
        {
            converged=!(Lloyd<DistT,DistT>(pos,vorindex,dist,regimg,this->f_transform.getValue(),NULL));
            // recompute voronoi
            cimg_foroff(dist,off) if(dist[off]!=-1) dist[off]=cimg::type<DistT>::max();
            for(unsigned int i=0; i<2; i++) AddSeedPoint<DistT>(trial,dist,regimg, this->f_transform.getValue(), pos[i],vorindex[i]);
            if(useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
            else fastMarching<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
            it++; if(it>=iterations.getValue()) converged=true;
        }

        // add region
        Coord p = transform->toImage(pos[1]); for (unsigned int j=0; j<3; j++)  p[j]=round(p[j]);
        indList l; cimg_forC(indices,v) if(indices(p[0],p[1],p[2],v)) l.insert(indices(p[0],p[1],p[2],v));
        regionData reg(l,vorindex[1]);
        reg.c=pos[1];
        reg.nb=0; cimg_foroff(regimg,off)  if(regimg(off) == reg.voronoiIndex) reg.nb++;
        regions.push_back(reg);

        // update old region data
        p = transform->toImage(pos[0]); for (unsigned int j=0; j<3; j++)  p[j]=round(p[j]);
        regions[index].indices.clear(); cimg_forC(indices,v) if(indices(p[0],p[1],p[2],v)) regions[index].indices.insert(indices(p[0],p[1],p[2],v));
        regions[index].c=pos[0];
        regions[index].nb=0; cimg_foroff(regimg,off)  if(regimg(off) == regions[index].voronoiIndex) regions[index].nb++;
    }

    /// compute volumes in the region based on its center
    void computeVolumes(regionData& region, const unsigned int order)
    {
        raInd rreg(this->f_region);
        const CImg<IndT>& regimg = rreg->getCImg(0);

        raTransform transform(this->f_transform);

        int dimBasis=(order+1)*(order+2)*(order+3)/6; // dimension of polynomial basis in 3d
        const Real dv =  this->f_transform.getValue().getScale()[0] * this->f_transform.getValue().getScale()[1] * this->f_transform.getValue().getScale()[2];

        region.vol.resize(dimBasis);
        region.vol.fill((Real)0);

        vector<Real> basis;
        basis.resize(dimBasis);

        cimg_forXYZ(regimg,x,y,z)
        if(regimg(x,y,z)==region.voronoiIndex)
        {
            Coord prel = region.c - transform->fromImage(Coord(x,y,z));
            defaulttype::getCompleteBasis(basis,prel,order);
            for(int k=0; k<dimBasis; k++) region.vol[k]+=basis[k]*dv;
        }
    }



    /// compute polynomial approximation of the weights in the region
    void fitWeights(regionData& region, const unsigned int order, const bool writeOutput=false)
    {
        // get tranform and images at time t
        raDist rweights(this->f_w);             if(!rweights->getCImgList().size())  { serr<<"Weights not found"<<sendl; return; }
        raInd rindices(this->f_index);          if(!rindices->getCImgList().size())  { serr<<"Indices not found"<<sendl; return; }
        raTransform transform(this->f_transform);
        const CImg<DistT>& weights = rweights->getCImg(0);  // suppose time=0
        const CImg<IndT>& indices = rindices->getCImg(0);  // suppose time=0
        int nb=region.nb;

        // get region image
        raInd rreg(this->f_region);
        const CImg<IndT>& regimg = rreg->getCImg(0);

        // list of relative coords
        vector<Coord> pi(nb);

        // list of weights (one for each parent)
        typedef std::map<unsigned int, vector<Real> > wiMap;
        wiMap wi;
        for(indList::iterator it=region.indices.begin(); it!=region.indices.end(); it++) wi[*it]=vector<Real>(nb,(Real)0);

        // get them from images
        unsigned int count=0;
        cimg_forXYZ(regimg,x,y,z)
        if(regimg(x,y,z)==region.voronoiIndex)
        {
            cimg_forC(indices,v) if(indices(x,y,z,v))
            {
                typename wiMap::iterator wIt=wi.find(indices(x,y,z,v));
                if(wIt!=wi.end()) wIt->second[count]=(Real)weights(x,y,z,v);
            }
            pi[count]= transform->fromImage(Coord(x,y,z)) - region.c;
            count++;
        }

        // fit weights
        region.coeff.clear();
        region.err = 0;
        for(typename wiMap::iterator wIt=wi.begin(); wIt!=wi.end(); wIt++)
        {
            vector<Real> coeff;
            defaulttype::PolynomialFit(coeff,wIt->second,pi,order);
            Real err = defaulttype::getPolynomialFit_Error(coeff,wIt->second,pi);
            region.coeff.push_back(coeff);
            region.err+=err;
            //            if(this->f_printLog.getValue()) std::cout<<"GaussPointSampler: weight fitting error on sample "<<region.voronoiIndex<<" ("<<wIt->first<<") = "<<err<< std::endl;
        }

        // write error into output image
        if(writeOutput)
        {
            waDist werr(this->f_error);
            CImg<DistT>& outimg = werr->getCImg(0);
            count=0;
            cimg_forXYZ(regimg,x,y,z)
            if(regimg(x,y,z)==region.voronoiIndex)
            {
                outimg(x,y,z)=0;
                unsigned int j=0;
                for(typename wiMap::iterator wIt=wi.begin(); wIt!=wi.end(); wIt++)  outimg(x,y,z)+=defaulttype::getPolynomialFit_Error(region.coeff[j++],wIt->second[count],pi[count]);
                count++;
            }

        }

    }



};

}
}
}

#endif
