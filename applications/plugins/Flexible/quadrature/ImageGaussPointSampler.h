/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_ImageGaussPointSAMPLER_H
#define SOFA_ImageGaussPointSAMPLER_H

#include <Flexible/config.h>
#include "../quadrature/BaseGaussPointSampler.h"
#include "../deformationMapping/BaseDeformationMapping.h"

#include "../types/PolynomialBasis.h"

#include <image/ImageTypes.h>
#include <image/ImageAlgorithms.h>

#include <sofa/helper/rmath.h>

#include <set>
#include <map>


namespace sofa
{
namespace component
{
namespace engine
{

/**
 * This class samples an object represented by an image with gauss points
 */

/// Default implementation does not compile
template <class ImageTypes, class MaskTypes>
struct ImageGaussPointSamplerSpecialization
{
};

/// forward declaration
template <class ImageTypes, class MaskTypes> class ImageGaussPointSampler;


/// Specialization for regular Image
template <class ImageT, class MaskT>
struct ImageGaussPointSamplerSpecialization<defaulttype::Image<ImageT>,defaulttype::Image<MaskT>>
{
    typedef ImageGaussPointSampler<defaulttype::Image<ImageT>,defaulttype::Image<MaskT>> ImageGaussPointSamplerT;

    typedef unsigned int IndT;
    typedef defaulttype::Image<IndT> IndTypes;

    static void init(ImageGaussPointSamplerT* This)
    {
        typedef typename ImageGaussPointSamplerT::IndTypes IndTypes;
        typedef typename ImageGaussPointSamplerT::waInd waInd;
        typedef typename ImageGaussPointSamplerT::DistTypes DistTypes;
        typedef typename ImageGaussPointSamplerT::raDist raDist;
        typedef typename ImageGaussPointSamplerT::waDist waDist;
        typedef typename ImageGaussPointSamplerT::waPositions waPositions;
        typedef typename ImageGaussPointSamplerT::waVolume waVolume;

        // retrieve data
        raDist rweights(This->f_w);             if(rweights->isEmpty())  { This->serr<<"Weights not found"<<This->sendl; return; }

        // init pos, vol, reg data; voronoi (=region data) and distances (=error image)
        typename ImageGaussPointSamplerT::imCoord dim = rweights->getDimensions();
        dim[DistTypes::DIMENSION_S]=dim[DistTypes::DIMENSION_T]=1;

        waPositions pos(This->f_position);          pos.clear();                // pos is cleared since it is always initialized with one point, so user placed points are not allowed for now..
        waVolume vol(This->f_volume);   vol.clear();

        waInd wreg(This->f_region);        wreg->setDimensions(dim);
        typename IndTypes::CImgT& regimg = wreg->getCImg();        regimg.fill(0);

        waDist werr(This->f_error);        werr->setDimensions(dim);
        typename DistTypes::CImgT& dist = werr->getCImg();        dist.fill(-1.0);

        This->Reg.clear();
    }


    /// midpoint integration : put samples uniformly and weight them by their volume
    static void midpoint(ImageGaussPointSamplerT* This)
    {
        typedef typename ImageGaussPointSamplerT::IndTypes IndTypes;
        typedef typename ImageGaussPointSamplerT::raInd raInd;
        typedef typename ImageGaussPointSamplerT::waInd waInd;
        typedef typename ImageGaussPointSamplerT::DistTypes DistTypes;
        typedef typename ImageGaussPointSamplerT::DistT DistT;
        typedef typename ImageGaussPointSamplerT::waDist waDist;
        typedef typename ImageGaussPointSamplerT::SeqPositions SeqPositions;
        typedef typename ImageGaussPointSamplerT::Coord Coord;
        typedef typename ImageGaussPointSamplerT::waPositions waPositions;
        typedef typename ImageGaussPointSamplerT::indList indList;
        typedef typename ImageGaussPointSamplerT::raTransform raTransform;
        typedef typename ImageGaussPointSamplerT::factType factType;

        typedef defaulttype::Vec<3,int> iCoord;
        typedef std::pair<DistT,iCoord > DistanceToPoint;

        // retrieve data
        raInd rindices(This->f_index);          if(rindices->isEmpty())  { This->serr<<"Indices not found"<<This->sendl; return; }        const typename IndTypes::CImgT& indices = rindices->getCImg();
        raTransform transform(This->f_transform);
        const Coord voxelsize(transform->getScale());

        waPositions pos(This->f_position);
        waInd wreg(This->f_region);        typename IndTypes::CImgT& regimg = wreg->getCImg();
        waDist werr(This->f_error);        typename DistTypes::CImgT& dist = werr->getCImg();   // use error image as a container for distances

        // init soft regions (=more than one parent) where uniform sampling will be done
        // rigid regions (one parent) are kept in the list of region (and dist remains=-1 so they will not be sampled)
        for(unsigned int i=0; i<This->Reg.size();i++)
        {
            if(This->Reg[i].parentsToNodeIndex.size()>1 || This->sampleRigidParts.getValue())
            {
                cimg_forXYZ(regimg,x,y,z) if(regimg(x,y,z)==*(This->Reg[i].voronoiIndices.begin()) )
                {
                    dist(x,y,z)=cimg_library::cimg::type<DistT>::max();
                    regimg(x,y,z)=0;
                }
                This->Reg.erase (This->Reg.begin()+i); i--;  //  erase region (soft regions will be generated after uniform sampling)
            }
        }
        unsigned int nbrigid = This->Reg.size();

        // fixed points = points set by the user in soft regions.
        // Disabled for now since pos is cleared
        SeqPositions fpos_voxelIndex;
        helper::vector<unsigned int> fpos_voronoiIndex;
        for(unsigned int i=0; i<pos.size(); i++)
        {
            Coord p = transform->toImageInt(pos[i]);
            if(indices.containsXYZC(p[0],p[1],p[2]))
            {
                indList l;
                cimg_forC(indices,v) if(indices(p[0],p[1],p[2],v)) l.insert(indices(p[0],p[1],p[2],v)-1);
                if(l.size()>1) { fpos_voxelIndex.push_back(p); fpos_voronoiIndex.push_back(i+1+nbrigid); }
            }
        }

        // target nb of points
        unsigned int nb = (fpos_voxelIndex.size()+nbrigid>This->targetNumber.getValue())?fpos_voxelIndex.size()+nbrigid:This->targetNumber.getValue();
        unsigned int nbsoft = nb-nbrigid;
        if(This->f_printLog.getValue()) std::cout<<This->getName()<<": Number of rigid/soft regions : "<<nbrigid<<"/"<<nbsoft<< std::endl;

        // init seeds for uniform sampling
        std::set<DistanceToPoint> trial;

        // farthest point sampling using geodesic distances
        SeqPositions newpos_voxelIndex;
        helper::vector<unsigned int> newpos_voronoiIndex;

        for(unsigned int i=0; i<fpos_voxelIndex.size(); i++) AddSeedPoint<DistT>(trial,dist,regimg, fpos_voxelIndex[i],fpos_voronoiIndex[i]);
        while(newpos_voxelIndex.size()+fpos_voxelIndex.size()<nbsoft)
        {
            DistT dmax=0;  Coord pmax;
            cimg_forXYZ(dist,x,y,z) if(dist(x,y,z)>dmax) { dmax=dist(x,y,z); pmax =Coord(x,y,z); }
            if(dmax)
            {
                newpos_voxelIndex.push_back(pmax);
                newpos_voronoiIndex.push_back(fpos_voxelIndex.size()+nbrigid+newpos_voxelIndex.size());
                AddSeedPoint<DistT>(trial,dist,regimg, newpos_voxelIndex.back(),newpos_voronoiIndex.back());
                if(This->useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg,voxelsize);
                else fastMarching<DistT,DistT>(trial,dist, regimg,voxelsize);
            }
            else break;
        }

        // Loyd
        unsigned int it=0;
        bool converged =(it>=This->iterations.getValue())?true:false;
        while(!converged)
        {
            converged=!(Lloyd<DistT>(newpos_voxelIndex,newpos_voronoiIndex,regimg));
            // recompute voronoi
            cimg_foroff(dist,off) if(dist[off]!=-1) dist[off]=cimg_library::cimg::type<DistT>::max();
            for(unsigned int i=0; i<fpos_voxelIndex.size(); i++) AddSeedPoint<DistT>(trial,dist,regimg, fpos_voxelIndex[i],fpos_voronoiIndex[i]);
            for(unsigned int i=0; i<newpos_voxelIndex.size(); i++) AddSeedPoint<DistT>(trial,dist,regimg, newpos_voxelIndex[i],newpos_voronoiIndex[i]);
            if(This->useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg,voxelsize); else fastMarching<DistT,DistT>(trial,dist, regimg,voxelsize);
            it++; if(it>=This->iterations.getValue()) converged=true;
        }
        if(This->f_printLog.getValue()) std::cout<<This->getName()<<": Completed in "<< it <<" Lloyd iterations"<<std::endl;

        // create soft regions and update teir data
        for(unsigned int i=0; i<fpos_voxelIndex.size(); i++)           // Disabled for now since fpos is empty
        {
            indList l; cimg_forXYZ(regimg,x,y,z) if(regimg(x,y,z)==fpos_voronoiIndex[i]) { cimg_forC(indices,v) if(indices(x,y,z,v)) l.insert(indices(x,y,z,v)-1); }   // collect indices over the region
            if(l.size())
            {
                factType reg(l,fpos_voronoiIndex[i]); reg.center=transform->fromImage(fpos_voxelIndex[i]);
                This->Reg.push_back(reg);
            }
        }
        for(unsigned int i=0; i<newpos_voxelIndex.size(); i++)
        {
            indList l; cimg_forXYZ(regimg,x,y,z) if(regimg(x,y,z)==newpos_voronoiIndex[i]) { cimg_forC(indices,v) if(indices(x,y,z,v)) l.insert(indices(x,y,z,v)-1); }   // collect indices over the region
            if(l.size())
            {
                factType reg(l,newpos_voronoiIndex[i]); reg.center=transform->fromImage(newpos_voxelIndex[i]);
                This->Reg.push_back(reg);
            }
        }
        // update rigid regions (might contain soft material due to voronoi proximity)
        for(unsigned int i=0; i<nbrigid; i++)
        {
            indList l; cimg_forXYZ(regimg,x,y,z) if(regimg(x,y,z)==*(This->Reg[i].voronoiIndices.begin()) ) { cimg_forC(indices,v) if(indices(x,y,z,v)) l.insert(indices(x,y,z,v)-1); }   // collect indices over the region
            This->Reg[i].setParents(l);
        }

        // update nb voxels in each region (used later in weight fitting)
        for(unsigned int i=0; i<This->Reg.size(); i++)
        {
            This->Reg[i].nb=0; cimg_foroff(regimg,off)  if(regimg(off) == *(This->Reg[i].voronoiIndices.begin())) This->Reg[i].nb++;
        }
    }

    /// returns true if (x,y,z) in the region of interest
    static bool isInMask(ImageGaussPointSamplerT* This,unsigned x,unsigned y, unsigned z)
    {
        typename ImageGaussPointSamplerT::raMask rmask(This->f_mask);
        if(rmask->isEmpty()) return true;
        typename ImageGaussPointSamplerT::raMaskLabels labels(This->f_maskLabels);
        typename ImageGaussPointSamplerT::MaskT val = rmask->getCImg()(x,y,z);
        for(unsigned int i=0;i<labels.size();i++) if(labels[i]==val) return true;
        return false;
    }

    /// Identify regions sharing similar parents
    /// returns a list of region containing the parents, the number of voxels and center; and fill the voronoi image
    static void Cluster_SimilarIndices(ImageGaussPointSamplerT* This)
    {
        typedef typename ImageGaussPointSamplerT::Real Real;
        typedef typename ImageGaussPointSamplerT::IndTypes IndTypes;
        typedef typename ImageGaussPointSamplerT::raInd raInd;
        typedef typename ImageGaussPointSamplerT::waInd waInd;
        typedef typename ImageGaussPointSamplerT::indList indList;
        typedef typename ImageGaussPointSamplerT::raTransform raTransform;
        typedef typename ImageGaussPointSamplerT::Coord Coord;
        typedef typename ImageGaussPointSamplerT::raPositions raPositions;
        typedef typename ImageGaussPointSamplerT::factType factType;

        // retrieve data
        raInd rindices(This->f_index);          if(rindices->isEmpty())  { This->serr<<"Indices not found"<<This->sendl; return; }        const typename IndTypes::CImgT& indices = rindices->getCImg();
        waInd wreg(This->f_region);        typename IndTypes::CImgT& regimg = wreg->getCImg();
        raTransform transform(This->f_transform);

        // map to find repartitions-> region index
        typedef std::map<indList, unsigned int> indMap;
        indMap List;

        // allows user to fix points. Currently disabled since pos is cleared
        raPositions pos(This->f_position);
        const unsigned int initialPosSize=pos.size();
        for(unsigned int i=0; i<initialPosSize; i++)
        {
            Coord p = transform->toImageInt(pos[i]);
            if(indices.containsXYZC(p[0],p[1],p[2]))
            {
                indList l;
                cimg_forC(indices,v) if(indices(p[0],p[1],p[2],v)) l.insert(indices(p[0],p[1],p[2],v)-1);
                List[l]=i;
                This->Reg.push_back(factType(l,i+1));
                regimg(p[0],p[1],p[2])=i+1;
            }
        }

        // traverse index image to identify regions with unique indices
        cimg_forXYZ(indices,x,y,z)
                if(indices(x,y,z))
                if(isInMask(This,x,y,z))
        {
            indList l;
            cimg_forC(indices,v) if(indices(x,y,z,v)) l.insert(indices(x,y,z,v)-1);
            typename indMap::iterator it=List.find(l);
            unsigned int index;
            if(it==List.end()) { index=List.size(); List[l]=index;  This->Reg.push_back(factType(l,index+1)); This->Reg.back().nb=1; }
            else { index=it->second; This->Reg[index].nb++;}

            This->Reg[index].center+=transform->fromImage(Coord(x,y,z));
            regimg(x,y,z)=*(This->Reg[index].voronoiIndices.begin());
        }

        // average to get centroid (may not be inside the region if not convex)
        for(unsigned int i=0; i<This->Reg.size(); i++) This->Reg[i].center/=(Real)This->Reg[i].nb;
    }

    /// subdivide region[index] in two regions
    static void subdivideRegion(ImageGaussPointSamplerT* This,const unsigned int index)
    {
        typedef typename ImageGaussPointSamplerT::Real Real;
        typedef typename ImageGaussPointSamplerT::IndTypes IndTypes;
        typedef typename ImageGaussPointSamplerT::waInd waInd;
        typedef typename ImageGaussPointSamplerT::DistTypes DistTypes;
        typedef typename ImageGaussPointSamplerT::DistT DistT;
        typedef typename ImageGaussPointSamplerT::waDist waDist;
        typedef typename ImageGaussPointSamplerT::raTransform raTransform;
        typedef typename ImageGaussPointSamplerT::Coord Coord;
        typedef typename ImageGaussPointSamplerT::factType factType;

        typedef defaulttype::Vec<3,int> iCoord;
        typedef std::pair<DistT,iCoord > DistanceToPoint;

        // retrieve data
        raTransform transform(This->f_transform);
        const Coord voxelsize(transform->getScale());

        waInd wreg(This->f_region);        typename IndTypes::CImgT& regimg = wreg->getCImg();
        waDist werr(This->f_error);        typename DistTypes::CImgT& dist = werr->getCImg();

        // compute
        helper::vector<Coord> pos(2);
        helper::vector<unsigned int> vorindex;
        vorindex.push_back(*(This->Reg[index].voronoiIndices.begin()));
        vorindex.push_back(This->Reg.size()+1);
        for(unsigned int i=0; i<This->Reg.size(); i++) if(vorindex[1]==*(This->Reg[i].voronoiIndices.begin())) vorindex[1]++; // check that the voronoi index is unique. not necessary in principle

        // get closest/farthest point from c and init distance image
        Real dmin=cimg_library::cimg::type<Real>::max(),dmax=0;
        cimg_forXYZ(regimg,x,y,z)
                if(regimg(x,y,z)==vorindex[0])
        {
            dist(x,y,z)=cimg_library::cimg::type<DistT>::max();
            Coord p = Coord(x,y,z);
            Real d = (transform->fromImage(p)-This->Reg[index].center).norm2();
            if(dmin>d) {dmin=d; pos[0]=p;}
            if(dmax<d) {dmax=d; pos[1]=p;}
        }
        else dist(x,y,z)=(DistT)(-1);

        // Loyd relaxation
        std::set<DistanceToPoint> trial;
        unsigned int it=0;
        bool converged =(it>=This->iterations.getValue())?true:false;

        for(unsigned int i=0; i<2; i++) AddSeedPoint<DistT>(trial,dist,regimg, pos[i],vorindex[i]);
        if(This->useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg,voxelsize); else fastMarching<DistT,DistT>(trial,dist, regimg,voxelsize);
        //dist.display();
        //regimg.display();
        while(!converged)
        {
            converged=!(Lloyd<DistT>(pos,vorindex,regimg));
            // recompute voronoi
            cimg_foroff(dist,off) if(dist[off]!=-1) dist[off]=cimg_library::cimg::type<DistT>::max();
            for(unsigned int i=0; i<2; i++) AddSeedPoint<DistT>(trial,dist,regimg, pos[i],vorindex[i]);
            if(This->useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg,voxelsize); else fastMarching<DistT,DistT>(trial,dist, regimg,voxelsize);
            it++; if(it>=This->iterations.getValue()) converged=true;
        }

        // add region
        factType reg;
        reg.parentsToNodeIndex=This->Reg[index].parentsToNodeIndex;
        reg.voronoiIndices.insert(vorindex[1]);
        reg.center=transform->fromImage(pos[1]);
        reg.nb=0; cimg_foroff(regimg,off)  if(regimg(off) == vorindex[1]) reg.nb++;
        This->Reg.push_back(reg);

        // update old region data
        This->Reg[index].center=transform->fromImage(pos[0]);
        This->Reg[index].nb=0; cimg_foroff(regimg,off)  if(regimg(off) == vorindex[0]) This->Reg[index].nb++;
    }



    /// update Polynomial Factors from the voxel map
    static void fillPolynomialFactors(ImageGaussPointSamplerT* This,const unsigned int factIndex, const bool writeErrorImg=false)
    {
        typedef typename ImageGaussPointSamplerT::Real Real;
        typedef typename ImageGaussPointSamplerT::IndTypes IndTypes;
        typedef typename ImageGaussPointSamplerT::raInd raInd;
        typedef typename ImageGaussPointSamplerT::DistTypes DistTypes;
        typedef typename ImageGaussPointSamplerT::raDist raDist;
        typedef typename ImageGaussPointSamplerT::waDist waDist;
        typedef typename ImageGaussPointSamplerT::Coord Coord;
        typedef typename ImageGaussPointSamplerT::indListIt indListIt;
        typedef typename ImageGaussPointSamplerT::raTransform raTransform;
        typedef typename ImageGaussPointSamplerT::factType factType;

        // retrieve data
        raDist rweights(This->f_w);             if(rweights->isEmpty())  { This->serr<<"Weights not found"<<This->sendl; return; }  const typename DistTypes::CImgT& weights = rweights->getCImg();
        raInd rindices(This->f_index);          if(rindices->isEmpty())  { This->serr<<"Indices not found"<<This->sendl; return; }  const typename IndTypes::CImgT& indices = rindices->getCImg();
        raInd rreg(This->f_region);        const typename IndTypes::CImgT& regimg = rreg->getCImg();
        raTransform transform(This->f_transform);
        const Coord voxelsize(transform->getScale());

        // list of absolute coords
        factType &fact = This->Reg[factIndex];
        helper::vector<Coord> pi(fact.nb);

        // weights (one line for each parent)
        typename ImageGaussPointSamplerT::Matrix wi(fact.parentsToNodeIndex.size(),fact.nb); wi.setZero();

        // get them from images
        unsigned int count=0;
        cimg_forXYZ(regimg,x,y,z)
        {
            indListIt it=fact.voronoiIndices.find(regimg(x,y,z));
            if(it!=fact.voronoiIndices.end())
            {
                cimg_forC(indices,v) if(indices(x,y,z,v))
                {
                    std::map<unsigned int,unsigned int>::iterator pit=fact.parentsToNodeIndex.find(indices(x,y,z,v)-1);
                    if(pit!=fact.parentsToNodeIndex.end())  wi(pit->second,count)= (Real)weights(x,y,z,v);
                }
                pi[count]= transform->fromImage(Coord(x,y,z));
                count++;
            }
        }

        fact.fill(wi,pi,This->fillOrder(),voxelsize,This->volOrder());

        //  std::cout<<"pt "<<*(fact.voronoiIndices.begin())-1<<" : "<<fact.center<<std::endl<<std::endl<<std::endl<<pi<<std::endl<<std::endl<<wi<<std::endl;
        //test: fact.directSolve(wi,pi); std::cout<<"Jacobi err="<<fact.getError()<<std::endl;

        // write error into output image
        if(writeErrorImg)
        {
            waDist werr(This->f_error); typename DistTypes::CImgT& outimg = werr->getCImg();
            count=0;
            cimg_forXYZ(regimg,x,y,z)
            {
                indListIt it=fact.voronoiIndices.find(regimg(x,y,z));
                if(it!=fact.voronoiIndices.end()) { outimg(x,y,z)=fact.getError(pi[count],wi.col(count)); count++; }
            }
        }
    }

};

///Samples an object represented by an image with gauss points
template <class ImageTypes_, class MaskTypes_>
class ImageGaussPointSampler : public BaseGaussPointSampler
{
    friend struct ImageGaussPointSamplerSpecialization<ImageTypes_,MaskTypes_>;
    typedef ImageGaussPointSamplerSpecialization<ImageTypes_,MaskTypes_> ImageGaussPointSamplerSpec;

public:
    typedef BaseGaussPointSampler Inherit;
    SOFA_CLASS(SOFA_TEMPLATE2(ImageGaussPointSampler,ImageTypes_, MaskTypes_),Inherit);

    /** @name  GaussPointSampler types */
    //@{
    typedef Inherit::Real Real;
    typedef Inherit::Coord Coord;
    typedef Inherit::SeqPositions SeqPositions;
    typedef Inherit::raPositions raPositions;
    typedef Inherit::waPositions waPositions;
    typedef Inherit::VTransform VTransform;
    //@}

    /** @name  Image data */
    //@{
    typedef typename ImageGaussPointSamplerSpec::IndT IndT;
    typedef typename ImageGaussPointSamplerSpec::IndTypes IndTypes;
    typedef helper::ReadAccessor<Data< IndTypes > > raInd;
    typedef helper::WriteOnlyAccessor<Data< IndTypes > > waInd;
    Data< IndTypes > f_index;

    typedef ImageTypes_ DistTypes;
    typedef typename DistTypes::T DistT;
    typedef typename DistTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< DistTypes > > raDist;
    typedef helper::WriteOnlyAccessor<Data< DistTypes > > waDist;
    Data< DistTypes > f_w;

    typedef MaskTypes_ MaskTypes;
    typedef typename MaskTypes::T  MaskT;
    typedef helper::ReadAccessor<Data< MaskTypes > > raMask;
    Data< MaskTypes > f_mask;
    typedef helper::vector<MaskT> MaskLabelsType;
    typedef helper::ReadAccessor<Data< MaskLabelsType > > raMaskLabels;
    Data< MaskLabelsType > f_maskLabels;

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
    Data<bool> f_clearData;
    Data<unsigned int> targetNumber;
    Data<bool> useDijkstra;
    Data<unsigned int> iterations;
    Data<bool> evaluateShapeFunction;   ///< If true, ImageGaussPointSampler::bwdInit() is called to evaluate shape functions over integration regions
                                        ///< and writes over values computed by sofa::component::mapping::LinearMapping.
                                        ///< Otherwise shape functions are interpolated only at sample locations using finite differencies in sofa::component::mapping::LinearMapping.
    Data<bool> sampleRigidParts;

    Data< unsigned int > f_fillOrder; ///< Fill Order  // For the mapping, we use second order fit (to have translation invariance of elastons, use first order)
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const ImageGaussPointSampler<ImageTypes_, MaskTypes_>* = NULL) { return ImageTypes_::Name()+std::string(",")+MaskTypes_::Name(); }

    virtual void init()
    {
        Inherit::init();

        addInput(&f_index);
        addInput(&f_w);
        addInput(&f_transform);
        addInput(&f_mask);
        addInput(&f_maskLabels);
        addOutput(&f_position);
        addOutput(&f_region);
        addOutput(&f_error);
        setDirtyValue();


        this->getContext()->get( deformationMapping, core::objectmodel::BaseContext::Local);
    }

    virtual void reinit() { update(); }

    virtual void bwdInit() {  updateMapping(); }

protected:
    ImageGaussPointSampler()    :   Inherit()
      , f_index(initData(&f_index,IndTypes(),"indices","image of dof indices"))
      , f_w(initData(&f_w,DistTypes(),"weights","weight image"))
      , f_mask(initData(&f_mask,MaskTypes(),"mask","optional mask to restrict the sampling region"))
      , f_maskLabels(initData(&f_maskLabels,"maskLabels","Mask labels where sampling is restricted"))
      , f_transform(initData(&f_transform,TransformType(),"transform",""))
      , f_region(initData(&f_region,IndTypes(),"region","sample region : labeled image with sample indices"))
      , f_error(initData(&f_error,DistTypes(),"error","weigth fitting error"))
      , f_clearData(initData(&f_clearData,true,"clearData","clear region and error images after computation"))
      , targetNumber(initData(&targetNumber,(unsigned int)0,"targetNumber","target number of samples"))
      , useDijkstra(initData(&useDijkstra,true,"useDijkstra","Use Dijkstra for geodesic distance computation (use fastmarching otherwise)"))
      , iterations(initData(&iterations,(unsigned int)100,"iterations","maximum number of Lloyd iterations"))
      , evaluateShapeFunction(initData(&evaluateShapeFunction,true,"evaluateShapeFunction","evaluate shape functions over integration regions for the mapping? (otherwise they will be interpolated at sample locations)"))
      , sampleRigidParts(initData(&sampleRigidParts,false,"sampleRigidParts","sample parts influenced only by one dofs? (otherwise put only one Gauss point)"))
      , f_fillOrder(initData(&f_fillOrder,(unsigned int)2,"fillOrder","fill order"))
      , deformationMapping(NULL)
    {
    }

    virtual ~ImageGaussPointSampler()
    {
        // what is that?
        f_index.setReadOnly(true);
        f_w.setReadOnly(true);
        f_mask.setReadOnly(true);
    }

    /** @name  region types */
    //@{
    typedef Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>  Matrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  Vector;
    typedef typename std::set<unsigned int> indList;  ///< list of parent indices
    typedef typename indList::iterator indListIt;
    typedef typename defaulttype::PolynomialFitFactors<Real> factType;
    helper::vector<factType> Reg;  ///< data related to each voronoi region
    //@}

    // polynomial orders
    inline unsigned int fillOrder() const {return f_fillOrder.getValue();} // For the mapping, we use second order fit (to have translation invariance of elastons, use first order)
    inline unsigned int fitOrder() const {return (this->f_order.getValue()==1)?0:1;} // for elastons, we measure the quality of the integration using first order least squares fit
    inline unsigned int volOrder() const {return (this->f_order.getValue()==1)?0:4;} // for elastons, we generate volume moments up to order 4

    static const int spatial_dimensions=3;
    mapping::BasePointMapper<spatial_dimensions,Real>* deformationMapping; ///< link to local deformation mapping for weights update

    virtual void update()
    {
        updateAllInputsIfDirty(); // the easy way...

        cleanDirty();

        ImageGaussPointSamplerSpec::init(this);
        ImageGaussPointSamplerSpec::Cluster_SimilarIndices(this);

        if(this->f_order.getValue()==1)                                     ImageGaussPointSamplerSpec::midpoint(this);
        else if(this->f_method.getValue().getSelectedId() == GAUSSLEGENDRE) serr<<"GAUSSLEGENDRE quadrature not yet implemented"<<sendl;
        else if(this->f_method.getValue().getSelectedId() == NEWTONCOTES)   serr<<"NEWTONCOTES quadrature not yet implemented"<<sendl;
        else if(this->f_method.getValue().getSelectedId() == ELASTON)       this->elaston();

        this->fitWeights();

        if(this->f_clearData.getValue())
        {
            waDist err(this->f_error); err->clear();
            waInd reg(this->f_region); reg->clear();
        }

        this->updateMapping();

        if(this->f_printLog.getValue()) if(this->f_position.getValue().size())    std::cout<<this->getName()<<": "<< this->f_position.getValue().size() <<" generated samples"<<std::endl;
    }



    /// elaston integration : put samples so as to maximize weight linearity

    void elaston()
    {
        if (this->Reg.size() == 0) return;
        // retrieve data
        waPositions pos(this->f_position);

        // fit weights
        for(unsigned int i=pos.size(); i<this->Reg.size(); i++)  {ImageGaussPointSamplerSpec::fillPolynomialFactors(this,i); this->Reg[i].solve(this->fitOrder());}

        // subdivide region with largest error until target number is reached
        while(this->Reg.size()<this->targetNumber.getValue())
        {
            Real maxerr=-1;
            unsigned int maxindex=0;
            for(unsigned int i=0; i<this->Reg.size(); i++) { Real e=this->Reg[i].getError(); if(maxerr<e) {maxerr=e; maxindex=i;} }
            if(maxerr==0) break;
            ImageGaussPointSamplerSpec::subdivideRegion(this,maxindex);

            ImageGaussPointSamplerSpec::fillPolynomialFactors(this,maxindex); this->Reg[maxindex].solve(this->fitOrder());
            ImageGaussPointSamplerSpec::fillPolynomialFactors(this,this->Reg.size()-1);    this->Reg.back().solve(this->fitOrder());
        }
    }


    /// fit weights to obtain final weights and derivatives
    /// optionaly write error image
    void fitWeights()
    {
        Real err=0;
        for(unsigned int i=0; i<this->Reg.size(); i++)
        {
            ImageGaussPointSamplerSpec::fillPolynomialFactors(this,i,!this->f_clearData.getValue());
            this->Reg[i].solve(this->fitOrder());
            err+=this->Reg[i].getError();
            //if(this->f_printLog.getValue()) std::cout<<this->getName()<<"GaussPointSampler: weight fitting error on sample "<<i<<" = "<<this->Reg[i].getError()<< std::endl;
        }
        //        waDist werr(this->f_error);        typename DistTypes& errimg = werr.wref();
        //        cimg_forXYZ(errimg,x,y,z) if(errimg(x,y,z)==-1) errimg(x,y,z)=0; // clean error output image (used as a container for distances)
        if(this->f_printLog.getValue()) std::cout<<this->getName()<<": total error = "<<err<<std::endl;
    }

    /// update mapping with weights fitted over a region (order 2)
    /// typically done in bkwinit (to overwrite weights computed in the mapping using shape function interpolation)
    virtual void updateMapping()
    {
        unsigned int nb = Reg.size();

        waPositions pos(this->f_position);
        waVolume vol(this->f_volume);
        helper::WriteOnlyAccessor<Data< VTransform > > transforms(this->f_transforms);

        pos.resize ( nb );
        vol.resize ( nb );
        transforms.resize ( nb );

        helper::vector<helper::vector<unsigned int> > index(nb);
        helper::vector<helper::vector<Real> > w(nb);
        helper::vector<helper::vector<defaulttype::Vec<spatial_dimensions,Real> > > dw(nb);
        helper::vector<helper::vector<defaulttype::Mat<spatial_dimensions,spatial_dimensions,Real> > > ddw(nb);

        for(unsigned int i=0; i<nb; i++)
        {
            factType* reg=&Reg[i];

            reg->solve(fillOrder());
            pos[i]=reg->center;
            vol[i].resize(reg->vol.rows());  for(unsigned int j=0; j<vol[i].size(); j++) vol[i][j]=reg->vol(j);
            reg->getMapping(index[i],w[i],dw[i],ddw[i]);
            // set sample orientation to identity (could be image orientation)
            transforms[i].identity();
        }

        // test
        /*for(unsigned int i=0; i<nb; i++)
        {
            Real sumw=0; for(unsigned int j=0; j<w[i].size(); j++) { sumw+=w[i][j]; }
            Vec<spatial_dimensions,Real>  sumdw; for(unsigned int j=0; j<dw[i].size(); j++) sumdw+=dw[i][j];
            if(sumdw.norm()>1E-2 || fabs(sumw-1)>1E-2) std::cout<<"error on "<<i<<" : "<<sumw<<","<<sumdw<<std::endl;
        }*/

        if(evaluateShapeFunction.getValue())
        {
            if(this->f_printLog.getValue())  std::cout<<this->getName()<<" : "<< nb <<" gauss points exported"<<std::endl;
            if(!deformationMapping) {serr<<"deformationMapping not found -> cannot map Gauss points"<<sendl; return;}
            else deformationMapping->resizeOut(pos.ref(),index,w,dw,ddw,transforms.ref());
        }
    }



};

}
}
}

#endif
