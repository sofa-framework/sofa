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

    typedef defaulttype::Image<bool> MaskTypes;
    typedef helper::ReadAccessor<Data< MaskTypes > > raMask;
    Data< MaskTypes > f_mask;

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

    virtual void init()
    {
        Inherit::init();

        addInput(&f_index);
        addInput(&f_w);
        addInput(&f_transform);
        addInput(&f_mask);
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
      , f_transform(initData(&f_transform,TransformType(),"transform",""))
      , f_region(initData(&f_region,IndTypes(),"region","sample region : labeled image with sample indices"))
      , f_error(initData(&f_error,DistTypes(),"error","weigth fitting error"))
      , targetNumber(initData(&targetNumber,(unsigned int)0,"targetNumber","target number of samples"))
      , useDijkstra(initData(&useDijkstra,true,"useDijkstra","Use Dijkstra for geodesic distance computation (use fastmarching otherwise)"))
      , iterations(initData(&iterations,(unsigned int)100,"iterations","maximum number of Lloyd iterations"))
      , deformationMapping(NULL)
    {
    }

    virtual ~ImageGaussPointSampler()
    {

    }

    /** @name  region types */
    //@{
    typedef Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>  Matrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  Vector;
    typedef typename std::set<unsigned int> indList;  ///< list of parent indices
    typedef typename indList::iterator indListIt;
    vector<defaulttype::PolynomialFitFactors<Real> > Reg;  ///< data related to each voronoi region
    //@}

    // polynomial orders
    inline unsigned int fillOrder() const {return 1;}     // For the mapping, we use first order fit (not 2nd order, ot have translation invariance of elastons)
    inline unsigned int fitOrder() const {return (this->f_order.getValue()==1)?0:1;} // for elastons, we measure the quality of the integration using first order least squares fit
    inline unsigned int volOrder() const {return (this->f_order.getValue()==1)?0:4;} // for elastons, we generate volume moments up to order 4

    static const int spatial_dimensions=3;
    mapping::BasePointMapper<spatial_dimensions,Real>* deformationMapping; ///< link to local deformation mapping for weights update

    virtual void update()
    {
        cleanDirty();

        // get tranform and images at time t
        raDist rweights(this->f_w);             if(rweights->isEmpty())  { serr<<"Weights not found"<<sendl; return; }
        raInd rindices(this->f_index);          if(rindices->isEmpty())  { serr<<"Indices not found"<<sendl; return; }
        raTransform transform(this->f_transform);
        const CImg<IndT>& indices = rindices->getCImg(0);  // suppose time=0

        imCoord dim = rweights->getDimensions();
        dim[3]=dim[4]=1; // remove nbchannels from dimensions (to allocate single channel images later)

//        raMask rmask(this->f_mask);
//        const CImg<bool>* mask = NULL;
//        if(!rmask->isEmpty()) mask=&rmask->getCImg();

        // get output data
        waPositions pos(this->f_position);
        waVolume vol(this->f_volume);
        pos.clear();                // pos is cleared since it is always initialized with one point, so user placed points are not allowed for now..

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
        Reg.clear();

        if(this->f_order.getValue()==1) // midpoint integration : put samples uniformly and weight them by their volume
        {
            // identify regions with similar repartitions
            Cluster_SimilarIndices();

            // init soft regions (=more than one parent) where uniform sampling will be done
            // rigid regions (one parent) are kept in the list of region (and dist remains=-1 so they will not be sampled)
            for(unsigned int i=0; i<Reg.size();i++)
            {
                if(Reg[i].parentsToNodeIndex.size()>1)
                {
                    cimg_forXYZ(regimg,x,y,z)
                            if(regimg(x,y,z)==*(Reg[i].voronoiIndices.begin()) )
                    {
                        dist(x,y,z)=cimg::type<DistT>::max();
                        regimg(x,y,z)=0;
                    }
                    Reg.erase (Reg.begin()+i); i--;  //  erase region (soft regions will be generated after uniform sampling)
                }
            }
            unsigned int nbrigid = Reg.size();


            // fixed points = points set by the user in soft regions.
            // Disabled for now since pos is cleared
            SeqPositions fpos_voxelIndex;
            vector<unsigned int> fpos_voronoiIndex;
            for(unsigned int i=0; i<pos.size(); i++)
            {
                Coord p = transform->toImage(pos[i]);
                for (unsigned int j=0; j<3; j++)  p[j]=sofa::helper::round(p[j]);
                if(indices.containsXYZC(p[0],p[1],p[2]))
                {
                    indList l;
                    cimg_forC(indices,v) if(indices(p[0],p[1],p[2],v)) l.insert(indices(p[0],p[1],p[2],v)-1);
                    if(l.size()>1) { fpos_voxelIndex.push_back(p); fpos_voronoiIndex.push_back(i+1+nbrigid); }
                }
            }

            // target nb of points
            unsigned int nb = (fpos_voxelIndex.size()+nbrigid>targetNumber.getValue())?fpos_voxelIndex.size()+nbrigid:targetNumber.getValue();
            unsigned int nbsoft = nb-nbrigid;
            if(this->f_printLog.getValue()) std::cout<<this->getName()<<": Number of rigid/soft regions : "<<nbrigid<<"/"<<nbsoft<< std::endl;

            // init seeds for uniform sampling
            std::set<std::pair<DistT,sofa::defaulttype::Vec<3,int> > > trial;

            // farthest point sampling using geodesic distances
            SeqPositions newpos_voxelIndex;
            vector<unsigned int> newpos_voronoiIndex;

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
                converged=!(Lloyd<DistT>(newpos_voxelIndex,newpos_voronoiIndex,regimg));
                // recompute voronoi
                cimg_foroff(dist,off) if(dist[off]!=-1) dist[off]=cimg::type<DistT>::max();
                for(unsigned int i=0; i<fpos_voxelIndex.size(); i++) AddSeedPoint<DistT>(trial,dist,regimg, fpos_voxelIndex[i],fpos_voronoiIndex[i]);
                for(unsigned int i=0; i<newpos_voxelIndex.size(); i++) AddSeedPoint<DistT>(trial,dist,regimg, newpos_voxelIndex[i],newpos_voronoiIndex[i]);
                if(useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
                else fastMarching<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
                it++; if(it>=iterations.getValue()) converged=true;
            }

            if(this->f_printLog.getValue()) std::cout<<this->getName()<<": Completed in "<< it <<" Lloyd iterations"<<std::endl;

            // create soft regions and update teir data
            for(unsigned int i=0; i<fpos_voxelIndex.size(); i++)           // Disabled for now since fpos is empty
            {
                indList l; cimg_forXYZ(regimg,x,y,z) if(regimg(x,y,z)==fpos_voronoiIndex[i]) { cimg_forC(indices,v) if(indices(x,y,z,v)) l.insert(indices(x,y,z,v)-1); }   // collect indices over the region
                if(l.size())
                {
                    defaulttype::PolynomialFitFactors<Real> reg(l,fpos_voronoiIndex[i]); reg.center=transform->fromImage(fpos_voxelIndex[i]);
                    Reg.push_back(reg);
                }
            }
            for(unsigned int i=0; i<newpos_voxelIndex.size(); i++)
            {
                indList l; cimg_forXYZ(regimg,x,y,z) if(regimg(x,y,z)==newpos_voronoiIndex[i]) { cimg_forC(indices,v) if(indices(x,y,z,v)) l.insert(indices(x,y,z,v)-1); }   // collect indices over the region
                if(l.size())
                {
                    defaulttype::PolynomialFitFactors<Real> reg(l,newpos_voronoiIndex[i]); reg.center=transform->fromImage(newpos_voxelIndex[i]);
                    Reg.push_back(reg);
                }
            }
            // update rigid regions (might contain soft material due to voronoi proximity)
            for(unsigned int i=0; i<nbrigid; i++)
            {
                indList l; cimg_forXYZ(regimg,x,y,z) if(regimg(x,y,z)==*(Reg[i].voronoiIndices.begin()) ) { cimg_forC(indices,v) if(indices(x,y,z,v)) l.insert(indices(x,y,z,v)-1); }   // collect indices over the region
                Reg[i].setParents(l);
            }

            // fit weights
            Real err=0;
            for(unsigned int i=0; i<Reg.size(); i++)
            {
                Reg[i].nb=0; cimg_foroff(regimg,off)  if(regimg(off) == *(Reg[i].voronoiIndices.begin())) Reg[i].nb++;
                fillPolynomialFactors(Reg[i],true);
                Reg[i].solve(fitOrder());
                err+=Reg[i].getError();
                //if(this->f_printLog.getValue()) std::cout<<this->getName()<<"GaussPointSampler: weight fitting error on sample "<<i<<" = "<<Reg[i].getError()<< std::endl;
            }
            if(this->f_printLog.getValue()) std::cout<<this->getName()<<": total error = "<<err<<std::endl;
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
                Cluster_SimilarIndices();

                // fit weights
                for(unsigned int i=pos.size(); i<Reg.size(); i++)  {fillPolynomialFactors(Reg[i]); Reg[i].solve(fitOrder());}

                // subdivide region with largest error until target number is reached
                while(Reg.size()<targetNumber.getValue())
                {
                    Real maxerr=-1;
                    unsigned int maxindex=0;
                    for(unsigned int i=0; i<Reg.size(); i++) { Real e=Reg[i].getError(); if(maxerr<e) {maxerr=e; maxindex=i;} }
                    if(maxerr==0) break;
                    subdivideRegion(maxindex);

                    fillPolynomialFactors(Reg[maxindex]); Reg[maxindex].solve(fitOrder());
                    fillPolynomialFactors(Reg.back());    Reg.back().solve(fitOrder());
                }

                // fit weights
                Real err=0;
                for(unsigned int i=0; i<Reg.size(); i++)
                {
                    fillPolynomialFactors(Reg[i],true);
                    err+=Reg[i].getError();
                    //if(this->f_printLog.getValue()) std::cout<<this->getName()<<": weight fitting error on sample "<<i<<" = "<<Reg[i].getError()<< std::endl;
                }
                if(this->f_printLog.getValue()) std::cout<<this->getName()<<": total error = "<<err<<std::endl;
            }
        }

        cimg_forXYZ(dist,x,y,z) if(dist(x,y,z)==-1) dist(x,y,z)=0; // clean error output image (used as a container for distances)

        updateMapping();

        if(this->f_printLog.getValue()) if(pos.size())    std::cout<<this->getName()<<": "<< pos.size() <<" generated samples"<<std::endl;
    }


    /// Identify regions sharing similar parents
    /// returns a list of region containing the parents, the number of voxels and center; and fill the voronoi image
    void Cluster_SimilarIndices()
    {
        // get tranform and images at time t
        raInd rindices(this->f_index);          if(rindices->isEmpty())  { serr<<"Indices not found"<<sendl; return; }
        raTransform transform(this->f_transform);
        const CImg<IndT>& indices = rindices->getCImg(0);  // suppose time=0

        raMask rmask(this->f_mask);
        const CImg<bool>* mask = NULL;
        if(!rmask->isEmpty()) mask=&rmask->getCImg();

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
            for (unsigned int j=0; j<3; j++)  p[j]=sofa::helper::round(p[j]);
            if(indices.containsXYZC(p[0],p[1],p[2]))
            {
                indList l;
                cimg_forC(indices,v) if(indices(p[0],p[1],p[2],v)) l.insert(indices(p[0],p[1],p[2],v)-1);
                List[l]=i;
                Reg.push_back(defaulttype::PolynomialFitFactors<Real>(l,i+1));
                regimg(p[0],p[1],p[2])=i+1;
            }
        }

        // traverse index image to identify regions with unique indices
        cimg_forXYZ(indices,x,y,z)
                if(indices(x,y,z))
                    if(!mask || (*mask)(x,y,z))
        {
            indList l;
            cimg_forC(indices,v) if(indices(x,y,z,v)) l.insert(indices(x,y,z,v)-1);
            indMap::iterator it=List.find(l);
            unsigned int index;
            if(it==List.end()) { index=List.size(); List[l]=index;  Reg.push_back(defaulttype::PolynomialFitFactors<Real>(l,index+1)); Reg.back().nb=1; }
            else { index=it->second; Reg[index].nb++;}

            Reg[index].center+=transform->fromImage(Coord(x,y,z));
            regimg(x,y,z)=*(Reg[index].voronoiIndices.begin());
        }

        // average to get centroid (may not be inside the region if not convex)
        for(unsigned int i=0; i<Reg.size(); i++) Reg[i].center/=(Real)Reg[i].nb;
    }

    /// subdivide region[index] in two regions
    void subdivideRegion(const unsigned int index)
    {
        raTransform transform(this->f_transform);
        raInd rindices(this->f_index);

        waInd wreg(this->f_region);
        CImg<IndT>& regimg = wreg->getCImg(0);

        waDist werr(this->f_error);
        CImg<DistT>& dist = werr->getCImg(0);

        vector<Coord> pos(2);
        vector<unsigned int> vorindex;
        vorindex.push_back(*(Reg[index].voronoiIndices.begin()));
        vorindex.push_back(Reg.size()+1);
        for(unsigned int i=0; i<Reg.size(); i++) if(vorindex[1]==*(Reg[i].voronoiIndices.begin())) vorindex[1]++; // check that the voronoi index is unique. not necessary in principle

        // get closest/farthest point from c and init distance image
        Real dmin=cimg::type<Real>::max(),dmax=0;
        cimg_forXYZ(regimg,x,y,z)
                if(regimg(x,y,z)==vorindex[0])
        {
            dist(x,y,z)=cimg::type<DistT>::max();
            Coord p = Coord(x,y,z);
            Real d = (transform->fromImage(p)-Reg[index].center).norm2();
            if(dmin>d) {dmin=d; pos[0]=p;}
            if(dmax<d) {dmax=d; pos[1]=p;}
        }
        else dist(x,y,z)=(DistT)(-1);

        // Loyd relaxation
        std::set<std::pair<DistT,sofa::defaulttype::Vec<3,int> > > trial;
        unsigned int it=0;
        bool converged =(it>=iterations.getValue())?true:false;

        for(unsigned int i=0; i<2; i++) AddSeedPoint<DistT>(trial,dist,regimg, pos[i],vorindex[i]);
        if(useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
        else fastMarching<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
        //dist.display();
        //regimg.display();
        while(!converged)
        {
            converged=!(Lloyd<DistT>(pos,vorindex,regimg));
            // recompute voronoi
            cimg_foroff(dist,off) if(dist[off]!=-1) dist[off]=cimg::type<DistT>::max();
            for(unsigned int i=0; i<2; i++) AddSeedPoint<DistT>(trial,dist,regimg, pos[i],vorindex[i]);
            if(useDijkstra.getValue()) dijkstra<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
            else fastMarching<DistT,DistT>(trial,dist, regimg, this->f_transform.getValue().getScale());
            it++; if(it>=iterations.getValue()) converged=true;
        }

        // add region
        defaulttype::PolynomialFitFactors<Real> reg;
        reg.parentsToNodeIndex=Reg[index].parentsToNodeIndex;
        reg.voronoiIndices.insert(vorindex[1]);
        reg.center=transform->fromImage(pos[1]);
        reg.nb=0; cimg_foroff(regimg,off)  if(regimg(off) == vorindex[1]) reg.nb++;
        Reg.push_back(reg);

        // update old region data
        Reg[index].center=transform->fromImage(pos[0]);
        Reg[index].nb=0; cimg_foroff(regimg,off)  if(regimg(off) == vorindex[0]) Reg[index].nb++;
    }



    /// update Polynomial Factors from the voxel map
    void fillPolynomialFactors(defaulttype::PolynomialFitFactors<Real>& fact, const bool writeOutput=false)
    {
        // get tranform and images at time t
        raDist rweights(this->f_w);             if(rweights->isEmpty())  { serr<<"Weights not found"<<sendl; return; }
        raInd rindices(this->f_index);          if(rindices->isEmpty())  { serr<<"Indices not found"<<sendl; return; }
        raTransform transform(this->f_transform);
        const CImg<DistT>& weights = rweights->getCImg(0);  // suppose time=0
        const CImg<IndT>& indices = rindices->getCImg(0);  // suppose time=0
        const Real dv =  transform->getScale()[0] * transform->getScale()[1] * transform->getScale()[2];

        // get region image
        raInd rreg(this->f_region);
        const CImg<IndT>& regimg = rreg->getCImg(0);

        // list of absolute coords
        vector<Coord> pi(fact.nb);

        // weights (one line for each parent)
        Matrix wi(fact.parentsToNodeIndex.size(),fact.nb); wi.setZero();

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

        fact.fill(wi,pi,fillOrder(),dv,volOrder());

        //  std::cout<<"pt "<<*(fact.voronoiIndices.begin())-1<<" : "<<fact.center<<std::endl<<std::endl<<std::endl<<pi<<std::endl<<std::endl<<wi<<std::endl;
        //test: fact.directSolve(wi,pi); std::cout<<"Jacobi err="<<fact.getError()<<std::endl;

        // write error into output image
        if(writeOutput)
        {
            waDist werr(this->f_error);
            CImg<DistT>& outimg = werr->getCImg(0);
            count=0;
            cimg_forXYZ(regimg,x,y,z)
            {
                indListIt it=fact.voronoiIndices.find(regimg(x,y,z));
                if(it!=fact.voronoiIndices.end()) { outimg(x,y,z)=fact.getError(pi[count],wi.col(count)); count++; }
            }
        }
    }


    /// update mapping with weights fitted over a region (order 2)
    /// typically done in bkwinit (to overwrite weights computed in the mapping using shape function interpolation)
    virtual void updateMapping()
    {
        if(!deformationMapping) {serr<<"deformationMapping not found -> cannot map Gauss points"<<sendl; return;}

        unsigned int nb = Reg.size();

        waPositions pos(this->f_position);
        waVolume vol(this->f_volume);

        pos.resize ( nb );
        vol.resize ( nb );

        vector<vector<unsigned int> > index(nb);
        vector<vector<Real> > w(nb);
        vector<vector<Vec<spatial_dimensions,Real> > > dw(nb);
        vector<vector<Mat<spatial_dimensions,spatial_dimensions,Real> > > ddw(nb);

        Mat<spatial_dimensions,spatial_dimensions,Real> I=Mat<spatial_dimensions,spatial_dimensions,Real>::Identity(); // could be image orientation
        vector<Mat<spatial_dimensions,spatial_dimensions,Real> > F0((int)nb,I);

        for(unsigned int i=0; i<nb; i++)
        {
            defaulttype::PolynomialFitFactors<Real>* reg=&Reg[i];

            reg->solve(fillOrder());
            pos[i]=reg->center;
            vol[i].resize(reg->vol.rows());  for(unsigned int j=0; j<vol[i].size(); j++) vol[i][j]=reg->vol(j);
            reg->getMapping(index[i],w[i],dw[i],ddw[i]);
        }

        // test
        /*for(unsigned int i=0; i<nb; i++)
        {
            Real sumw=0; for(unsigned int j=0; j<w[i].size(); j++) { sumw+=w[i][j]; }
            Vec<spatial_dimensions,Real>  sumdw; for(unsigned int j=0; j<dw[i].size(); j++) sumdw+=dw[i][j];
            if(sumdw.norm()>1E-2 || fabs(sumw-1)>1E-2) std::cout<<"error on "<<i<<" : "<<sumw<<","<<sumdw<<std::endl;
        }*/

        if(this->f_printLog.getValue())  std::cout<<this->getName()<<" : "<< nb <<" gauss points exported"<<std::endl;
        deformationMapping->resizeOut(pos.ref(),index,w,dw,ddw,F0);
    }



};

}
}
}

#endif
