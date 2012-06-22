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
#ifndef FLEXIBLE_DiffusionShapeFunction_H
#define FLEXIBLE_DiffusionShapeFunction_H

#include "../initFlexible.h"
#include "../shapeFunction/BaseShapeFunction.h"
#include "../types/PolynomialBasis.h"

#include "ImageTypes.h"
#include "ImageAlgorithms.h"

#include <sofa/helper/OptionsGroup.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>

#define ISOTROPIC 0
#define ANISOTROPIC 1

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
Shape functions computed using heat diffusion in images
  */

template <class ShapeFunctionTypes_,class ImageTypes_>
class DiffusionShapeFunction : public BaseShapeFunction<ShapeFunctionTypes_>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DiffusionShapeFunction, ShapeFunctionTypes_,ImageTypes_) , SOFA_TEMPLATE(BaseShapeFunction, ShapeFunctionTypes_));
    typedef BaseShapeFunction<ShapeFunctionTypes_> Inherit;

    /** @name  Shape function types */
    //@{
    typedef typename Inherit::Real Real;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::VCoord VCoord;
    enum {material_dimensions=Inherit::material_dimensions};
    typedef typename Inherit::VReal VReal;
    typedef typename Inherit::VGradient VGradient;
    typedef typename Inherit::VHessian VHessian;
    typedef typename Inherit::VRef VRef;

    typedef typename Inherit::Gradient Gradient;
    typedef typename Inherit::Hessian Hessian;
    enum {spatial_dimensions=Inherit::spatial_dimensions};
    typedef typename Inherit::MaterialToSpatial MaterialToSpatial;
    typedef typename Inherit::VMaterialToSpatial VMaterialToSpatial;
    //@}

    /** @name  Image data */
    //@{
    typedef ImageTypes_ ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    typedef Real DistT;
    typedef defaulttype::Image<DistT> DistTypes;
    typedef helper::ReadAccessor<Data< DistTypes > > raDist;
    typedef helper::WriteAccessor<Data< DistTypes > > waDist;
    Data< DistTypes > f_distances;
    Data< DistTypes > f_w;

    typedef unsigned int IndT;
    typedef defaulttype::Image<IndT> IndTypes;
    typedef helper::ReadAccessor<Data< IndTypes > > raInd;
    typedef helper::WriteAccessor<Data< IndTypes > > waInd;
    Data< IndTypes > f_index;
    //@}

    /** @name  Options */
    //@{
    Data<helper::OptionsGroup> method;
    Data<bool> biasDistances;
    Data<bool> averageInRegion;
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const DiffusionShapeFunction<ShapeFunctionTypes_,ImageTypes_>* = NULL) { return ShapeFunctionTypes_::Name()+std::string(",")+ImageTypes_::Name(); }

    /// interpolate weights and their derivatives at a spatial position
    void computeShapeFunction(const Coord& childPosition, MaterialToSpatial& M, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL)
    {
        // resize input
        unsigned int nbRef=this->f_nbRef.getValue();
        ref.resize(nbRef); ref.fill(0);
        w.resize(nbRef); w.fill(0);
        if(dw) { dw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*dw)[j].fill(0); }
        if(ddw) { ddw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*ddw)[j].fill(0); }

        // get transform
        raTransform inT(this->transform);

        // material to world transformation = image orientation
        helper::Quater<Real> q = helper::Quater< Real >::createQuaterFromEuler(inT->getRotation() * (Real)M_PI / (Real)180.0);
        Mat<3,3,Real> R; q.toMatrix(R);
        for ( unsigned int i = 0; i < spatial_dimensions; i++ )  for ( unsigned int j = 0; j < material_dimensions; j++ ) M[i][j]=R[i][j];

        // get precomputed indices and weights
        raInd indData(this->f_index);
        raDist weightData(this->f_w);
        if(!indData->getCImgList().size() || !weightData->getCImgList().size()) { serr<<"Weights not available"<<sendl; return; }

        const CImg<IndT>& indices = indData->getCImg();
        const CImg<DistT>& weights = weightData->getCImg();

        // interpolate weights in neighborhood
        Coord p = inT->toImage(childPosition);
        Coord P;  for (unsigned int j=0; j<3; j++)  P[j]=round(p[j]);
        unsigned int order=0;
        /*if(ddw) order=2; else */  // do not use order 2 for local weight interpolation. Order two is used only in weight fitting over regions
        if(dw) order=1;

        //get closest voxel with non zero weights
        bool project=false;
        if(P[0]<0 || P[1]<0 || P[2]<0 || P[0]>indices.width()-1 || P[1]>indices.height()-1 || P[2]>indices.depth()-1) project=true;
        else if(indices(P[0],P[1],P[2],0)==0) project=true;
        if(project)
        {
            Real dmin=cimg::type<Real>::max();
            Coord newP=P;
            cimg_forXYZ(indices,x,y,z) if(indices(x,y,z,0)) {Real d=(Coord(x,y,z)-p).norm2(); if(d<dmin) { newP=Coord(x,y,z); dmin=d; } }
            if(dmin==cimg::type<Real>::max()) return;
            P=newP;
        }

        // prepare neighborood
        sofa::defaulttype::Vec<27,  Coord > lpos;      // precomputed local positions
        int count=0;
        for (int k=-1; k<=1; k++) for (int j=-1; j<=1; j++) for (int i=-1; i<=1; i++) lpos[count++]= inT->fromImage(P+Coord(i,j,k)) - childPosition;

        // get indices at P
        int index=0;
        for (unsigned int r=0; r<nbRef; r++)
        {
            IndT ind=indices(P[0],P[1],P[2],r);
            if(ind>0)
            {
                vector<DistT> val; val.reserve(27);
                vector<Coord> pos; pos.reserve(27);
                // add neighbors with same index
                count=0;
                for (int k=-1; k<=1; k++) for (int j=-1; j<=1; j++) for (int i=-1; i<=1; i++)
                        {
                            for (unsigned int r2=0; r2<nbRef; r2++)
                                if(indices(P[0]+i,P[1]+j,P[2]+k,r2)==ind)
                                {
                                    val.push_back(weights(P[0]+i,P[1]+j,P[2]+k,r2));
                                    pos.push_back(lpos[count]);
                                }
                            count++;
                        }
                // fit weights
                vector<Real> coeff;
                defaulttype::PolynomialFit(coeff,val,pos, order);
                //std::cout<<ind<<":"<<coeff[0]<<", err= "<<getPolynomialFit_Error(coeff,val,pos)<< std::endl;
                if(!dw) defaulttype::getPolynomialFit_differential(coeff,w[index]);
                else if(!ddw) defaulttype::getPolynomialFit_differential(coeff,w[index],&(*dw)[index]);
                else defaulttype::getPolynomialFit_differential(coeff,w[index],&(*dw)[index],&(*ddw)[index]);
                ref[index]=ind-1;
                if(w[index]<0) // clamp negative weights
                {
                    w[index]=0;
                    if(dw) (*dw)[index].fill(0);
                    if(ddw) (*ddw)[index].fill(0);
                }
                index++;

            }
        }

        // normalize
        this->normalize(w,dw,ddw);
    }


    /// fit weights and their derivatives in gauss point regions
    void computeShapeFunction(const VCoord& childPosition, VMaterialToSpatial& M, vector<VRef>& ref, vector<VReal>& w, vector<VGradient>& dw,vector<VHessian>& ddw, const unsigned int* region)
    {
        if(!region || !this->averageInRegion.getValue())
            Inherit::computeShapeFunction(childPosition,M,ref,w,dw,ddw);        // weight averaging over a region not supported -> get interpolated values
        else
        {
            // get precomputed indices and weights
            raInd indData(this->f_index);
            if(!indData->getCImgList().size()) { serr<<"Weights not available"<<sendl; return; }
            const CImg<IndT>& indices = indData->getCImg();
            // cast region into a get shared memory image
            const CImg<unsigned int> reg(region,indices.width(),indices.height(),indices.depth(),1,true);
            // compute
            unsigned int nb=childPosition.size();
            M.resize(nb); ref.resize(nb);        w.resize(nb);   dw.resize(nb);  ddw.resize(nb);
            for(unsigned i=0; i<nb; i++)   computeShapeFunctionInRegion(i,reg,childPosition[i],M[i],ref[i],w[i],&dw[i],&ddw[i]);
        }
    }

    void computeShapeFunctionInRegion(const unsigned int index, const CImg<unsigned int>& region,const Coord& childPosition, MaterialToSpatial& M, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL)
    {
        // resize input
        unsigned int nbRef=this->f_nbRef.getValue();
        ref.resize(nbRef); ref.fill(0);
        w.resize(nbRef); w.fill(0);
        if(dw) { dw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*dw)[j].fill(0); }
        if(ddw) { ddw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*ddw)[j].fill(0); }

        // get transform
        raTransform inT(this->transform);

        // material to world transformation = image orientation
        helper::Quater<Real> q = helper::Quater< Real >::createQuaterFromEuler(inT->getRotation() * (Real)M_PI / (Real)180.0);
        Mat<3,3,Real> R; q.toMatrix(R);
        for ( unsigned int i = 0; i < spatial_dimensions; i++ )  for ( unsigned int j = 0; j < material_dimensions; j++ ) M[i][j]=R[i][j];

        // get precomputed indices and weights
        raInd indData(this->f_index);
        raDist weightData(this->f_w);
        if(!indData->getCImgList().size() || !weightData->getCImgList().size()) { serr<<"Weights not available"<<sendl; return; }

        const CImg<IndT>& indices = indData->getCImg();
        const CImg<DistT>& weights = weightData->getCImg();

        // fit weights in region
        unsigned int order=0; if(ddw) order=2; else if(dw) order=1;

        // get neighborood
        vector<Coord> pi;
        cimg_forXYZ(region,x,y,z) if(region(x,y,z)==index+1) pi.push_back( inT->fromImage(Coord(x,y,z)) - childPosition );
        unsigned int nbs=pi.size();

        // get indices
        typedef std::map<unsigned int, vector<DistT> > wiMap;
        typedef typename wiMap::iterator wiMapIt;
        wiMap wi;
        unsigned int count =0;
        cimg_forXYZ(region,x,y,z) if(region(x,y,z)==index+1)
        {
            for (unsigned int v=0; v<nbRef; v++)
            {
                IndT ind=indices(x,y,z,v);
                if(ind>0)
                {
                    wiMapIt wIt=wi.find(ind);
                    if(wIt==wi.end()) {wi[ind]=vector<Real>((int)nbs,0); wIt=wi.find(ind);}
                    wIt->second[count]=weights(x,y,z,v);
                }
            }
            count++;
        }
        // clamp to nbref weights
        while(wi.size()>nbRef)
        {
            DistT nmin = cimg::type<DistT>::max();
            wiMapIt wItMin;
            for(wiMapIt wIt=wi.begin(); wIt!=wi.end(); wIt++)
            {
                DistT n=0; for (unsigned int i=0; i<nbs; i++) n+=wIt->second[i];
                if(nmin>n) { nmin=n; wItMin=wIt;}
            }
            wi.erase(wItMin);
        }

        // fit
        count =0;
        Real totalerr=0;
        for(wiMapIt wIt=wi.begin(); wIt!=wi.end(); wIt++)
        {
            vector<Real> coeff;
            defaulttype::PolynomialFit(coeff,wIt->second,pi,order);
            Real err = defaulttype::getPolynomialFit_Error(coeff,wIt->second,pi);
            totalerr+=err;
            if(!dw) defaulttype::getPolynomialFit_differential(coeff,w[count]);
            else if(!ddw) defaulttype::getPolynomialFit_differential(coeff,w[count],&(*dw)[count]);
            else defaulttype::getPolynomialFit_differential(coeff,w[count],&(*dw)[count],&(*ddw)[count]);
            ref[count]=wIt->first-1;

            if(w[count]<0) // clamp negative weights
            {
                w[count]=0;
                if(dw) (*dw)[count].fill(0);
                if(ddw) (*ddw)[count].fill(0);
            }

            count++;
        }
        //if(this->f_printLog.getValue()) std::cout<<"VoronoiShapeFunction: weight fitting error on sample "<<index<<" = "<<totalerr<<std::endl;

        // normalize
        this->normalize(w,dw,ddw);
    }



    virtual void init()
    {
        Inherit::init();

        helper::ReadAccessor<Data<vector<Coord> > > parent(this->f_position);
        if(!parent.size()) { serr<<"Parent nodes not found"<<sendl; return; }
        /*
                // get tranform and image at time t
                raImage in(this->image);
                raTransform inT(this->transform);
                if(!in->getCImgList().size())  { serr<<"Image not found"<<sendl; return; }
                const CImg<T>& inimg = in->getCImg(0);  // suppose time=0
                const CImg<T>* biasFactor=biasDistances.getValue()?&inimg:NULL;
                const Vec<3,Real>& voxelsize=this->transform.getValue().getScale();

                // init distances
                imCoord dim = in->getDimensions(); dim[3]=dim[4]=1;

                waDist distData(this->f_distances);         distData->setDimensions(dim);
                CImg<DistT>& dist = distData->getCImg(); dist.fill(-1);
                cimg_forXYZC(inimg,x,y,z,c) if(inimg(x,y,z,c)) dist(x,y,z)=cimg::type<DistT>::max();

                // compute Diffusion and distances based on nodes
               typedef sofa::defaulttype::Vec<3,int> iCoord;
                typedef std::pair<Real,iCoord > DistanceToPoint;
                std::set<DistanceToPoint> trial;                // list of seed points
                for(unsigned int i=0;i<parent.size();i++) AddSeedPoint<Real>(trial,dist,Diffusion, inT.ref(), parent[i], i+1);

                if(useDijkstra.getValue()) dijkstra<Real,T>(trial,dist, Diffusion, voxelsize , biasFactor);
                else fastMarching<Real,T>(trial,dist, Diffusion, voxelsize ,biasFactor );

                // init indices and weights images
                unsigned int nbref=this->f_nbRef.getValue();
                dim[3]=nbref;
                waInd indData(this->f_index); indData->setDimensions(dim);
                CImg<IndT>& indices = indData->getCImg(); indices.fill(0);

                waDist weightData(this->f_w);         weightData->setDimensions(dim);
                CImg<DistT>& weights = weightData->getCImg(); weights.fill(0);

                // compute weights
                if(this->method.getValue().getSelectedId() == ISOTROPIC)
                {

                }
                else
                {

                }
        */

    }

protected:
    DiffusionShapeFunction()
        :Inherit()
        , image(initData(&image,ImageTypes(),"image",""))
        , transform(initData(&transform,TransformType(),"transform",""))
        , f_distances(initData(&f_distances,DistTypes(),"distances",""))
        , f_w(initData(&f_w,DistTypes(),"weights",""))
        , f_index(initData(&f_index,IndTypes(),"indices",""))
        , method ( initData ( &method,"method","method (param)" ) )
        , biasDistances(initData(&biasDistances,false,"bias","Bias distances using inverse pixel values"))
        , averageInRegion(initData(&averageInRegion,true,"averageInRegion","average shape function in Gauss point region; interpolate otherwise."))
    {
        image.setReadOnly(true);
        transform.setReadOnly(true);

        helper::OptionsGroup methodOptions(3,"0 - Isotropic"
                ,"1 - Anisotropic"
                                          );
        methodOptions.setSelectedItem(ISOTROPIC);
        method.setValue(methodOptions);

        image.setGroup("input");
        transform.setGroup("input");
        f_distances.setGroup("output");
        f_w.setGroup("output");
        f_index.setGroup("output");
        method.setGroup("parameters");
        biasDistances.setGroup("parameters");
        averageInRegion.setGroup("parameters");
    }

    virtual ~DiffusionShapeFunction()
    {

    }


};


}
}
}


#endif
