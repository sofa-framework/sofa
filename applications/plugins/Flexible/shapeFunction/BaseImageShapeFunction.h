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
#ifndef FLEXIBLE_BaseImageShapeFunction_H
#define FLEXIBLE_BaseImageShapeFunction_H

#include "../initFlexible.h"
#include "../shapeFunction/BaseShapeFunction.h"
#include "../types/PolynomialBasis.h"

#include <image/ImageTypes.h>
#include <image/BranchingImage.h>
#include <image/ImageAlgorithms.h>

#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>

namespace sofa
{
namespace component
{
namespace shapefunction
{

using core::behavior::BaseShapeFunction;
using defaulttype::Mat;
using defaulttype::Vec;





/// Default implementation does not compile
template <int imageTypeLabel>
struct BaseImageShapeFunctionSpecialization
{
};


/// Specialization for regular Image
template <>
struct BaseImageShapeFunctionSpecialization<defaulttype::IMAGELABEL_IMAGE>
{
    template<class BaseImageShapeFunction>
    static void constructor( BaseImageShapeFunction* bisf )
    {
        bisf->f_fineToCoarseMapping.setDisplayed( false );
        bisf->f_fineToCoarseMappingTransform.setDisplayed( false );
    }

    template<class BaseImageShapeFunction>
    static void computeShapeFunction( BaseImageShapeFunction* bisf, const typename BaseImageShapeFunction::Coord& childPosition, typename BaseImageShapeFunction::MaterialToSpatial& M, typename BaseImageShapeFunction::VRef& ref, typename BaseImageShapeFunction::VReal& w, typename BaseImageShapeFunction::VGradient* dw=NULL, typename BaseImageShapeFunction::VHessian* ddw=NULL )
    {
        typedef typename BaseImageShapeFunction::Real Real;
        typedef typename BaseImageShapeFunction::IndT IndT;
        typedef typename BaseImageShapeFunction::DistT DistT;
        typedef typename BaseImageShapeFunction::Coord Coord;

        // resize input
        unsigned int nbRef=bisf->f_nbRef.getValue();
        ref.resize(nbRef); ref.fill(0);
        w.resize(nbRef); w.fill(0);
        if(dw) { dw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*dw)[j].fill(0); }
        if(ddw) { ddw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*ddw)[j].fill(0); }

        // get transform
        typename BaseImageShapeFunction::raTransform inT(bisf->transform);

        // material to world transformation = image orientation
        helper::Quater<Real> q = helper::Quater< Real >::createQuaterFromEuler(inT->getRotation() * (Real)M_PI / (Real)180.0);
        Mat<3,3,Real> R; q.toMatrix(R);
        for ( unsigned int i = 0; i < BaseImageShapeFunction::spatial_dimensions; i++ )  for ( unsigned int j = 0; j < BaseImageShapeFunction::material_dimensions; j++ ) M[i][j]=R[i][j];

        // get precomputed indices and weights
        typename BaseImageShapeFunction::raInd indData(bisf->f_index);
        typename BaseImageShapeFunction::raDist weightData(bisf->f_w);
        if(!indData->getCImgList().size() || !weightData->getCImgList().size()) { bisf->serr<<"Weights not available"<<bisf->sendl; return; }

        const CImg<IndT>& indices = indData->getCImg();
        const CImg<DistT>& weights = weightData->getCImg();

        // interpolate weights in neighborhood
        Coord p = inT->toImage(childPosition);
        Coord P;  for (unsigned int j=0; j<3; j++)  P[j]=sofa::helper::round(p[j]);
        unsigned int order=0;
        /*if(ddw) order=2; else */  // do not use order 2 for local weight interpolation. Order two is used only in weight fitting over regions
        if(dw) order=1;

        //get closest voxel with non zero weights
        bool project=false;
        if(P[0]<=0 || P[1]<=0 || P[2]<=0 || P[0]>=indices.width()-1 || P[1]>=indices.height()-1 || P[2]>=indices.depth()-1) project=true;
        else if(indices(P[0],P[1],P[2],0)==0) project=true;
        if(project)
        {
            Real dmin=cimg::type<Real>::max();
            Coord newP=P;
            cimg_for_insideXYZ(indices,x,y,z,1) if(indices(x,y,z,0)) {Real d=(Coord(x,y,z)-p).norm2(); if(d<dmin) { newP=Coord(x,y,z); dmin=d; } }
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
                    for (unsigned int r2=0;r2<nbRef;r2++)
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
        bisf->normalize(w,dw,ddw);
    }
};


/// Specialization for Branching Image
template <>
struct BaseImageShapeFunctionSpecialization<defaulttype::IMAGELABEL_BRANCHINGIMAGE>
{
    template<class BaseImageShapeFunction>
    static void constructor( BaseImageShapeFunction* bisf )
    {
        bisf->addAlias( &bisf->image, "branchingImage" );
    }


    // @TODO implemented without any verifications -> needs to be checked when really used
    template<class BaseImageShapeFunction>
    static void computeShapeFunction( BaseImageShapeFunction* bisf, const typename BaseImageShapeFunction::Coord& childPosition, typename BaseImageShapeFunction::MaterialToSpatial& M, typename BaseImageShapeFunction::VRef& ref, typename BaseImageShapeFunction::VReal& w, typename BaseImageShapeFunction::VGradient* dw=NULL, typename BaseImageShapeFunction::VHessian* ddw=NULL )
    {

        // AN IDEA
        // rather than having an indice image + a weight image, it seems it would be possible to have only a weight branching image with connectivity, where connected voxels is similar to have the same indice
        // all same indice voxel are connected, right? So if connections in the weight image are only close neighbours, it should be enough
        // if using a indice image, no connectivity needs to be store in indice&weight images, but only superimposed voxel values



        typedef typename BaseImageShapeFunction::Real Real;
        typedef typename BaseImageShapeFunction::IndT IndT;
        typedef typename BaseImageShapeFunction::DistT DistT;
        typedef typename BaseImageShapeFunction::Coord Coord;

        // resize input
        unsigned int nbRef=bisf->f_nbRef.getValue();
        ref.resize(nbRef); ref.fill(0);
        w.resize(nbRef); w.fill(0);
        if(dw) { dw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*dw)[j].fill(0); }
        if(ddw) { ddw->resize(nbRef); for (unsigned int j=0; j<nbRef; j++ ) (*ddw)[j].fill(0); }

        // get transform
        typename BaseImageShapeFunction::raTransform inT(bisf->transform);

        // material to world transformation = image orientation
        helper::Quater<Real> q = helper::Quater< Real >::createQuaterFromEuler(inT->getRotation() * (Real)M_PI / (Real)180.0);
        Mat<3,3,Real> R; q.toMatrix(R);
        for ( unsigned int i = 0; i < BaseImageShapeFunction::spatial_dimensions; i++ )  for ( unsigned int j = 0; j < BaseImageShapeFunction::material_dimensions; j++ ) M[i][j]=R[i][j];

        // get precomputed indices and weights
        typename BaseImageShapeFunction::raInd indData(bisf->f_index);
        typename BaseImageShapeFunction::raDist weightData(bisf->f_w);
        if(!indData->imgList.size() || !weightData->imgList.size()) { bisf->serr<<"Weights not available"<<bisf->sendl; return; }

        const typename BaseImageShapeFunction::IndTypes::BranchingImage3D& indices = indData->imgList[0];
        const typename BaseImageShapeFunction::DistTypes::BranchingImage3D& weights = weightData->imgList[0];

        // interpolate weights in neighborhood

        typename BaseImageShapeFunction::raFineToCoarseMappingImage fineToCoarseMappingImage( bisf->f_fineToCoarseMappingImage );
        const CImg<unsigned long>& fineToCoarseMapping = fineToCoarseMappingImage->getCImg();
        typename BaseImageShapeFunction::raTransform fineToCoarseMappingTransform( bisf->f_fineToCoarseMappingTransform );


        Coord pfine = fineToCoarseMappingTransform->toImage(childPosition); // float fine image coord
        Coord Pfine;  for (unsigned int j=0; j<3; j++)  Pfine[j] = sofa::helper::round(pfine[j]); // int fine image coord

        unsigned int order=0;
        /*if(ddw) order=2; else */  // do not use order 2 for local weight interpolation. Order two is used only in weight fitting over regions
        if(dw) order=1;


        // get closest existing fine voxel
        if( Pfine[0]<=0 || Pfine[1]<=0 || Pfine[2]<=0 || Pfine[0]>=fineToCoarseMapping.width()-1 || Pfine[1]>=fineToCoarseMapping.height()-1 || Pfine[2]>=fineToCoarseMapping.depth()-1 ||
            fineToCoarseMapping(Pfine[0],Pfine[1],Pfine[2],0)==0 )
        {
            Real dmin=cimg::type<Real>::max();
            cimg_for_insideXYZ(fineToCoarseMapping,x,y,z,1) if(fineToCoarseMapping(x,y,z,0)) {Real d=(Coord(x,y,z)-pfine).norm2(); if(d<dmin) { Pfine.set(x,y,z); dmin=d; } }
            if(dmin==cimg::type<Real>::max()) return;
        }


        typename BaseImageShapeFunction::IndTypes::VoxelIndex voxelIndex; // coarse voxel index
        voxelIndex.offset = fineToCoarseMapping(Pfine[0],Pfine[1],Pfine[2],0);
        Coord P = inT->toImageInt( fineToCoarseMappingTransform->fromImage( Pfine[0],Pfine[1],Pfine[2] ) ); // int coarse image coord
        voxelIndex.index = indices->index3Dto1D( P[0], P[1], P[2] );



        // prepare neighborood
        sofa::defaulttype::Vec<27,  Coord > lpos;      // precomputed local positions
        int count=0;
        for (int k=-1; k<=1; k++) for (int j=-1; j<=1; j++) for (int i=-1; i<=1; i++) lpos[count++] = inT->fromImage(P+Coord(i,j,k)) - childPosition;

        // get indices at P
        int index=0;
        for (unsigned int r=0; r<nbRef; r++)
        {
            IndT ind = indices[voxelIndex.index][voxelIndex.offset][r];
            if(ind>0)
            {
                vector<DistT> val; val.reserve(27);
                vector<Coord> pos; pos.reserve(27);
                // add neighbors with same index
                count=0;
                for (int k=-1; k<=1; k++) for (int j=-1; j<=1; j++) for (int i=-1; i<=1; i++)
                {
                    unsigned neighbourIndex = indices->index3Dto1D( P[0]+i, P[1]+j, P[2]+k );
                    for (unsigned v=0;v<indices[neighbourIndex].size();v++) // superimposedvoxels
                    for (unsigned int r2=0;r2<nbRef;r2++)
                        if(indices[neighbourIndex][v][r2]==ind)
                        {
                            val.push_back(weights[neighbourIndex][v][r2]);
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
        bisf->normalize(w,dw,ddw);
    }
};





/**
abstract class for shape functions computed from a set of images (typically rasterized objects)
  */
template <class ShapeFunctionTypes_,class ImageTypes_>
class BaseImageShapeFunction : public BaseShapeFunction<ShapeFunctionTypes_>
{
    friend struct BaseImageShapeFunctionSpecialization<defaulttype::IMAGELABEL_IMAGE>;
    friend struct BaseImageShapeFunctionSpecialization<defaulttype::IMAGELABEL_BRANCHINGIMAGE>;

public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(BaseImageShapeFunction, ShapeFunctionTypes_,ImageTypes_) , SOFA_TEMPLATE(BaseShapeFunction, ShapeFunctionTypes_));
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
    Data< DistTypes > f_w;

    typedef unsigned int IndT;
    typedef defaulttype::Image<IndT> IndTypes;
    typedef helper::ReadAccessor<Data< IndTypes > > raInd;
    typedef helper::WriteAccessor<Data< IndTypes > > waInd;
    Data< IndTypes > f_index;

    // only used for branching image
    typedef defaulttype::Image<unsigned long> FineToCoarseMappingImage;
    typedef helper::ReadAccessor<Data< FineToCoarseMappingImage > > raFineToCoarseMappingImage;
    typedef helper::WriteAccessor<Data< FineToCoarseMappingImage > > waFineToCoarseMappingImage;
    Data< FineToCoarseMappingImage > f_fineToCoarseMapping;
    Data< TransformType > f_fineToCoarseMappingTransform;
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const BaseImageShapeFunction<ShapeFunctionTypes_,ImageTypes_>* = NULL) { return ShapeFunctionTypes_::Name()+std::string(",")+ImageTypes_::Name(); }


    /// interpolate weights and their derivatives at a spatial position
    void computeShapeFunction(const Coord& childPosition, MaterialToSpatial& M, VRef& ref, VReal& w, VGradient* dw=NULL,VHessian* ddw=NULL)
    {
        BaseImageShapeFunctionSpecialization<ImageTypes::label>::computeShapeFunction( this, childPosition, M, ref, w, dw, ddw );
    }

/*
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
        //if(this->f_printLog.getValue()) std::cout<<"BaseImageShapeFunction: weight fitting error on sample "<<index<<" = "<<totalerr<<std::endl;

        // normalize
        this->normalize(w,dw,ddw);
    }
*/

    virtual void init()
    {
        Inherit::init();
    }

protected:
    BaseImageShapeFunction()
        :Inherit()
        , image(initData(&image,ImageTypes(),"image",""))
        , transform(initData(&transform,TransformType(),"transform",""))
        , f_w(initData(&f_w,DistTypes(),"weights",""))
        , f_index(initData(&f_index,IndTypes(),"indices",""))
        , f_fineToCoarseMapping(initData(&f_fineToCoarseMapping,FineToCoarseMappingImage(),"fineToCoarseMapping",""))
        , f_fineToCoarseMappingTransform(initData(&f_fineToCoarseMappingTransform,TransformType(),"fineToCoarseMappingTransform",""))
    {
        image.setReadOnly(true);
        transform.setReadOnly(true);

        image.setGroup("input");
        transform.setGroup("input");
        f_w.setGroup("output");
        f_index.setGroup("output");

        BaseImageShapeFunctionSpecialization<ImageTypes::label>::constructor( this );
    }

    virtual ~BaseImageShapeFunction()
    {

    }


};


}
}
}


#endif
