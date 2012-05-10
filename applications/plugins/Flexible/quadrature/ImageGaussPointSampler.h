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
    //    Data< IndTypes > f_voronoi;
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

    Data< DistTypes > outputImage;
    //@}

    /** @name  Options */
    //@{
    Data<unsigned int> targetNumber;
    //@}

    virtual std::string getTemplateName() const    { return templateName(this); }
    static std::string templateName(const ImageGaussPointSampler<ImageTypes_>* = NULL) { return ImageTypes_::Name(); }


    virtual void init()
    {
        Inherit::init();

        addInput(&f_index);
        addInput(&f_w);
        addInput(&f_transform);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:
    ImageGaussPointSampler()    :   Inherit()
        , f_index(initData(&f_index,IndTypes(),"indices",""))
        , f_w(initData(&f_w,DistTypes(),"weights",""))
        , f_transform(initData(&f_transform,TransformType(),"transform",""))
        , outputImage(initData(&outputImage,DistTypes(),"outputImage",""))
        , targetNumber(initData(&targetNumber,(unsigned int)0,"targetNumber","target number of samples"))
    {
    }

    virtual ~ImageGaussPointSampler()
    {

    }

    typedef std::set<unsigned int> indList;
    typedef std::map<indList, unsigned int> indMap;

    struct regionData
    {
        unsigned int nb;
        Coord c;
        Real err;
        indList indices;
        vector<vector<Real> > coeff;
        regionData(indList ind):nb(1),c(Coord()),err(0),indices(ind) {}
    };

    virtual void update()
    {
        cleanDirty();

        // get tranform and images at time t
        raDist rweights(this->f_w);             if(!rweights->getCImgList().size())  { serr<<"Weights not found"<<sendl; return; }
        raInd rindices(this->f_index);          if(!rindices->getCImgList().size())  { serr<<"Indices not found"<<sendl; return; }
        raTransform transform(this->f_transform);
        const CImg<DistT>& weights = rweights->getCImg(0);  // suppose time=0
        const CImg<IndT>& indices = rindices->getCImg(0);  // suppose time=0
        imCoord dim = rweights->getDimensions();
        const int nbref=dim[3]; dim[3]=dim[4]=1;
        const Real dv =  transform->getScale()[0] * transform->getScale()[1] * transform->getScale()[2];

        // get output data
        waPositions pos(this->f_position);
        waVolume vol(this->f_volume);

        waDist wout(this->outputImage);
        wout->setDimensions(dim);
        CImg<DistT>& outimg = wout->getCImg(0);
        outimg.fill((DistT)(-1));

        // identify regions with similar repartitions
        indMap List;
        vector<regionData> regions;

        // add initial points from user
        const unsigned int initialPosSize=pos.size();

        for(unsigned int i=0; i<initialPosSize; i++)
        {
            Coord p = transform->toImage(pos[i]);
            for (unsigned int j=0; j<3; j++)  p[j]=round(p[j]);
            if(indices(p[0],p[1],p[2]))
            {
                indList l;
                cimg_forC(indices,v) l.insert(indices(p[0],p[1],p[2],v));
                List[l]=i;
                regions.push_back(regionData(l));
                outimg(p[0],p[1],p[2])=(DistT)i;
            }
        }

        // traverse index image to identify other regions
        cimg_forXYZ(indices,x,y,z)
        if(indices(x,y,z))
        {
            indList l;
            cimg_forC(indices,v) if(indices(x,y,z,v)) l.insert(indices(x,y,z,v));
            indMap::iterator it=List.find(l);
            unsigned int sampleindex;
            if(it==List.end()) { sampleindex=List.size(); List[l]=sampleindex;  regions.push_back(regionData(l));}
            else { sampleindex=it->second; regions[sampleindex].nb++;}
            outimg(x,y,z)=(DistT)sampleindex;
        }


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
            // add point in the center of each region, and compute region data (volumes and weights)
            pos.resize ( (List.size()>targetNumber.getValue())?List.size():targetNumber.getValue() );
            vol.resize ( pos.size() );

            int dimBasis=(this->f_order.getValue()+1)*(this->f_order.getValue()+2)*(this->f_order.getValue()+3)/6; // dimension of polynomial basis in 3d

            for(unsigned int i=initialPosSize; i<List.size(); i++)
            {
                int nb=regions[i].nb;
                pos[i]=Coord(0,0,0);
                vol[i]=vector<Real>(dimBasis,(Real)0);

                vector<Coord> pi(nb);
                typedef std::map<unsigned int, vector<Real> > wiMap;
                wiMap wi;
                for(indList::iterator it=regions[i].indices.begin(); it!=regions[i].indices.end(); it++) wi[*it]=vector<Real>(nb,(Real)0);

                unsigned int count=0;
                cimg_forXYZ(outimg,x,y,z)
                if(outimg(x,y,z)==i)
                {
                    // store weights relative to each index of the region
                    for(int k=0; k<nbref; k++)
                    {
                        typename wiMap::iterator wIt;
                        wIt=wi.find(indices(x,y,z,k));
                        if(wIt!=wi.end()) wIt->second[count]=(Real)weights(x,y,z,k);
                    }
                    // store voxel positions
                    pi[count]=transform->fromImage(Coord(x,y,z));
                    pos[i]+=pi[count];
                    count++;
                }
                // compute region center
                pos[i]/=(Real)nb;

                // compute relative positions and volume integral
                for(int j=0; j<nb; j++)
                {
                    pi[j]-=pos[i]; pi[j]*=(Real)(-1);
                    vector<Real> basis; defaulttype::getCompleteBasis(basis,pi[j],this->f_order.getValue());
                    for(int k=0; k<dimBasis; k++) vol[i][k]+=basis[k]*dv;
                }

                // fit weights
                for(typename wiMap::iterator wIt=wi.begin(); wIt!=wi.end(); wIt++)
                {
                    vector<Real> coeff;
                    defaulttype::PolynomialFit(coeff,wIt->second,pi,2);
                    std::cout<<"weight fitting error on sample "<<i<<" ("<<wIt->first<<") = "<<defaulttype::getPolynomialFit_Error(coeff,wIt->second,pi)<< std::endl;
                    regions[i].coeff.push_back(coeff);

                }

                count=0;
                cimg_forXYZ(outimg,x,y,z)
                if(outimg(x,y,z)==i)
                {
                    outimg(x,y,z)=0;
                    unsigned int j=0;
                    for(typename wiMap::iterator wIt=wi.begin(); wIt!=wi.end(); wIt++)
                        outimg(x,y,z)+=defaulttype::getPolynomialFit_Error(regions[i].coeff[j++],wIt->second[count],pi[count]);
                    count++;
                }
            }

        }


        cimg_forXYZ(outimg,x,y,z)
        if(outimg(x,y,z)==-1)
            outimg(x,y,z)=0;

        if(this->f_printLog.getValue()) if(pos.size())    std::cout<<"ImageGaussPointSampler: "<< pos.size() <<" generated samples"<<std::endl;
    }


    virtual void draw(const core::visual::VisualParams* vparams)
    {
        Inherit::draw(vparams);
    }



};

}
}
}

#endif
