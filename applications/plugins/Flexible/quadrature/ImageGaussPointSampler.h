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
            cimg_forC(indices,v) l.insert(indices(x,y,z,v));
            indMap::iterator it=List.find(l);
            unsigned int sampleindex;
            if(it==List.end()) { sampleindex=List.size(); List[l]=sampleindex;  regions.push_back(regionData(l));}
            else { sampleindex=it->second; regions[sampleindex].nb++;}
            outimg(x,y,z)=(DistT)sampleindex;
        }
//        unsigned int maxsampleNumber=0;
//        for(unsigned int i=0;i<regions.size();i++) if(maxsampleNumber<regions[i].nb) maxsampleNumber=regions[i].nb;

        if(this->f_method.getValue().getSelectedId() == MIDPOINT)
        {
            // add point in the center of each region
            pos.resize ( (List.size()>targetNumber.getValue())?List.size():targetNumber.getValue() );
            vol.resize ( pos.size() );

            for(unsigned int i=initialPosSize; i<List.size(); i++)
            {
                pos[i]=Coord(0,0,0);
                vector<Coord> pi((int)regions[i].nb);
                vector<vector<DistT> > wi((int)nbref,vector<DistT>((int)regions[i].nb));
                unsigned int count=0;
                cimg_forXYZ(outimg,x,y,z)
                if(outimg(x,y,z)==i)
                {
                    for(int j=0; j<nbref; j++) wi[j][count]=weights(x,y,z,j);
                    pi[count]=transform->fromImage(Coord(x,y,z));
                    pos[i]+=pi[count];
                    count++;
                }
                pos[i]/=(Real)regions[i].nb;

                for(unsigned int j=0; j<regions[i].nb; j++)
                {
                    pi[j]-=pos[i]; pi[j]*=(Real)(-1);
                }
            }

        }
        else if(this->f_method.getValue().getSelectedId() == SIMPSON)
        {
            serr<<"SIMPSON quadrature not yet implemented"<<sendl;
        }
        else if(this->f_method.getValue().getSelectedId() == GAUSSLEGENDRE)
        {
            serr<<"GAUSSLEGENDRE quadrature not yet implemented"<<sendl;
        }

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
