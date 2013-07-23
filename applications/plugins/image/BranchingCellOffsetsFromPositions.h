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
#ifndef SOFA_IMAGE_BranchingCellOffsetsFromPositions_H
#define SOFA_IMAGE_BranchingCellOffsetsFromPositions_H

#include "initImage.h"
#include "ImageTypes.h"
#include "BranchingImage.h"
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{
namespace component
{
namespace engine
{

using helper::vector;
using defaulttype::Vec;
using defaulttype::Mat;
using namespace helper;
using namespace cimg_library;

/**
 * Returns offsets of superimposed voxels at positions corresponding to certains labels (image intensity values)
 */


template <class _BranchingImageTypes>
class BranchingCellOffsetsFromPositions : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(BranchingCellOffsetsFromPositions,_BranchingImageTypes),Inherited);

    typedef SReal Real;

    typedef _BranchingImageTypes BranchingImageTypes;
    typedef typename BranchingImageTypes::T T;
    typedef helper::ReadAccessor<Data< BranchingImageTypes > > raBranchingImage;

    typedef vector<T> labelType;
    typedef helper::ReadAccessor<Data< labelType > > raLabels;
    Data< labelType > labels;

    Data< BranchingImageTypes > branchingImage;

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > branchingImageTransform;

    typedef vector<Vec<3,Real> > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    Data< SeqPositions > position;

    typedef vector<unsigned int> valuesType;
    typedef helper::WriteAccessor<Data< valuesType > > waValues;
    Data< valuesType > cell;  ///< output values

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const BranchingCellOffsetsFromPositions<BranchingImageTypes>* = NULL) { return BranchingImageTypes::Name();    }

    BranchingCellOffsetsFromPositions()    :   Inherited()
      , labels(initData(&labels,"labels","labels in input image used to select superimposed voxels"))
      , branchingImage(initData(&branchingImage,BranchingImageTypes(),"branchingImage",""))
      , branchingImageTransform(initData(&branchingImageTransform,TransformType(),"branchingImageTransform",""))
      , position(initData(&position,SeqPositions(),"position","input positions"))
      , cell( initData ( &cell,"cell","cell offsets" ) )
      , time((unsigned int)0)
    {
        this->addAlias(&branchingImageTransform, "branchingTransform");
        this->addAlias(&branchingImageTransform, "transform");
        this->addAlias(&branchingImage, "image");
        branchingImage.setReadOnly(true);
        branchingImageTransform.setReadOnly(true);
        f_listening.setValue(true);
    }

    virtual void init()
    {
        addInput(&labels);
        addInput(&branchingImage);
        addInput(&branchingImageTransform);
        addInput(&position);
        addOutput(&cell);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    unsigned int time;

    virtual void update()
    {
        cleanDirty();

        raBranchingImage in(this->branchingImage);
        raTransform inT(this->branchingImageTransform);
        raPositions pos(this->position);

        // get images at time t
        const typename BranchingImageTypes::BranchingImage3D& img = in->imgList[this->time];

        raLabels lab(this->labels);
        waValues val(this->cell);
        unsigned int outval=0;
        val.resize(pos.size());

        for(unsigned int i=0; i<pos.size(); i++)
        {
            Coord Tp = inT->toImageInt(pos[i]);
            if(!in->isInside((int)Tp[0],(int)Tp[1],(int)Tp[2]))  val[i] = outval;
            else
            {
                typename BranchingImageTypes::VoxelIndex vi (in->index3Dto1D(Tp[0],Tp[1],Tp[2]), 0);
                bool found=false;
                for(vi.offset = 0 ; vi.offset<img[vi.index1d].size() ; vi.offset++)
                {
                    T v= in->imgList[this->time][vi.index1d][vi.offset][0]; // assume that labels are stored in first channel
                    for(unsigned int l = 0 ; l<lab.size() ; l++) if(v==lab[l]) found=true;
                    if(found) break;
                }
                if(found) val[i]=vi.offset+1; else val[i]=outval;
            }
        }
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if ( dynamic_cast<simulation::AnimateEndEvent*>(event))
        {
            raBranchingImage in(this->branchingImage);
            raTransform inT(this->branchingImageTransform);

            // get current time modulo dimt
            const unsigned int dimt=in->getDimensions()[4];
            if(!dimt) return;
            Real t=inT->toImage(this->getContext()->getTime()) ;
            t-=(Real)((int)((int)t/dimt)*dimt);
            t=(t-floor(t)>0.5)?ceil(t):floor(t); // nearest
            if(t<0) t=0.0; else if(t>=(Real)dimt) t=(Real)dimt-1.0; // clamp

            if(this->time!=(unsigned int)t) { this->time=(unsigned int)t; update(); }
        }
    }

};


} // namespace engine
} // namespace component
} // namespace sofa

#endif // SOFA_IMAGE_BranchingCellOffsetsFromPositions_H
