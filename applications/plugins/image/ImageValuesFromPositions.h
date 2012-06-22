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
#ifndef SOFA_IMAGE_ImageValuesFromPositions_H
#define SOFA_IMAGE_ImageValuesFromPositions_H

#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/defaulttype/Vec.h>

#define INTERPOLATION_NEAREST 0
#define INTERPOLATION_LINEAR 1
#define INTERPOLATION_CUBIC 2

namespace sofa
{
namespace component
{
namespace engine
{

using helper::vector;
using defaulttype::Vec;
using defaulttype::Mat;
using namespace cimg_library;

/**
 * This class computes an isosurface from an image using marching cubes algorithm
 */


template <class _ImageTypes>
class ImageValuesFromPositions : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageValuesFromPositions,_ImageTypes),Inherited);

    typedef SReal Real;

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;

    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    typedef vector<Vec<3,Real> > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    Data< SeqPositions > position;

    Data< helper::OptionsGroup > Interpolation;  ///< nearest, linear, cubic

    typedef vector<Real> valuesType;
    typedef helper::WriteAccessor<Data< valuesType > > waValues;
    Data< valuesType > values;  ///< output interpolated values
    Data< Real > outValue;


    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ImageValuesFromPositions<ImageTypes>* = NULL) { return ImageTypes::Name();    }

    ImageValuesFromPositions()    :   Inherited()
        , image(initData(&image,ImageTypes(),"image",""))
        , transform(initData(&transform,TransformType(),"transform",""))
        , position(initData(&position,SeqPositions(),"position","input positions"))
        , Interpolation( initData ( &Interpolation,"interpolation","Interpolation method." ) )
        , values( initData ( &values,"values","Interpolated values." ) )
        , outValue(initData(&outValue,(Real)0,"outValue","default value outside image"))
        , time((unsigned int)0)
    {
        helper::OptionsGroup InterpolationOptions(3,"Nearest", "Linear", "Cubic");
        InterpolationOptions.setSelectedItem(INTERPOLATION_LINEAR);
        Interpolation.setValue(InterpolationOptions);

        image.setReadOnly(true);
        transform.setReadOnly(true);
        f_listening.setValue(true);
    }

    virtual void init()
    {
        addInput(&image);
        addInput(&transform);
        addInput(&position);
        addOutput(&values);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    unsigned int time;

    virtual void update()
    {
        cleanDirty();

        raImage in(this->image);
        raTransform inT(this->transform);
        raPositions pos(this->position);

        // get image at time t
        const CImg<T>& img = in->getCImg(this->time);

        waValues val(this->values);
        Real outval=this->outValue.getValue();
        val.resize(pos.size());

        if(Interpolation.getValue().getSelectedId()==INTERPOLATION_NEAREST)
        {
            for(unsigned int i=0; i<pos.size(); i++)
            {
                Coord Tp = inT->toImage(pos[i]);
                if(Tp[0]<0 || Tp[1]<0 || Tp[2]<0 || Tp[0]>=img.width() || Tp[1]>=img.height() || Tp[2]>=img.depth())  val[i] = outval;
                else val[i] = (Real)img.atXYZ(round((double)Tp[0]),round((double)Tp[1]),round((double)Tp[2]));
            }
        }
        else if(Interpolation.getValue().getSelectedId()==INTERPOLATION_LINEAR)
        {
            for(unsigned int i=0; i<pos.size(); i++)
            {
                Coord Tp = inT->toImage(pos[i]);
                if(Tp[0]<0 || Tp[1]<0 || Tp[2]<0 || Tp[0]>=img.width() || Tp[1]>=img.height() || Tp[2]>=img.depth()) val[i] = outval;
                else val[i] = (Real)img.linear_atXYZ(Tp[0],Tp[1],Tp[2],0,(T)outval);
            }
        }
        else
        {
            for(unsigned int i=0; i<pos.size(); i++)
            {
                Coord Tp = inT->toImage(pos[i]);
                if(Tp[0]<0 || Tp[1]<0 || Tp[2]<0 || Tp[0]>=img.width() || Tp[1]>=img.height() || Tp[2]>=img.depth())  val[i] = outval;
                else val[i] = (Real)img.cubic_atXYZ(Tp[0],Tp[1],Tp[2],0,(T)outval,cimg::type<T>::min(),cimg::type<T>::max());
            }
        }

    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if ( dynamic_cast<simulation::AnimateEndEvent*>(event))
        {
            raImage in(this->image);
            raTransform inT(this->transform);

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

#endif // SOFA_IMAGE_ImageValuesFromPositions_H
