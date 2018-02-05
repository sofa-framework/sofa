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
#ifndef SOFA_IMAGE_ImageCoordValuesFromPositions_H
#define SOFA_IMAGE_ImageCoordValuesFromPositions_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>
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

/**
 * Get interpolated coordinates at sample locations in an image with 3 channels
 */

/// Default implementation does not compile
template <class ImageType>
struct ImageCoordValuesFromPositionsSpecialization
{
};

/// forward declaration
template <class ImageType> class ImageCoordValuesFromPositions;


/// Specialization for regular Image
template <class T>
struct ImageCoordValuesFromPositionsSpecialization<defaulttype::Image<T>>
{
    typedef ImageCoordValuesFromPositions<defaulttype::Image<T>> ImageCoordValuesFromPositionsT;

    static void update(ImageCoordValuesFromPositionsT& This)
    {
        typedef typename ImageCoordValuesFromPositionsT::Real Real;
        typedef typename ImageCoordValuesFromPositionsT::Coord Coord;

        typename ImageCoordValuesFromPositionsT::raTransform inT(This.transform);
        typename ImageCoordValuesFromPositionsT::raPositions pos(This.position);

        typename ImageCoordValuesFromPositionsT::raImage in(This.image);
        if(in->isEmpty()) return;
        const cimg_library::CImg<T>& img = in->getCImg(This.time);

        typename ImageCoordValuesFromPositionsT::waValues val(This.values);
        Coord outval (This.outValue.getValue(),This.outValue.getValue(),This.outValue.getValue());
        val.resize(pos.size());

        if(This.addPosition.getValue()) for(unsigned int i=0; i<pos.size(); i++) val[i]=pos[i];
        else for(unsigned int i=0; i<pos.size(); i++) val[i]=Coord();

        if(img.spectrum()!=3)
        {
            This.serr<<"input image must have 3 channels"<<This.sendl;
            return;
        }

        switch(This.Interpolation.getValue().getSelectedId())
        {
        case INTERPOLATION_CUBIC :
        {
            for(unsigned int i=0; i<pos.size(); i++)
            {
                Coord Tp = inT->toImage(pos[i]);
                if(!in->isInside(Tp[0],Tp[1],Tp[2]))  val[i] = outval;
                else val[i] += Coord(
                            (Real)img.cubic_atXYZ(Tp[0],Tp[1],Tp[2],0,(T)outval[0],cimg_library::cimg::type<T>::min(),cimg_library::cimg::type<T>::max()),
                        (Real)img.cubic_atXYZ(Tp[0],Tp[1],Tp[2],1,(T)outval[0],cimg_library::cimg::type<T>::min(),cimg_library::cimg::type<T>::max()),
                        (Real)img.cubic_atXYZ(Tp[0],Tp[1],Tp[2],2,(T)outval[0],cimg_library::cimg::type<T>::min(),cimg_library::cimg::type<T>::max())
                        );
            }
        }
            break;

        case INTERPOLATION_LINEAR :
        {
            for(unsigned int i=0; i<pos.size(); i++)
            {
                Coord Tp = inT->toImage(pos[i]);
                if(!in->isInside(Tp[0],Tp[1],Tp[2])) val[i] = outval;
                else val[i] += Coord(
                            (Real)img.linear_atXYZ(Tp[0],Tp[1],Tp[2],0,(T)outval[0]),
                        (Real)img.linear_atXYZ(Tp[0],Tp[1],Tp[2],1,(T)outval[0]),
                        (Real)img.linear_atXYZ(Tp[0],Tp[1],Tp[2],2,(T)outval[0])
                        );
            }
        }
            break;

        default : // NEAREST
        {
            for(unsigned int i=0; i<pos.size(); i++)
            {
                Coord Tp = inT->toImageInt(pos[i]);
                if(!in->isInside((int)Tp[0],(int)Tp[1],(int)Tp[2]))  val[i] = outval;
                else val[i] += Coord(
                            (Real)img.atXYZ(Tp[0],Tp[1],Tp[2],0),
                        (Real)img.atXYZ(Tp[0],Tp[1],Tp[2],1),
                        (Real)img.atXYZ(Tp[0],Tp[1],Tp[2],2)
                        );

            }
        }
            break;
        }

    }

};




template <class _ImageTypes>
class ImageCoordValuesFromPositions : public core::DataEngine
{
    friend struct ImageCoordValuesFromPositionsSpecialization<_ImageTypes>;

public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageCoordValuesFromPositions,_ImageTypes),Inherited);

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

    typedef helper::vector<Coord > SeqPositions;
    typedef helper::ReadAccessor<Data< SeqPositions > > raPositions;
    Data< SeqPositions > position;

    Data< helper::OptionsGroup > Interpolation;  ///< nearest, linear, cubic

    typedef helper::WriteOnlyAccessor<Data< SeqPositions > > waValues;
    Data< SeqPositions > values;  ///< output interpolated values
    Data< Real > outValue;

    Data< bool > addPosition;

    virtual std::string getTemplateName() const    override { return templateName(this);    }
    static std::string templateName(const ImageCoordValuesFromPositions<ImageTypes>* = NULL) { return ImageTypes::Name();    }

    ImageCoordValuesFromPositions()    :   Inherited()
      , image(initData(&image,ImageTypes(),"image",""))
      , transform(initData(&transform,TransformType(),"transform",""))
      , position(initData(&position,SeqPositions(),"position","input positions"))
      , Interpolation( initData ( &Interpolation,"interpolation","Interpolation method." ) )
      , values( initData ( &values,"values","Interpolated values." ) )
      , outValue(initData(&outValue,(Real)0,"outValue","default value outside image"))
      , addPosition(initData(&addPosition,true,"addPosition","add positions to interpolated values (to get translated positions)"))
      , time((unsigned int)0)
    {
        helper::OptionsGroup InterpolationOptions(3,"Nearest", "Linear", "Cubic");
        InterpolationOptions.setSelectedItem(INTERPOLATION_LINEAR);
        Interpolation.setValue(InterpolationOptions);

        image.setReadOnly(true);
        transform.setReadOnly(true);
        f_listening.setValue(true);
    }

    virtual void init() override
    {
        addInput(&image);
        addInput(&transform);
        addInput(&position);
        addInput(&addPosition);
        addOutput(&values);
        setDirtyValue();
    }

    virtual void reinit() override { update(); }

protected:

    unsigned int time;

    virtual void update() override
    {
        ImageCoordValuesFromPositionsSpecialization<ImageTypes>::update( *this );
        cleanDirty();
    }

    void handleEvent(sofa::core::objectmodel::Event *event) override
    {
        if (simulation::AnimateEndEvent::checkEventType(event))
        {
            raImage in(this->image);
            raTransform inT(this->transform);

            // get current time modulo dimt
            const unsigned int dimt=in->getDimensions()[ImageTypes::DIMENSION_T];
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

#endif // SOFA_IMAGE_ImageCoordValuesFromPositions_H
