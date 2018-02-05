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
#ifndef SOFA_IMAGE_IMAGEACCUMULATOR_H
#define SOFA_IMAGE_IMAGEACCUMULATOR_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/helper/system/thread/CTime.h>


namespace sofa
{

namespace component
{

namespace engine
{


/**
 * This class wraps images from a video stream into a single image
 */


template <class _ImageTypes>
class ImageAccumulator : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageAccumulator,_ImageTypes),Inherited);

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;
    typedef helper::WriteOnlyAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;

    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef helper::WriteOnlyAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    typedef sofa::helper::system::thread::ctime_t ctime_t;
    typedef sofa::helper::system::thread::CTime CTime;

    Data< bool > accumulate;
    Data< ImageTypes > inputImage;
    Data< TransformType > inputTransform;
    Data< ImageTypes > outputImage;
    Data< TransformType > outputTransform;

    virtual std::string getTemplateName() const    override { return templateName(this);    }
    static std::string templateName(const ImageAccumulator<ImageTypes>* = NULL) { return ImageTypes::Name(); }

    ImageAccumulator()    :   Inherited()
        , accumulate(initData(&accumulate,false,"accumulate","accumulate ?"))
        , inputImage(initData(&inputImage,ImageTypes(),"inputImage",""))
        , inputTransform(initData(&inputTransform,TransformType(),"inputTransform",""))
        , outputImage(initData(&outputImage,ImageTypes(),"outputImage",""))
        , outputTransform(initData(&outputTransform,TransformType(),"outputTransform",""))
        , SimuTime(0.0)
        , count(0)
    {
        inputImage.setReadOnly(true);
        inputTransform.setReadOnly(true);
        outputImage.setReadOnly(true);
        outputTransform.setReadOnly(true);
        f_listening.setValue(true);
    }

    virtual ~ImageAccumulator() {}

    virtual void init() override
    {
        addInput(&inputImage);
        addInput(&inputTransform);
        addOutput(&outputImage);
        addOutput(&outputTransform);
        setDirtyValue();
    }

    virtual void reinit() override { update(); }

protected:
    double SimuTime;
    ctime_t t0,t;
    int count;

    virtual void update() override
    {
        if(SimuTime==this->getContext()->getTime()) return; // check if simutime has changed
        SimuTime=this->getContext()->getTime();

        if(!accumulate.getValue()) return;

        raImage in(this->inputImage);
        if(in->isEmpty()) return;
        raTransform inT(this->inputTransform);

        cleanDirty();

        waImage out(this->outputImage);
        waTransform outT(this->outputTransform);

        if(out->isEmpty()) {t0=CTime::getTime(); outT->operator=(inT);}
        else { count++; t=CTime::getTime(); outT->getScaleT()=0.000001*(t-t0)/(Real)count; } // update time scale to fit acquisition rate

        out->getCImgList().push_back(in->getCImg(0));
    }

    void handleEvent(sofa::core::objectmodel::Event *event) override
    {
        if ( /*simulation::AnimateEndEvent* ev =*/ simulation::AnimateEndEvent::checkEventType(event)) update();
    }
};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_IMAGEACCUMULATOR_H
