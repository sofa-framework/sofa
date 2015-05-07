/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ENGINE_TestImageEngine_H
#define SOFA_COMPONENT_ENGINE_TestImageEngine_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/component/component.h>
#include "ImageTypes.h"
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateEndEvent.h>


namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class is only used to test engine with image data.
 * Given a input image the ouput image will have the same dimension as input image and all pixels will be 0.
 */
template <class _ImageTypes>
class TestImageEngine : public core::DataEngine
{

public:

    SOFA_CLASS(SOFA_TEMPLATE(TestImageEngine,_ImageTypes),core::DataEngine);

    typedef core::DataEngine Inherited;
    typedef _ImageTypes ImageTypes;
    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef typename TransformType::Coord Coord;
    typedef typename ImageTypes::imCoord imCoord;
    typedef typename ImageTypes::T T;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;

    Data< ImageTypes > inputImage;  ///< input image
    Data< ImageTypes > outputImage; ///< ouput image

    TestImageEngine() :   Inherited()
        , inputImage(initData(&inputImage,ImageTypes(),"inputImage","input image"))
        , outputImage(initData(&outputImage,ImageTypes(),"outputImage","ouput image"))
    {
        inputImage.setReadOnly(true);
    }

    ~TestImageEngine() {}

    void init()
    {
        addInput(&inputImage);
        addOutput(&outputImage);
        setDirtyValue();
    }

    void reinit()
    {
        update();
    }

    void update()
    {
        std::cout << "Call update method of TestImageEngine.h" << std::endl;
        cleanDirty();

        waImage out(this->outputImage);
        raImage in(this->inputImage);

        // Get the dimensions of input image
        const cimg_library::CImg<T>& img = in->getCImg(this->getContext()->getTime());
        imCoord dim = in->getDimensions();

        // Set the dimensions of outputImage
        out->setDimensions(dim);

        //  Fill all pixel values of ouputImage with 0
        cimg_library::CImg<T>& outImg=out->getCImg(0);
        outImg.fill(0);

    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if ( dynamic_cast<simulation::AnimateEndEvent*>(event))
        {
            update(); // update at each time step if listening=true
        }
    }


};

} // namespace engine

} // namespace component

} // namespace sofa

#endif
