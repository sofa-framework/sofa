/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_ENGINE_TestImageEngine_H
#define SOFA_COMPONENT_ENGINE_TestImageEngine_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <image/ImageTypes.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>


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
    typedef helper::WriteOnlyAccessor<Data< ImageTypes > > waImage;
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
        waImage out(this->outputImage);
        raImage in(this->inputImage);

        // Get the dimensions of input image
        imCoord dim = in->getDimensions();

        // Set the dimensions of outputImage
        out->setDimensions(dim);

        //  Copy input on output
//        cimg_library::CImg<T>& outImg = out->getCImg(0);

        out->getCImg(0) = in->getCImg(0);
//        std::cerr << "TestImageEngine input shared: " << in->getCImg(0).is_shared() << std::endl;
        cleanDirty();
    }

    void handleEvent(sofa::core::objectmodel::Event *event)
    {
        if (simulation::AnimateEndEvent::checkEventType(event))
        {
            update(); // update at each time step if listening=true
        }
    }


};

} // namespace engine

} // namespace component

} // namespace sofa

#endif
