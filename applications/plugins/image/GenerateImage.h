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
#ifndef SOFA_IMAGE_GenerateImage_H
#define SOFA_IMAGE_GenerateImage_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>


namespace sofa
{

namespace component
{

namespace engine
{


/**
 * Create an image with custom dimensions
 */


template <class _ImageTypes>
class GenerateImage : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(GenerateImage,_ImageTypes),Inherited);

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;
    typedef typename ImageTypes::imCoord imCoord;

    Data< imCoord > dimxyzct;
    Data< ImageTypes > image;

    virtual std::string getTemplateName() const    override { return templateName(this);    }
    static std::string templateName(const GenerateImage<ImageTypes>* = NULL) { return ImageTypes::Name(); }

    GenerateImage()    :   Inherited()
      , dimxyzct(initData(&dimxyzct,"dim","dimensions (x,y,z,c,t)",""))
      , image(initData(&image,ImageTypes(),"image",""))
    {
        this->addAlias(&dimxyzct, "dimensions");
    }

    virtual ~GenerateImage() {}

    virtual void init() override
    {
        addInput(&dimxyzct);
        addOutput(&image);
        setDirtyValue();
    }

    virtual void reinit() override { update(); }

protected:

    virtual void update() override
    {
        const imCoord& dim = this->dimxyzct.getValue();
        helper::WriteOnlyAccessor<Data< ImageTypes > > out(this->image);
        cleanDirty();
        out->setDimensions(dim);
    }

};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_GenerateImage_H
