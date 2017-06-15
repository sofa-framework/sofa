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
#ifndef SOFA_IMAGE_ImageTransformEngine_H
#define SOFA_IMAGE_ImageTransformEngine_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/Quater.h>


namespace sofa
{

namespace component
{

namespace engine
{

/**
 * Apply a transform to the data 'transform'
 * in future: could be templated on ImageTransform type
 * @author Benjamin GILLES
 */


class SOFA_IMAGE_API ImageTransformEngine : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(ImageTransformEngine,Inherited);

    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef TransformType::Coord Coord;
    typedef helper::WriteOnlyAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    Data< TransformType > inputTransform;
    Data< TransformType > outputTransform;

    Data<defaulttype::Vector3> translation; // translation
    Data<defaulttype::Vector3> rotation; // rotation
    Data<Real> scale; // scale
    Data<bool> inverse;

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ImageTransformEngine* = NULL) { return std::string();  }

    ImageTransformEngine()    :   Inherited()
      , inputTransform(initData(&inputTransform,TransformType(),"inputTransform",""))
      , outputTransform(initData(&outputTransform,TransformType(),"outputTransform",""))
      , translation(initData(&translation, defaulttype::Vector3(0,0,0),"translation", "translation vector ") )
      , rotation(initData(&rotation, defaulttype::Vector3(0,0,0), "rotation", "rotation vector ") )
      , scale(initData(&scale, (Real)1.0,"scale", "scale factor") )
      , inverse(initData(&inverse, false, "inverse", "true to apply inverse transformation"))
    {
    }

    virtual ~ImageTransformEngine() {}

    virtual void init()
    {
        addInput(&translation);
        addInput(&rotation);
        addInput(&scale);
        addInput(&inverse);
        addInput(&inputTransform);
        addOutput(&outputTransform);
        setDirtyValue();
    }

    virtual void reinit() { update(); }

protected:

    virtual void update()
    {
		raTransform inT(this->inputTransform);
        waTransform outT(this->outputTransform);

        Real s;
        helper::Quater<Real> r;
        Coord t;

        if(this->inverse.getValue())
        {
            s=(Real)1./this->scale.getValue();
            r=helper::Quater< Real >::createQuaterFromEuler(this->rotation.getValue()* (Real)M_PI / (Real)180.0 ).inverse();
            t=-r.rotate(this->translation.getValue())*s;
        }
        else
        {
            s=this->scale.getValue();
            t=this->translation.getValue();
            r= helper::Quater< Real >::createQuaterFromEuler(this->rotation.getValue()* (Real)M_PI / (Real)180.0 );
        }

        outT->getScale() = inT->getScale() * s;
        outT->getTranslation() = r.rotate(inT->getTranslation())*s + t;

        helper::Quater<Real> q = r*inT->qrotation;
        outT->getRotation()=q.toEulerVector() * (Real)180.0 / (Real)M_PI ;

        outT->update(); // update internal data

        cleanDirty();
    }

};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_IMAGE_ImageTransformEngine_H
