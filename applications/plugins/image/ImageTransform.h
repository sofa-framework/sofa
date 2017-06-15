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
#ifndef SOFA_IMAGE_IMAGETRANSFORM_H
#define SOFA_IMAGE_IMAGETRANSFORM_H

#include <image/config.h>
#include "ImageTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>

#include <sofa/helper/OptionsGroup.h>

#include "ImageContainer.h"

namespace sofa
{
namespace component
{
namespace engine
{

/**
 * This class gives read and write access of an image transformation.
 * The component should be in the same node as an ImageContainer component.
 * It reads/writes the 12-params transformation data of an ImageContainer and
 * converts it into/from individual data (translation, rotation, scale, ...)
 */
template <class _ImageTypes>
class ImageTransform : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(ImageTransform,_ImageTypes),Inherited);

    typedef _ImageTypes ImageTypes;
    typedef SReal Real;
    typedef sofa::defaulttype::Vec<3,Real> Vec3;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    typedef helper::WriteOnlyAccessor<Data< TransformType > > waTransform;
    typedef sofa::component::container::ImageContainer<ImageTypes> ImageContainer;

    ImageTransform()    :   Inherited()
        ,_translation(initData(&_translation, Vec3(),"translation","Translation"))
        ,_euler(initData(&_euler, Vec3(),"euler","Euler angles"))
        ,_scale(initData(&_scale, Vec3(1,1,1),"scale","Voxel size"))
        ,_isPerspective(initData(&_isPerspective, 0,"isPerspective","Is perspective?"))
        ,_timeOffset(initData(&_timeOffset, (Real)0,"timeOffset","Time offset"))
        ,_timeScale(initData(&_timeScale, (Real)1,"timeScale","Time scale"))
        ,_update(initData(&_update, "update","Type of update"))
    {
        _translation.setGroup("Transformation");
        _euler.setGroup("Transformation");
        _scale.setGroup("Transformation");
        _isPerspective.setGroup("Transformation");
        _timeOffset.setGroup("Transformation");
        _timeScale.setGroup("Transformation");

        f_listening.setValue(true);

        helper::OptionsGroup fluidOptions(3,"No update", "Every time step", "Every draw");
        _update.setValue(fluidOptions);
    }

    ~ImageTransform() {}

    void init()
    {
        container = this->getContext()->template get<ImageContainer>(core::objectmodel::BaseContext::SearchUp);
        if (!container)
            serr << "No ImageContainer found" << sendl;

        reinit();
    }

    void reinit()
    {
        update();
    }

    Data<Vec3> _translation;
    Data<Vec3> _euler;
    Data<Vec3> _scale;
    Data<int> _isPerspective;
    Data<Real> _timeOffset;
    Data<Real> _timeScale;

    enum UPDATE_TYPE{NO_UPDATE = 0, EVERY_TIMESTEP, EVERY_DRAW};
    Data<sofa::helper::OptionsGroup> _update;

protected:

    ImageContainer* container;

    virtual void update()
    {
        if (!container) return;

        waTransform wTransform(container->transform);
        if (!_translation.isSet()) _translation.setValue(wTransform->getTranslation()); else wTransform->getTranslation()=_translation.getValue();
        if (!_euler.isSet()) _euler.setValue(wTransform->getRotation()); else wTransform->getRotation()=_euler.getValue();
        if (!_scale.isSet()) _scale.setValue(wTransform->getScale()); else wTransform->getScale()=_scale.getValue();
        if (!_isPerspective.isSet()) _isPerspective.setValue(wTransform->isPerspective()); else wTransform->isPerspective()=_isPerspective.getValue();
        if (!_timeOffset.isSet()) _timeOffset.setValue(wTransform->getOffsetT()); else wTransform->getOffsetT()=_timeOffset.getValue();
        if (!_timeScale.isSet()) _timeScale.setValue(wTransform->getScaleT()); else wTransform->getScaleT()=_timeScale.getValue();
    }

public:

    virtual void draw(const core::visual::VisualParams*)
    {
        if (_update.getValue().getSelectedId()==EVERY_DRAW)
            update();
    }

    void handleEvent(core::objectmodel::Event *event)
    {
        if (sofa::simulation::AnimateBeginEvent::checkEventType(event) && _update.getValue().getSelectedId()==EVERY_TIMESTEP)
            update();
    }

    virtual std::string getTemplateName() const    { return templateName(this);    }
    static std::string templateName(const ImageTransform<ImageTypes>* = NULL) { return ImageTypes::Name(); }

};





} // namespace engine
} // namespace component
} // namespace sofa

#endif // SOFA_IMAGE_IMAGETRANSFORM_H
