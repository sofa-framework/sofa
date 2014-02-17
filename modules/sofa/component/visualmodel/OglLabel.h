/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_VISUALMODEL_LABEL_H
#define SOFA_COMPONENT_VISUALMODEL_LABEL_H

#include <sofa/core/objectmodel/Data.h>
#include <sofa/component/component.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/Vec.h>
#include <string>

namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API OglLabel : public virtual sofa::core::visual::VisualModel
{
public:
    SOFA_CLASS(OglLabel, sofa::core::visual::VisualModel);

protected:
    OglLabel();
    virtual ~OglLabel() {
    }

public:


    Data<std::string> label;
    Data<unsigned int> x;
    Data<unsigned int> y;
    Data<unsigned int> fontsize;
    Data<std::string> color;
	Data<bool> f_visible;


    void reinit();
    void drawVisual(const core::visual::VisualParams* vparams);

private:
    void setColor(float r, float g, float b, float a);
    void setColor(std::string color);


    float r,g,b,a;
    typedef defaulttype::Vec4f Color;   // Color with alpha value

};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
