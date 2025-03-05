/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#ifndef LABELIMAGETOOLBOX_H
#define LABELIMAGETOOLBOX_H 

#include <sofa/core/visual/VisualParams.h>
#include <QObject>
#include <image_gui/config.h>
#include <image/ImageTypes.h>
#include "sofa/defaulttype/config.h"
#include "sofa/defaulttype/VecTypes.h"
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>


#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>



//#include "labelimagetoolboxaction.h"



namespace sofa
{
namespace gui
{
namespace qt
{
class SOFA_IMAGE_GUI_API LabelImageToolBoxAction;
}
}
}

namespace sofa
{

namespace component
{

namespace engine
{

using type::vector;
using type::Vec;
using type::Mat;
using namespace cimg_library;

/**
 * This class corresponds to a label visualized by imagetoolbox
 */

class SOFA_IMAGE_GUI_API LabelImageToolBox : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(LabelImageToolBox,Inherited);
    
    
    
    
    Data< bool > d_islinkedToToolBox; ///< true if a toobbox use this Label
    Data< sofa::type::Vec4d > d_color;

//    virtual std::string getTemplateName() const    { return templateName(this);    }
//    static std::string templateName(const LabelImageToolBox* = NULL) { return ImageTypes::Name();    }

    LabelImageToolBox();

    void init() override
    {
        //addInput(&image);
        //addOutput(&triangles);
        setDirtyValue();
    }

    void reinit() override { update(); }

protected:

    unsigned int time;

    void doUpdate() override
    {
    }

    void handleEvent(sofa::core::objectmodel::Event * /*event*/) override
    {
    }

    void draw(const core::visual::VisualParams* /*vparams*/) override
    {
    }

public:
    
    virtual sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QWidget* /*parent*/=nullptr )=0;

};


} // namespace engine

} // namespace component

} // namespace sofa


#endif // LABELIMAGETOOLBOX_H
