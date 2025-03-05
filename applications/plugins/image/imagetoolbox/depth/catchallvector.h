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
#ifndef SOFA_IMAGE_CATCHALLVECTOR_H
#define SOFA_IMAGE_CATCHALLVECTOR_H

#include <image_gui/config.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/type/Vec.h>

#include <sofa/helper/OptionsGroup.h>

#include <image/ImageContainer.h>

namespace sofa
{
namespace component
{
namespace engine
{

template <class _Type>
class SOFA_IMAGE_GUI_API CatchAllVector : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(CatchAllVector,_Type),Inherited);

    typedef _Type Type;

    CatchAllVector()    :   Inherited()
        ,_data(initData(&_data,"data","data"))
    {

    }

    ~CatchAllVector() override {}

    void init() override
    {
        reinit();
    }

    void reinit() override
    {
    }

    Data< type::vector<Type> > _data; ///< data

protected:

    void doUpdate() override
    {
    }

public:

   /* virtual void draw(const core::visual::VisualParams*)
    {
    }

    void handleEvent(core::objectmodel::Event *event)
    {
    }*/
};





} // namespace engine
} // namespace component
} // namespace sofa

#endif // SOFA_IMAGE_CatchAllVector_H
