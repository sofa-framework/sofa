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
#ifndef SOFA_IMAGE_MERGEDCATCHALLVECTOR_H
#define SOFA_IMAGE_MERGEDCATCHALLVECTOR_H

#include <image/image_gui/config.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>

#include <sofa/helper/OptionsGroup.h>

#include <image/ImageContainer.h>

namespace sofa
{
namespace component
{
namespace engine
{

template <class _Type>
class SOFA_IMAGE_GUI_API MergedCatchAllVector : public core::DataEngine
{
public:
    typedef core::DataEngine Inherited;
    SOFA_CLASS(SOFA_TEMPLATE(MergedCatchAllVector,_Type),Inherited);

    typedef _Type Type;

    MergedCatchAllVector()    :   Inherited()
        ,_data(initData(&_data,"data_out","data_out"))
        ,_data1(initData(&_data1,"data_in1","data_in1"))
        ,_data2(initData(&_data2,"data_in2","data_in2"))

    {

    }

    ~MergedCatchAllVector() {}

    void init()
    {
        reinit();
    }

    void reinit()
    {
        helper::vector<Type> data;
        const helper::vector<Type> &data1 = _data1.getValue();
        const helper::vector<Type> &data2 = _data2.getValue();


        for(size_t i=0;i<data1.size();i++)
        {
            data.push_back(data1[i]);
        }

        for(size_t i=0;i<data2.size();i++)
        {
            data.push_back(data2[i]);
        }

        _data.setValue(data);

    }

    Data< helper::vector<Type> > _data;
    Data< helper::vector<Type> > _data1;
    Data< helper::vector<Type> > _data2;

protected:

    virtual void update()
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

#endif // SOFA_IMAGE_MergedCatchAllVector_H
