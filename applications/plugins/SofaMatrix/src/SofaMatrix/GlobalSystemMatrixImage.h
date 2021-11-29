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
#pragma once
#include <SofaMatrix/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <image/CImgData.h>
#include <image/ImageTypes.h>

namespace sofa::component::linearsolver
{

/**
 * Component to convert a BaseMatrix into an image. The image can be visualized in the GUI.
 */
class SOFA_SOFAMATRIX_API GlobalSystemMatrixImage : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(GlobalSystemMatrixImage, core::objectmodel::BaseObject);
    using ImageType = sofa::defaulttype::ImageUC;
    using ImagePlaneType = defaulttype::ImagePlane<ImageType::T>;

protected:

    GlobalSystemMatrixImage();
    ~GlobalSystemMatrixImage() override;

    void init() override;
    void handleEvent(core::objectmodel::Event *event) override;

    Data< ImageType > d_image;
    Data< ImagePlaneType > d_plane;
    SingleLink<GlobalSystemMatrixImage, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_linearSolver;
};

} //namespace sofa::component::linearsolver