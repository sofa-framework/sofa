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
#include <SofaMatrix/GlobalSystemMatrixImage.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/simulation/AnimateEndEvent.h>

namespace sofa::component::linearsolver
{

int GlobalSystemMatrixImageClass = core::RegisterObject("View the global linear system matrix as an binary image.")
    .add<GlobalSystemMatrixImage>();

GlobalSystemMatrixImage::GlobalSystemMatrixImage()
    : Inherit1()
    , d_image(initData(&d_image, ImageType(), "image", "The global linear system matrix concerted to an image. This Data is compatible with the component in the plugin 'image'."))
    , d_bitmap(initData(&d_bitmap, BitmapType(), "bitmap", "Visualization of the produced image."))
    , l_linearSolver(initLink("linearSolver", "Link to the linear solver containing a matrix"))
{
    d_image.setGroup("Image");
    d_image.setReadOnly(true);

    d_bitmap.setGroup("Image");
    d_bitmap.setWidget("simplebitmap");
    d_bitmap.setReadOnly(true);

    this->f_listening.setValue(true);
}

GlobalSystemMatrixImage::~GlobalSystemMatrixImage() = default;

void GlobalSystemMatrixImage::init()
{
    Inherit1::init();

    if (!l_linearSolver)
    {
        l_linearSolver.set(this->getContext()->template get<sofa::core::behavior::LinearSolver>());
    }

    if (!l_linearSolver)
    {
        msg_error() << "No linear solver found in the current context, whereas it is required. This component retrieves the matrix from a linear solver.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }

    helper::WriteOnlyAccessor<Data< BitmapType > > wplane(d_bitmap);
    wplane->setInput(d_image.getValue());

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void GlobalSystemMatrixImage::handleEvent(core::objectmodel::Event* event)
{
    BaseObject::handleEvent(event);

    if (this->d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
    {
        return;
    }

    if (simulation::AnimateEndEvent::checkEventType(event))
    {
        if (const auto* matrix = l_linearSolver->getSystemBaseMatrix())
        {
            const auto nx = matrix->colSize();
            const auto ny = matrix->rowSize();

            helper::WriteOnlyAccessor<Data< ImageType > > out(d_image);
            out->clear();

            out->setDimensions(ImageType::imCoord(nx, ny, 1, 1, 1));

            cimg_library::CImgList<ImageType::T>& img = out->getCImgList();

            for (sofa::SignedIndex y = 0; y < ny; ++y)
            {
                for (sofa::SignedIndex x = 0; x < nx; ++x)
                {
                    img(0)(x, y) = !static_cast<bool>(matrix->element(x, y)) * std::numeric_limits<ImageType::T>::max();
                }
            }

            d_bitmap.update();

        }
    }
}
} //namespace sofa::component::linearsolver
