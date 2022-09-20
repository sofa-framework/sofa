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
    , d_bitmap(initData(&d_bitmap, type::BaseMatrixImageProxy(), "bitmap", "Visualization of the representation of the matrix as a binary image. White pixels are zeros, black pixels are non-zeros."))
    , l_linearSolver(initLink("linearSolver", "Link to the linear solver containing a matrix"))
{
    d_bitmap.setGroup("Image");
    d_bitmap.setWidget("matrixbitmap"); //the widget used to display the image is registered in a factory with the key 'matrixbitmap'
    d_bitmap.setReadOnly(true);

    this->f_listening.setValue(true);
}

GlobalSystemMatrixImage::~GlobalSystemMatrixImage() = default;

void GlobalSystemMatrixImage::doBaseObjectInit()
{
    Inherit1::doBaseObjectInit();

    if (!l_linearSolver)
    {
        l_linearSolver.set(this->getContext()->template get<sofa::core::behavior::LinearSolver>());
    }

    if (!l_linearSolver)
    {
        msg_error() << "No linear solver found in the current context, whereas it is required. This component retrieves the matrix from a linear solver.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }

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
        // even if the pointer to the matrix stays the same, the write accessor leads to an update of the widget
        auto& bitmap = *helper::getWriteOnlyAccessor(d_bitmap);
        bitmap.setMatrix(l_linearSolver->getSystemBaseMatrix());
    }
}
} //namespace sofa::component::linearsolver
