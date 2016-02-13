/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaGraphComponent/PauseAnimation.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace misc
{

PauseAnimation::PauseAnimation()
    : root(NULL)
{
}

PauseAnimation::~PauseAnimation()
{
}

void PauseAnimation::init()
{
    BaseObject::init();
    //simu = sofa::simulation::getSimulation();
    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext());
    root = dynamic_cast<simulation::Node *>(context->getRootContext());

    //root = dynamic_cast<sofa::core::objectmodel::BaseNode*>(this->getContext());
    // TODO: add methods in BaseNode to get parent nodes and/or root node
}

void PauseAnimation::pause()
{
    if (root)
        root->getContext()->setAnimate(false);
}

} // namespace misc

} // namespace component

} // namespace sofa
