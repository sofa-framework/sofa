/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_COMPONENT_ENGINE_TESTIMAGEENGINE_CPP
#include "TestImageEngine.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(TestImageEngine)

int TestImageEngineClass = core::RegisterObject("TestImageEngine to test engine with data image")

        .add<TestImageEngine<ImageUC> >(true)
        .add<TestImageEngine<ImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<TestImageEngine<ImageC> >()
        .add<TestImageEngine<ImageI> >()
        .add<TestImageEngine<ImageUI> >()
        .add<TestImageEngine<ImageS> >()
        .add<TestImageEngine<ImageUS> >()
        .add<TestImageEngine<ImageL> >()
        .add<TestImageEngine<ImageUL> >()
        .add<TestImageEngine<ImageF> >()
        .add<TestImageEngine<ImageB> >()
#endif
        ;

template class TestImageEngine<ImageUC>;
template class TestImageEngine<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class TestImageEngine<ImageC>;
template class TestImageEngine<ImageI>;
template class TestImageEngine<ImageUI>;
template class TestImageEngine<ImageS>;
template class TestImageEngine<ImageUS>;
template class TestImageEngine<ImageL>;
template class TestImageEngine<ImageUL>;
template class TestImageEngine<ImageF>;
template class TestImageEngine<ImageB>;
#endif
} // namespace constraint

} // namespace component

} // namespace sofa

