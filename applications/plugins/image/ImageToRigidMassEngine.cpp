/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_IMAGE_ImageToRigidMassEngine_CPP

#include "ImageToRigidMassEngine.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ImageToRigidMassEngine)

int ImageToRigidMassEngineClass = core::RegisterObject("Compute rigid mass from a density image")
        .add<ImageToRigidMassEngine<ImageD> >(true)
        .add<ImageToRigidMassEngine<ImageB> >()
        .add<ImageToRigidMassEngine<ImageUC> >()
//#ifdef BUILD_ALL_IMAGE_TYPES
//        .add<ImageToRigidMassEngine<ImageC> >()
//        .add<ImageToRigidMassEngine<ImageI> >()
//        .add<ImageToRigidMassEngine<ImageUI> >()
//        .add<ImageToRigidMassEngine<ImageS> >()
//        .add<ImageToRigidMassEngine<ImageUS> >()
//        .add<ImageToRigidMassEngine<ImageL> >()
//        .add<ImageToRigidMassEngine<ImageUL> >()
//        .add<ImageToRigidMassEngine<ImageF> >()
//#endif
        ;

template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageD>;
template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageB>;
template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageUC>;
//#ifdef BUILD_ALL_IMAGE_TYPES
//template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageC>;
//template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageI>;
//template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageUI>;
//template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageS>;
//template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageUS>;
//template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageL>;
//template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageUL>;
//template class SOFA_IMAGE_API ImageToRigidMassEngine<ImageF>;
//#endif


} //
} // namespace component

} // namespace sofa

