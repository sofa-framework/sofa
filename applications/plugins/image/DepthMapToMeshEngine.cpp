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
#define SOFA_IMAGE_DEPTHMAPTOMESHENGINE_CPP

#include "DepthMapToMeshEngine.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

void registerDepthMapToMeshEngine(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Compute a mesh from a depth map image")
    .add<DepthMapToMeshEngine<ImageUC> >(true)
    .add<DepthMapToMeshEngine<ImageD> >()
    .add<DepthMapToMeshEngine<ImageB> >()
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
    .add<DepthMapToMeshEngine<ImageC> >()
    .add<DepthMapToMeshEngine<ImageI> >()
    .add<DepthMapToMeshEngine<ImageUI> >()
    .add<DepthMapToMeshEngine<ImageS> >()
    .add<DepthMapToMeshEngine<ImageUS> >()
    .add<DepthMapToMeshEngine<ImageL> >()
    .add<DepthMapToMeshEngine<ImageUL> >()
    .add<DepthMapToMeshEngine<ImageF> >()
#endif    
        );
}


template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageUC>;
template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageD>;
template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageB>;
#if PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL
template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageC>;
template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageI>;
template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageUI>;
template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageS>;
template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageUS>;
template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageL>;
template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageUL>;
template class SOFA_IMAGE_API DepthMapToMeshEngine<ImageF>;
#endif // PLUGIN_IMAGE_COMPILE_SET == PLUGIN_IMAGE_COMPILE_SET_FULL

} //
} // namespace component

} // namespace sofa

