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
#define SOFA_COMPONENT_ENGINE_BOXROI_CPP
#include <SofaEngine/BoxROI.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace engine
{

namespace boxroi
{


using namespace sofa::defaulttype;

SOFA_DECL_CLASS(BoxROI)

int BoxROIClass = core::RegisterObject("Find the primitives (vertex/edge/triangle/quad/tetrahedron/hexahedron) inside given boxes")
#ifdef SOFA_WITH_DOUBLE
        .add< BoxROI<Vec3dTypes> >(true) //default
        .add< BoxROI<Rigid3dTypes> >()
        .add< BoxROI<Vec6dTypes> >()
#endif //SOFA_WITH_DOUBLE
#ifdef SOFA_WITH_FLOAT
        .add< BoxROI<Vec3fTypes> >()
        .add< BoxROI<Rigid3fTypes> >()
        .add< BoxROI<Vec6fTypes> >()
#endif //SOFA_WITH_FLOAT
        ;

#ifdef SOFA_WITH_DOUBLE
template class SOFA_ENGINE_API BoxROI<Vec3dTypes>;
template class SOFA_ENGINE_API BoxROI<Rigid3dTypes>;
template class SOFA_ENGINE_API BoxROI<Vec6dTypes>;
#endif // SOFA_WITH_DOUBLE
#ifdef SOFA_WITH_FLOAT
template class SOFA_ENGINE_API BoxROI<Vec3fTypes>;
template class SOFA_ENGINE_API BoxROI<Rigid3fTypes>;
template class SOFA_ENGINE_API BoxROI<Vec6fTypes>;
#endif //SOFA_WITH_FLOAT

} // namespace boxroi

} // namespace constraint

} // namespace component

} // namespace sofa

