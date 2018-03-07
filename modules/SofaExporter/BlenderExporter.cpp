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
#include <SofaExporter/BlenderExporter.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

    namespace component
    {

        namespace misc
        {
            using namespace defaulttype;

            SOFA_DECL_CLASS(BlenderExporter)

                int BlenderExportClass = core::RegisterObject("Export the simulation result as blender point cache files")
#ifndef SOFA_FLOAT
                .add< BlenderExporter<Vec3dTypes> >()
                .add< BlenderExporter<Rigid3dTypes> >()
				.add< BlenderExporter<ExtVec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
				.add< BlenderExporter<Vec3fTypes> >()
				.add< BlenderExporter<Rigid3fTypes> >()
				.add< BlenderExporter<ExtVec3fTypes> >()
#endif
				;

        } // namespace misc

    } // namespace component

} // namespace sofa
