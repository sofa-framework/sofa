/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "MeshSTEPLoaderPlugin.h"
#include <sofa/core/Plugin.h>

#include "MeshSTEPLoader.h"
#include "SingleComponent.inl"
#include "STEPShapeMapping.h"
#include "ParametricTriangleTopologyContainer.h"

using sofa::component::loader::MeshSTEPLoader;
using sofa::component::engine::SingleComponent;
using sofa::component::engine::STEPShapeExtractor;
using sofa::component::topology::ParametricTriangleTopologyContainer;

class MeshSTEPLoaderPlugin: public sofa::core::Plugin {
public:
    MeshSTEPLoaderPlugin(): Plugin("MeshSTEPLoader") {
        setDescription("Load STEP files into SOFA Framework.");
        setVersion("0.5");
        setLicense("LGPL");

        addComponent<MeshSTEPLoader>("Specific mesh loader for STEP file format (see PluginMeshSTEPLoader.txt for further information).");

        addComponent<ParametricTriangleTopologyContainer>("Topology container for triangles with parametric coordinates.");

        addComponent<STEPShapeExtractor>("Extract a shape from a MeshSTEPLoader according to a shape number.");

        addComponent< SingleComponent<int> >("Load mesh of one shape, in the case there are several components.");
    }
};

SOFA_PLUGIN(MeshSTEPLoaderPlugin);

template class SOFA_MESHSTEPLOADER_API SingleComponent<int>;
