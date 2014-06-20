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
#include "ManifoldTopologies.h"
#include <sofa/core/Plugin.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include "ManifoldEdgeSetGeometryAlgorithms.h"
#include "ManifoldEdgeSetTopologyModifier.h"
#include "ManifoldEdgeSetTopologyContainer.h"
#include "ManifoldEdgeSetTopologyAlgorithms.inl"
#include "ManifoldTriangleSetTopologyModifier.h"
#include "ManifoldTriangleSetTopologyContainer.h"
#include "ManifoldTriangleSetTopologyAlgorithms.inl"
#include "ManifoldTetrahedronSetTopologyContainer.h"

using namespace sofa::component::topology;
using namespace sofa::defaulttype;


class ManifoldTopologiesPlugin: public sofa::core::Plugin {
public:
    ManifoldTopologiesPlugin(): Plugin("ManifoldTopologies") {
        setDescription("Use Manifold Topologies functionalities in SOFA.");
        setVersion("1.0");

#ifdef SOFA_FLOAT
        addComponent< ManifoldEdgeSetGeometryAlgorithms<Vec3fTypes> >();
#else
        addComponent< ManifoldEdgeSetGeometryAlgorithms<Vec3dTypes> >();
#endif
        setDescription("ManifoldEdgeSetGeometryAlgorithms", "Manifold Edge set geometry algorithms.");
#if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
        addTemplateInstance< ManifoldEdgeSetGeometryAlgorithms<Vec3fTypes> >();
#endif
#ifndef SOFA_FLOAT
        addTemplateInstance< ManifoldEdgeSetGeometryAlgorithms<Vec2dTypes> >();
        addTemplateInstance< ManifoldEdgeSetGeometryAlgorithms<Vec1dTypes> >();
        addTemplateInstance< ManifoldEdgeSetGeometryAlgorithms<Rigid3dTypes> >();
        addTemplateInstance< ManifoldEdgeSetGeometryAlgorithms<Rigid2dTypes> >();
#endif
#ifndef SOFA_DOUBLE
        addTemplateInstance< ManifoldEdgeSetGeometryAlgorithms<Vec2fTypes> >();
        addTemplateInstance< ManifoldEdgeSetGeometryAlgorithms<Vec1fTypes> >();
        addTemplateInstance< ManifoldEdgeSetGeometryAlgorithms<Rigid3fTypes> >();
        addTemplateInstance< ManifoldEdgeSetGeometryAlgorithms<Rigid2fTypes> >();
#endif

#ifdef SOFA_FLOAT
        addComponent< ManifoldEdgeSetTopologyAlgorithms<Vec3fTypes> >();
#else
        addComponent< ManifoldEdgeSetTopologyAlgorithms<Vec3dTypes> >();
#endif
        setDescription("ManifoldEdgeSetTopologyAlgorithms", "ManifoldEdge set topology algorithms.");
#if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
        addTemplateInstance< ManifoldEdgeSetTopologyAlgorithms<Vec3fTypes> >();
#endif
#ifndef SOFA_FLOAT
        addTemplateInstance< ManifoldEdgeSetTopologyAlgorithms<Vec2dTypes> >();
        addTemplateInstance< ManifoldEdgeSetTopologyAlgorithms<Vec1dTypes> >();
        addTemplateInstance< ManifoldEdgeSetTopologyAlgorithms<Rigid3dTypes> >();
        addTemplateInstance< ManifoldEdgeSetTopologyAlgorithms<Rigid2dTypes> >();
#endif
#ifndef SOFA_DOUBLE
        addTemplateInstance< ManifoldEdgeSetTopologyAlgorithms<Vec2fTypes> >();
        addTemplateInstance< ManifoldEdgeSetTopologyAlgorithms<Vec1fTypes> >();
        addTemplateInstance< ManifoldEdgeSetTopologyAlgorithms<Rigid3fTypes> >();
        addTemplateInstance< ManifoldEdgeSetTopologyAlgorithms<Rigid2fTypes> >();
#endif

        addComponent<ManifoldEdgeSetTopologyContainer>("Manifold Edge set topology container.");

        addComponent<ManifoldEdgeSetTopologyModifier>("Manifold Edge set topology modifier.");

#ifdef SOFA_FLOAT
        addComponent< ManifoldTriangleSetTopologyAlgorithms<Vec3fTypes> >();
#else
        addComponent< ManifoldTriangleSetTopologyAlgorithms<Vec3dTypes> >();
#endif
        setDescription("ManifoldTriangleSetTopologyAlgorithms", "Manifold Triangle set topology algorithms.");
#if !defined(SOFA_DOUBLE) && !defined(SOFA_FLOAT)
        addTemplateInstance< ManifoldTriangleSetTopologyAlgorithms<Vec3fTypes> >();
#endif
#ifndef SOFA_FLOAT
        addTemplateInstance< ManifoldTriangleSetTopologyAlgorithms<Vec2dTypes> >();
        addTemplateInstance< ManifoldTriangleSetTopologyAlgorithms<Vec1dTypes> >();
        // addTemplateInstance< ManifoldTriangleSetTopologyAlgorithms<Rigid3dTypes> >();
        // addTemplateInstance< ManifoldTriangleSetTopologyAlgorithms<Rigid2dTypes> >();
#endif
#ifndef SOFA_DOUBLE
        addTemplateInstance< ManifoldTriangleSetTopologyAlgorithms<Vec2fTypes> >();
        addTemplateInstance< ManifoldTriangleSetTopologyAlgorithms<Vec1fTypes> >();
        // addTemplateInstance< ManifoldTriangleSetTopologyAlgorithms<Rigid3fTypes> >();
        // addTemplateInstance< ManifoldTriangleSetTopologyAlgorithms<Rigid2fTypes> >();
#endif

        addComponent<ManifoldTriangleSetTopologyContainer>("Manifold Triangle set topology container.");

        addComponent<ManifoldTriangleSetTopologyModifier>("Triangle set topology manifold modifier.");

        addComponent<ManifoldTetrahedronSetTopologyContainer>("Manifold Tetrahedron set topology container.");
    }
};

SOFA_PLUGIN(ManifoldTopologiesPlugin);
