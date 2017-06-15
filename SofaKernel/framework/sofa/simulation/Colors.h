/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_SIMULATION_COLORS_H
#define SOFA_SIMULATION_COLORS_H

#include <stdlib.h>
#include <string.h>

namespace sofa
{

namespace simulation
{


namespace Colors
{

enum
{
    NODE,
    OBJECT,
    CONTEXT,
    BMODEL,
    CMODEL,
    MMODEL,
    PROJECTIVECONSTRAINTSET,
    CONSTRAINTSET,
    IFFIELD,
    FFIELD,
    SOLVER,
    COLLISION,
    MMAPPING,
    MAPPING,
    MASS,
    TOPOLOGY,
    VMODEL,
    LOADER ,
    CONFIGURATIONSETTING,
    ALLCOLORS
};

// See http://www.graphviz.org/doc/info/colors.html
// The following is mostly the "set312" colors

static const char* COLOR[ALLCOLORS]=
{
    /*Node                  =*/ "#dedede", // color 9
    /*Object                =*/ "#ffffff", // white
    /*Context               =*/ "#d7191c", // color spectral4/1
    /*BehaviorModel         =*/ "#93ff49", // color 7 (brighter)
    /*CollisionModel        =*/ "#fccde5", // color 8
    /*MechanicalState       =*/ "#8dd3c7", // color 1
    /*ProjectiveConstraintSet  =*/ "#fdb462", // color 6
    /*ConstraintSet         =*/ "#f98912", // color 6
    /*InteractionForceField =*/ "#fb8072", // color 4
    /*ForceField            =*/ "#bebada", // color 3
    /*Solver                =*/ "#b3de69", // color 7
    /*CollisionPipeline     =*/ "#bc80bd", // color 10
    /*MechanicalMapping     =*/ "#2b83da", // color spectral4/4
    /*Mapping               =*/ "#80b1d3", // color 5
    /*Mass                  =*/ "#ffffb3", // color 2
    /*Topology              =*/ "#ffed6f", // color 12
    /*VisualModel           =*/ "#eefdea", // color 11 (brighter)
    /*Loader                =*/ "#00daff", // cyan
    /*ConfigurationSetting  =*/ "#aaaaaa", // pale pink
};

inline const char* getColor(const char* classname)
{
    if (!strcmp(classname,"BaseNode")) return COLOR[NODE];
    if (!strcmp(classname,"BaseObject")) return COLOR[OBJECT];
    if (!strcmp(classname,"ContextObject")) return COLOR[CONTEXT];
    if (!strcmp(classname,"BehaviorModel")) return COLOR[BMODEL];
    if (!strcmp(classname,"CollisionModel")) return COLOR[CMODEL];
    if (!strcmp(classname,"MechanicalState")) return COLOR[MMODEL];
    if (!strcmp(classname,"ProjectiveConstraintSet")) return COLOR[PROJECTIVECONSTRAINTSET];
    if (!strcmp(classname,"ConstraintSet")) return COLOR[CONSTRAINTSET];
    if (!strcmp(classname,"InteractionForceField")) return COLOR[IFFIELD];
    if (!strcmp(classname,"ForceField")) return COLOR[FFIELD];
    if (!strcmp(classname,"BaseAnimationLoop")) return COLOR[SOLVER];
    if (!strcmp(classname,"OdeSolver")) return COLOR[SOLVER];
    if (!strcmp(classname,"CollisionPipeline")) return COLOR[COLLISION];
    if (!strcmp(classname,"MechanicalMapping")) return COLOR[MMAPPING];
    if (!strcmp(classname,"Mapping")) return COLOR[MAPPING];
    if (!strcmp(classname,"Mass")) return COLOR[MASS];
    if (!strcmp(classname,"Topology")) return COLOR[TOPOLOGY];
    if (!strcmp(classname,"VisualModel")) return COLOR[VMODEL];
    if (!strcmp(classname,"Loader")) return COLOR[LOADER];
    if (!strcmp(classname,"ConfigurationSetting")) return COLOR[CONFIGURATIONSETTING];
    return "";

}

}


} // namespace simulation

} // namespace sofa

#endif
