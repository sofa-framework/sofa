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
#include "Colors.h"
#include <map>
#include <stdexcept>

namespace sofa::simulation::Colors
{

namespace
{
// See http://www.graphviz.org/doc/info/colors.html
// The following is mostly the "set312" colors
std::vector<std::string> SOFA_SIMULATION_CORE_API DEFAULTCOLORS = {
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
    /*MechanicalMapping     =*/ "#4ba3fa", // color spectral4/4 (brighter)
    /*Mapping               =*/ "#80b1d3", // color 5
    /*Mass                  =*/ "#ffffb3", // color 2
    /*Topology              =*/ "#ffed6f", // color 12
    /*VisualModel           =*/ "#eefdea", // color 11 (brighter)
    /*Loader                =*/ "#00daff", // cyan
    /*ConfigurationSetting  =*/ "#aaaaaa", // pale pink
};

std::map<const std::string, int> colors = {
        {"BaseNode",                NODE},
        {"BaseObject",              OBJECT},
        {"ContextObject",           CONTEXT},
        {"BehaviorModel",           BMODEL},
        {"CollisionModel",          CMODEL},
        {"MechanicalState",         MMODEL},
        {"ProjectiveConstraintSet", PROJECTIVECONSTRAINTSET},
        {"ConstraintSet",           CONSTRAINTSET},
        {"InteractionForceField",   IFFIELD},
        {"ForceField",              FFIELD},
        {"BaseAnimationLoop",       SOLVER},
        {"OdeSolver",               SOLVER}, // même valeur que ci-dessus
        {"CollisionPipeline",       COLLISION},
        {"MechanicalMapping",       MMAPPING},
        {"Mapping",                 MAPPING},
        {"Mass",                    MASS},
        {"Topology",                TOPOLOGY},
        {"VisualModel",             VMODEL},
        {"Loader",                  LOADER},
        {"ConfigurationSetting",    CONFIGURATIONSETTING}
};
}

bool hasColor(const std::string& className)
{
    return colors.find(className) != colors.end();
}

bool hasColor(const sofa::Index& id)
{
    return id < DEFAULTCOLORS.size();
}

const char* getColor(const std::string& className)
{
    if (auto it = colors.find(className); it != colors.end()) {
        return DEFAULTCOLORS[it->second].c_str();
    }
    throw std::runtime_error("No Color for this name");
}

const char* getColor(const sofa::Index id)
{
    if(id >= DEFAULTCOLORS.size())
        throw std::runtime_error("Color indice too large");
    return DEFAULTCOLORS[id].c_str();
}

sofa::Index registerColor(const std::string& hexColor)
{
    DEFAULTCOLORS.emplace_back(hexColor);
    return DEFAULTCOLORS.size()-1;
}

sofa::Index registerColor(const std::string& className, const std::string& hexColor)
{
    // Reuse an existing color or create a new one
    if(auto it = colors.find(className); it != colors.end())
    {
        DEFAULTCOLORS[it->second] = hexColor;
        return it->second;
    }
    auto id = registerColor(hexColor);
    colors[className] = id;
    return id;
}

const char* DeprecatedColor::operator[](size_t id)
{
    if(id >= DEFAULTCOLORS.size())
        throw std::runtime_error("Color indice too large");

    return DEFAULTCOLORS[id].c_str();
}
DeprecatedColor COLOR {};

}
