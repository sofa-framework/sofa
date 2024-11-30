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
#pragma once

#include <string>

/**
 * The macro allows a simpler definition and maintenance of the module data
 * structure
 * @param name Module name
 */
#define START_MODULE(name)\
    private:\
    template<class Parent>\
    struct Module ## name\
    {\
    private:\
        static inline std::string moduleName = Parent::moduleName + "." + #name;\
        using MyClass = Module ## name;\
    public:\
        Module ## name() = default;\
        Module ## name(const Module ## name&) = delete;\
        Module ## name& operator=(const Module ## name&) = delete;\
        operator std::string() const { return moduleName; }

#define END_MODULE(name)\
    };\
    public:\
    Module ## name<MyClass> name;

#define MODULE(name) \
    START_MODULE(name) END_MODULE(name)

inline struct ModuleSofa
{
private:
    static inline std::string moduleName = "Sofa";
    using MyClass = ModuleSofa;
public:
    ModuleSofa() = default;
    ModuleSofa(const ModuleSofa&) = delete;
    ModuleSofa& operator=(const ModuleSofa&) = delete;
    operator std::string() const { return moduleName; }

    START_MODULE(Component)
        MODULE(AnimationLoop)
        START_MODULE(Collision)
            START_MODULE(Detection)
                MODULE(Algorithm)
                MODULE(Intersection)
            END_MODULE(Detection)
            MODULE(Geometry)
            START_MODULE(Response)
                MODULE(Contact)
                MODULE(Mapper)
            END_MODULE(Response)
        END_MODULE(Collision)

        START_MODULE(Constraint)
            START_MODULE(Lagrangian)
                MODULE(Correction)
                MODULE(Model)
                MODULE(Solver)
            END_MODULE(Lagrangian)

            MODULE(Projective)
        END_MODULE(Constraint)

        MODULE(Controller)
        MODULE(Diffusion)

        START_MODULE(Engine)
            MODULE(Analyze)
            MODULE(Generate)
            MODULE(Select)
            MODULE(Transform)
        END_MODULE(Engine)

        MODULE(Haptics)

        START_MODULE(IO)
            MODULE(Mesh)
        END_MODULE(IO)

        START_MODULE(LinearSolver)
            MODULE(Direct)
            MODULE(Iterative)
            MODULE(Ordering)
            MODULE(Preconditioner)
        END_MODULE(LinearSolver)

        MODULE(LinearSystem)

        START_MODULE(Mapping)
            MODULE(Linear)
            MODULE(NonLinear)
        END_MODULE(Mapping)

        MODULE(Mass)
        MODULE(MechanicalLoad)

        START_MODULE(ODESolver)
            MODULE(Backward)
            MODULE(Forward)
        END_MODULE(ODESolver)

        MODULE(Playback)
        MODULE(SceneUtility)
        MODULE(Setting)

        START_MODULE(SolidMechanics)
            START_MODULE(FEM)
                MODULE(Elastic)
                MODULE(HyperElastic)
                MODULE(NonUniform)
            END_MODULE(FEM)

            MODULE(Spring)
            MODULE(TensorMass)
        END_MODULE(SolidMechanics)

        MODULE(StateContainer)

        START_MODULE(Topology)
            START_MODULE(Container)
                MODULE(Constant)
                MODULE(Dynamic)
                MODULE(Grid)
            END_MODULE(Container)
            MODULE(Mapping)
            MODULE(Utility)
        END_MODULE(Topology)

        MODULE(Visual)
    END_MODULE(Component)

    START_MODULE(GL)
        START_MODULE(Component)
            MODULE(Engine)
            MODULE(Rendering2D)
            MODULE(Rendering3D)
            MODULE(Shader)
        END_MODULE(Component)
    END_MODULE(GL)

    START_MODULE(GUI)
        MODULE(Component)
    END_MODULE(GUI)

}
/**
 * Nested data structure reproducing the architecture of SOFA modules (tree-like
 * structure). All modules and submodules can convert implicitly to std::string.
 * This allows to use this structure as a string in functions accepting a module
 * name as a string. From the developer point of view, the use of this data
 * structure is less error-prone compared to any string and the auto-completion
 * allows faster developments.
 */
Sofa;

#undef START_MODULE
#undef END_MODULE
#undef MODULE
