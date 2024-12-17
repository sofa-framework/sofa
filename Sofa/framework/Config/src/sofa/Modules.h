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
#define START_MODULE(name, fullModuleName)\
    private:\
    struct Module ## name\
    {\
    private:\
        static constexpr const char* moduleName = fullModuleName;\
    public:\
        Module ## name() = default;\
        Module ## name(const Module ## name&) = delete;\
        Module ## name& operator=(const Module ## name&) = delete;\
        operator std::string() const { return moduleName; }

#define END_MODULE(name)\
    };\
    public:\
    Module ## name name;

#define MODULE(name, fullModuleName) \
    START_MODULE(name, fullModuleName) END_MODULE(name)

inline struct ModuleSofa
{
private:
    static constexpr const char* moduleName = "Sofa";
public:
    ModuleSofa() = default;
    ModuleSofa(const ModuleSofa&) = delete;
    ModuleSofa& operator=(const ModuleSofa&) = delete;
    operator std::string() const { return moduleName; }

    START_MODULE(Component, "Sofa.Component")
        MODULE(AnimationLoop, "Sofa.Component.AnimationLoop")
        START_MODULE(Collision, "Sofa.Component.Collision")
            START_MODULE(Detection, "Sofa.Component.Collision.Detection")
                MODULE(Algorithm, "Sofa.Component.Collision.Detection.Algorithm")
                MODULE(Intersection, "Sofa.Component.Collision.Detection.Intersection")
            END_MODULE(Detection)
            MODULE(Geometry, "Sofa.Component.Collision.Geometry")
            START_MODULE(Response, "Sofa.Component.Collision.Response")
                MODULE(Contact, "Sofa.Component.Collision.Response.Contact")
                MODULE(Mapper, "Sofa.Component.Collision.Response.Mapper")
            END_MODULE(Response)
        END_MODULE(Collision)

        START_MODULE(Constraint, "Sofa.Component.Constraint")
            START_MODULE(Lagrangian, "Sofa.Component.Constraint.Lagrangian")
                MODULE(Correction, "Sofa.Component.Constraint.Lagrangian.Correction")
                MODULE(Model, "Sofa.Component.Constraint.Lagrangian.Model")
                MODULE(Solver, "Sofa.Component.Constraint.Lagrangian.Solver")
            END_MODULE(Lagrangian)

            MODULE(Projective, "Sofa.Component.Constraint.Projective")
        END_MODULE(Constraint)

        MODULE(Controller, "Sofa.Component.Controller")
        MODULE(Diffusion, "Sofa.Component.Diffusion")

        START_MODULE(Engine, "Sofa.Component.Engine")
            MODULE(Analyze, "Sofa.Component.Engine.Analyze")
            MODULE(Generate, "Sofa.Component.Engine.Generate")
            MODULE(Select, "Sofa.Component.Engine.Select")
            MODULE(Transform, "Sofa.Component.Engine.Transform")
        END_MODULE(Engine)

        MODULE(Haptics, "Sofa.Component.Haptics")

        START_MODULE(IO, "Sofa.Component.IO")
            MODULE(Mesh, "Sofa.Component.IO.Mesh")
        END_MODULE(IO)

        START_MODULE(LinearSolver, "Sofa.Component.LinearSolver")
            MODULE(Direct, "Sofa.Component.LinearSolver.Direct")
            MODULE(Iterative, "Sofa.Component.LinearSolver.Iterative")
            MODULE(Ordering, "Sofa.Component.LinearSolver.Ordering")
            MODULE(Preconditioner, "Sofa.Component.LinearSolver.Preconditioner")
        END_MODULE(LinearSolver)

        MODULE(LinearSystem, "Sofa.Component.LinearSystem")

        START_MODULE(Mapping, "Sofa.Component.Mapping")
            MODULE(Linear, "Sofa.Component.Mapping.Linear")
            MODULE(MappedMatrix, "Sofa.Component.Mapping.MappedMatrix")
            MODULE(NonLinear, "Sofa.Component.Mapping.NonLinear")
        END_MODULE(Mapping)

        MODULE(Mass, "Sofa.Component.Mass")
        MODULE(MechanicalLoad, "Sofa.Component.MechanicalLoad")

        START_MODULE(ODESolver, "Sofa.Component.ODESolver")
            MODULE(Backward, "Sofa.Component.ODESolver.Backward")
            MODULE(Forward, "Sofa.Component.ODESolver.Forward")
        END_MODULE(ODESolver)

        MODULE(Playback, "Sofa.Component.Playback")
        MODULE(SceneUtility, "Sofa.Component.SceneUtility")
        MODULE(Setting, "Sofa.Component.Setting")

        START_MODULE(SolidMechanics, "Sofa.Component.SolidMechanics")
            START_MODULE(FEM, "Sofa.Component.SolidMechanics.FEM")
                MODULE(Elastic, "Sofa.Component.SolidMechanics.FEM.Elastic")
                MODULE(HyperElastic, "Sofa.Component.SolidMechanics.FEM.HyperElastic")
                MODULE(NonUniform, "Sofa.Component.SolidMechanics.FEM.NonUniform")
            END_MODULE(FEM)

            MODULE(Spring, "Sofa.Component.SolidMechanics.Spring")
            MODULE(TensorMass, "Sofa.Component.SolidMechanics.TensorMass")
        END_MODULE(SolidMechanics)

        MODULE(StateContainer, "Sofa.Component.StateContainer")

        START_MODULE(Topology, "Sofa.Component.Topology")
            START_MODULE(Container, "Sofa.Component.Topology.Container")
                MODULE(Constant, "Sofa.Component.Topology.Container.Constant")
                MODULE(Dynamic, "Sofa.Component.Topology.Container.Dynamic")
                MODULE(Grid, "Sofa.Component.Topology.Container.Grid")
            END_MODULE(Container)
            MODULE(Mapping, "Sofa.Component.Topology.Mapping")
            MODULE(Utility, "Sofa.Component.Topology.Utility")
        END_MODULE(Topology)

        MODULE(Visual, "Sofa.Component.Visual")
    END_MODULE(Component)

    START_MODULE(GL, "Sofa.GL")
        START_MODULE(Component, "Sofa.GL.Component")
            MODULE(Engine, "Sofa.GL.Component.Engine")
            MODULE(Rendering2D, "Sofa.GL.Component.Rendering2D")
            MODULE(Rendering3D, "Sofa.GL.Component.Rendering3D")
            MODULE(Shader, "Sofa.GL.Component.Shader")
        END_MODULE(Component)
    END_MODULE(GL)

    START_MODULE(GUI, "Sofa.GUI")
        MODULE(Component, "Sofa.GUI.Component")
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
