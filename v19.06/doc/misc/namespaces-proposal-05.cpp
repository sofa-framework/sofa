/** New namespace organization proposal by Jeremie A.

Note that I did not include most of the new topology and matrix-related
classes in this version
The names are not final. Some cleanups are required, such as always use
Base or Basic but not both.
I specified the parent class of all classes except final components, as
they are very important to see what the technical requirements are.
The criteria I used are:
- separate "standardized" classes from implementation and auxiliary classes
- group classes related to each other
- all dependencies are one-way, to classes and namespaces defined before in
the document, which is why Utils namespace is first, and Default is
before Framework (as BaseContext needs it)
- limit the number of namespaces
- limit the recursion level of sub-namespaces (max 3 levels currently)
- limit redundancies (for instance components type such as Mass or
ForceField is often already mentioned in the class name)
**/

/** Suggestions by Francois Faure, 13/12/06
- The components have no idea of the data structure (tree, network...) and the scheduling (sequential, parallel,...). This is a strength of the design, because it guarantees that we can use the components in a broad variety of run-time architectures. To make sure that this feature remains, I suggest to suppress namespace Sofa::Framework::SimulationModel and to move its classes to Sofa::Simulation.
- the suffix "Model" seems to mean "Abstract", so I suggest to rename Framework::ObjectModel to Framework::Object
- "Deformable" normally includes fluids. I suggest to rename Component::Deformable to Component::Solid, since "Solid" is commonly used to denote viscoelastic deformable bodies, as opposed to "Rigid"
- Component::CommonBehavior gathers a lot of different things. I suggest to split it in several, more focused, namespaces.
*/

/* Comments from Christian : 13/12/06
" Solid is commonly used to denote viscoelastic deformable bodies " (are you sure ?.. I think "solid" also includes rigid bodies !!!) => I think the distinction - deformable / rigid / fluid - is more clear
Lagrange multipliers are specific treatment of constraints... I put it in constraint... but I think it would be much better to create a new class of lagrangeMultiplierSolver in numericalSolver and suppress these class
Mass is part of the mechanical law of each type of mechanical object and changes depending the type of object. (for instance, mass is integrated using FEM interpolation function
is not the same than - mass + inertia - used for rigid bodies, even if in the particular case of diagonal mass, it could be close)
I suggest to distinguish solver and time-integrator(ODE solver).

I propose to put RigidMapping and ImplicitSurfaceMapping in the namespace "Mapping"... so that in the namespaces:
- Deformable
- Rigid
- fluid
People find only mechanical laws which concerns Deformable, Rigid and Fluid
Moreover, RigidMapping can be used for some class of deformable objects (such as global CorotionalModels or quasi-rigid objects) and
ImplicitSurfaceMapping for some deformable objects (such as vessels)

/*Comments from Laure : 16/01/06
Towards a more flat tree : less namespaces because they seem to be redundant.
I just talk about namespacse without specifying the classes.
*/


proposal-05
{
    namespace Sofa {
    namespace Utils{// The same
    namespace IO ;
    namespace GL ;
    namespace System ;
    }// namespace Utils
    namespace Core{// Instead of Framework
    namespace Default;
    //No more namespace Object;
    // No more namespace ComponentModel {
    //namespace Behavior ;
    //namespace Topology ;
    //namespace Collision ;
    //}// namespace ComponentModel
    }// namespace Core
    namespace Components{
    //No more namespace Topology;
    //No more namespace ODESolver;
    //No more namespace NumericalSolver;
    //No more namespace Mapping;
    namespace Context;
    namespace Deformable{
    //No more namespace Springs;
    //No more namespace FEM ;
    //No more namespace Tensor;
    //No more namespace Mass;
    }
    namespace Rigid{
    //No more namespace Mass;
    //No more namespace Springs;
    }
    namespace Fluid {
    //No more namespace LennardJones;
    //No more namespace SPH;
    //No more namespace Euler;
    }
    //No more namespace Collision;
    //No more namespace InteractionForce;
    //No more namespace Constraint ;
    namespace UserInteraction ;
    namespace GL ;
    }// namespace Components
    namespace Simulation{
    namespace Tree {
    namespace XML ;
    }
    namespace AutomateScheduler;
    }// namespace Simulation
    namespace GUI {
    namespace QT;
    namespace FLTK ;
    } // namespace GUI

    } // namespace Sofa
}

proposal-04
{
    namespace Sofa {
    namespace Utils{
    namespace IO ;
    namespace GL ;
    namespace System ;
    }// namespace Utils
    namespace Default;
    namespace Framework{
    namespace Object;
    namespace ComponentModel {
    namespace Behavior ;
    namespace Topology ;
    namespace Collision ;
    }// namespace ComponentModel
    }// namespace Framewaok
    namespace Components{
    namespace Topology;
    namespace ODESolver;
    namespace NumericalSolver;
    namespace Mapping;
    namespace Context;
    namespace Deformable{
    namespace Springs;
    namespace FEM ;
    namespace Tensor;
    namespace Mass;
    }
    namespace Rigid{
    namespace Mass;
    namespace Springs;
    }
    namespace Fluid {
    namespace LennardJones;
    namespace SPH;
    namespace Euler;
    }
    namespace Collision;
    namespace InteractionForce;
    namespace Constraint ;
    namespace UserInteraction ;
    namespace GL ;
    }// namespace Components
    namespace Simulation{
    namespace Tree {
    namespace XML ;
    }
    namespace AutomateScheduler;
    }// namespace Simulation
    namespace GUI {
    namespace QT;
    namespace FLTK ;
    } // namespace GUI

    } // namespace Sofa
}

