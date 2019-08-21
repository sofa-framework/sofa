/** New namespace organization proposal by Jeremie A.

    Short version without classes and discussions.
**/

namespace Sofa
{

/// Utility helper classes that we need, but that are not the core of the
/// plateform.
namespace Utils
{

/// Basic design patterns and data structures
// class Factory, Dispatcher, fixed_array, vector, ...

/// Image and Mesh I/O
namespace IO {}

/// GL drawing helper classes, no actual visual models
namespace GL {}

/// OS-specific classes
namespace System
{
/// Portable multithreading helper classes (thread, mutex, ...), no
/// full-featured scheduler.
namespace Thread {}
}

} // namespace Utils

/// Default data types for the most common cases (1D/2D/3D vectors, rigid
/// frames).
namespace Default
{
// class Vec, Mat, Quat, StdVectorTypes, RigidTypes, ...
} // namespace Default

/// Base standardized classes
namespace Framework
{

/// Object Model
/// Specifies how pointer to generic objects and context are handled, as
/// well as the basic functionnalities of all objects (name, fields, ...).
namespace ObjectModel {}

/// Abstract classes defining the 'high-level' sofa models
/// They could be in a sub-namespace, but I didn't find any name for it ;)
/// Also being in the main namespace shows that they define the primary
/// level of the framework.
// class BehaviorModel, VisualModel, CollisionModel, BasicMapping

/// Component Model
/// A composent is defined as an object with a specific role in a Sofa
/// simulation.
namespace ComponentModel
{
namespace Behavior {}
namespace Topology {}
namespace Collision {}
}

/// Simulation Model: class handling the whole simulation (attaching
/// components, scheduling computations).
namespace SimulationModel {}

} // namespace Framework

/// Implementation of components
namespace Components
{

namespace Topology {}

/// Components used for most simulated bodies
namespace CommonBehavior {}

/// Components used only for deformable bodies
namespace Deformable {}
namespace Springs {}
namespace FEM {}
namespace Tensor {}

/// Components used only for rigid bodies
namespace Rigid {}

/// Components used only for fluids
namespace Fluid
{
namespace LennardJones {}
namespace SPH {}
namespace Euler {}
}

/// Collision related components
namespace Collision {}

/// Penality-based interactions
namespace Penality {}

/// LM based interactions
namespace LagrangianMultipliers {}

/// User interactions
namespace UserInteraction {}

/// OpenGL visual models
namespace GL {}

} // namespace Components

/// Simulation data structure and scheduling (implementation of
/// Framework::SimulationModel).
namespace Simulation
{

/// Run-time model for a tree structure.
/// (current Components::Graph namespace)
namespace Tree
{
/// XML I/O classes
namespace XML {}
}

/// Automate-based multithread scheduler.
/// (current Components::Thread namespace)
namespace AutomateScheduler {}

/// To be completed by other data structures and schedulers...
/// (network, multiple frequencies)

} // namespace Simulation

/// User Interfaces (no change)
namespace GUI
{
namespace QT {}
namespace FLTK {}
} // namespace GUI

} // namespace Sofa
