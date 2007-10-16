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

namespace Sofa
{

/// Utility helper classes that we need, but that are not the core of the
/// plateform.
/// Old "Components::Common" namespace, moved out of Components as it does not
/// contain any component.
/// Alternate name: Common, Helpers
namespace Utils
{

/// Basic design patterns and data structures
class Factory;
class Dispatcher;
class TypeInfo;
class fixed_array; // from Boost
class static_assert; // from Boost
class vector; // std::vector with range cheching in debug mode
class ext_vector; // vector where data is stored on a given location

/// Image and Mesh I/O
namespace IO
{
class Image;
class ImageBMP : Image;
class ImagePNG : Image;
class Mesh;
class MeshOBJ : Mesh;
}

/// GL drawing helper classes, no actual visual models
namespace GL
{
class Axis;
class Shader;
class Texture;
...
}

/// OS-specific classes
namespace System
{
SetDirectory;

/// Portable multithreading helper classes (thread, mutex, ...), no
/// full-featured scheduler.
namespace Thread
{
class BaseThread;
class Mutex;
class Signal;
}
}

} // namespace Utils

/// Default data types for the most common cases (1D/2D/3D vectors, rigid
/// frames).
/// Users can define other types, but it is best to try to use these when
/// applicable, as many components are already instanciated with them.
/// It is however not a requirement (nothing prevents a user to define his own
/// Vec3 class for instance).
/// Alternate name: DefaultTypes
namespace Default
{

template<N,Real> class Vec;
template<N,Real> class Mat;
template<Real> class Quat;
template<N,Real> class StdVectorTypes;
template<N,Real> class ExtVectorTypes;

template<Real> class Frame;
template<Real> class SpatialVelocity;
template<Real> class RigidTypes;
template<Real> class RigidMass;

} // namespace Default

/// Base standardized classes
/// Old "Abstract" namespace, renamed as not all classes are abstract
/// Also contains old Core and Components::Collision namespaces
/// Alternate name: Base, Core, API
namespace Framework
{

/// Object Model
/// Specifies how pointer to generic objects and context are handled, as
/// well as the basic functionnalities of all objects (name, fields, ...).
namespace ObjectModel
{
class Base;
class BaseField;
class Event;
class BaseContext : Base; // Note that here we use Default::Vec3 for gravity, and Default::Frame for local coordinate system
class BaseObject : Base;
class ContextObject : BaseObject;

/// The following classes provide a default implementation.
/// It is unclear if they should be in another namespace, as
/// there are no real benefit in providing alternate implementations.

class Field : BaseField;
class DataField : BaseField;
class Context : BaseContext;
class KeypressedEvent : Event;
}

/// Abstract classes defining the 'high-level' sofa models
/// They could be in a sub-namespace, but I didn't find any name for it ;)
/// Also being in the main namespace shows that they define the primary
/// level of the framework.
/// Note that they *must* derive from BaseObject, otherwise operations such
/// as gnode->addObject(visualmodel) would not be possible.

class BehaviorModel : BaseObject;
class CollisionModel : BaseObject;
class CollisionElementIterator; // Can be moved inside CollisionModel
class VisualModel : BaseObject;
class BasicMapping : BaseObject;
template<In,Out> class Mapping : BasicMapping; // Remove?

/// Component Model
/// A composent is defined as an object with a specific role in a Sofa
/// simulation.
namespace ComponentModel
{
namespace Behavior   /// or Mechanics, Animation
{
class OdeSolver : BaseObject; // Derive from BehaviorModel? Rename to Animator, Integrator?
class BasicMechanicalModel : BaseObject; // Rename to DOFContainer, MechanicalDOF?
class BasicMechanicalMapping : BasicMapping;
class BasicConstraint : BaseObject;
class BasicForceField : BaseObject;
class BasicMass : BaseObject;
class InteractionForceField : BaseObject;

/// 'templated' classes

template<DataTypes> class MechanicalModel : BasicMechanicalModel;
template<DataTypes> class Constraint : BasicConstraint;
template<DataTypes> class ForceField : BasicForceField;
template<DataTypes> class Mass : ForceField<DataTypes>, BasicMass;
template<In,Out> class MechanicalMapping : BasicMechanicalMapping;
}
namespace Topology
{
class BasicTopology : BaseObject;
class BasicTopologicalMapping : BasicMapping;
class TopologyAlgorithms;
class TopologyChange;
class TopologyContainer;
class TopologyModifier;
template<In,Out> class TopologicalMapping : BasicTopologicalMapping;
}
namespace Collision
{
class Pipeline : BaseObject;
class ElementIntersector;
class Intersection : BaseObject;
class DetectionOutput;
class Detection : BaseObject;
class BroadPhaseDetection : Detection;
class BruteForceDetection : Detection;
class Contact;
class ContactManager : BaseObject;
class CollisionGroupManager : BaseObject;
}
}

/// Simulation Model: class handling the whole simulation (attaching
/// components, scheduling computations).
/// Alternate names: ContainerModel, SceneModel, ControlModel
namespace SimulationModel
{
class BaseNode : BaseContext;
class BaseSimulation; // TODO: define from interface of Graph::Simulation
class BaseScheduler; // TODO: define from interface of Graph::ActionScheduler
}

} // namespace Framework

/// Implementation of components
namespace Components
{

namespace Topology
{
MeshTopology;
GridTopology;
RegularGridTopology;
TrimmedRegularGridTopology;
}

/// Components used for most simulated bodies
namespace CommonBehavior
{
MechanicalObject;

EulerSolver;
RungeKutta4Solver;
CgImplicitSolver;
StaticSolver;

DiagonalMass;
UniformMass;

ExternalForceField;
PlaneForceField;

FixedConstraint;
BoxConstraint;
OscillatorConstraint;

BarycentricMapping;
IdentityMapping;

Gravity;
}

/// Components used only for deformable bodies
namespace Deformable
{
MechanicalObject<Vec3Types>; // instantiation of MechanicalObject on Vec3 types
DiagonalMass<Vec3Types>; // instantiation of DiagonalMass on Vec3 types
UniformMass<Vec3Types>; // instantiation of UniformMass on Vec3 types

namespace Springs
{
SpringForceField;
StiffSpringForceField;
MeshSpringForceField;
RegularGridSpringForceField;
}

namespace FEM
{
TetrahedronFEMForceField;
TriangleFEMForceField;
}

namespace Tensor
{
TensorForceField;
}
}

/// Components used only for rigid bodies
namespace Rigid
{
MechanicalObject<RigidTypes>; // instantiation of MechanicalObject on rigid types
DiagonalMass<RigidTypes>; // instantiation of DiagonalMass on rigid types
UniformMass<RigidTypes>; // instantiation of UniformMass on rigid types
RigidMapping;
RigidConstraintSolver; // hypothetical rigid body constraint solver
}

/// Components used only for fluids
namespace Fluid
{
SpatialGridContainer;
ImplicitSurfaceMapping;

namespace LennardJones
{
LennardJonesForceField;
}

namespace SPH
{
SPHFluidForceField;
}

namespace Euler
{
Grid2D;
Fluid2D;
Grid3D;
Fluid3D;
}
}

/// Collision related components
namespace Collision
{
SphereModel;
CubeModel;
PointModel;
EdgeModel;
TriangleModel;

PipelineSofa; // Rename to DefaultPipeline?
ContinuousIntersection;
DiscreteIntersection;
ProximityIntersection;
BruteForceDetection;
ContactManagerSofa; // Rename to DefaultContactManager?
CollisionGroupManagerSofa; // Rename to DefaultCollisionGroupManager?
SolverMerger;
}

/// Penality-based interactions
namespace Penality
{
BarycentricPenalityContact;
RepulsiveSpringForceField;
}

/// LM based interactions
namespace LagrangianMultipliers
{
LagrangianMultiplierConstraint;
LagrangianMultiplierFixedConstraint;
LagrangianMultiplierAttachConstraint;
LagrangianMultiplierContactConstraint;
}

/// User interactions
namespace UserInteraction
{
RayModel;
RayContact;
RayPickInteractor;
}

/// OpenGL visual models
namespace GL
{
OglModel;
}
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
namespace XML
{
BaseElement; // currently BaseNode
Element; // currently Node
NodeElement; // currently NodeNode
ObjectElement; // currently ObjectNode
}

Action;
ActionScheduler;
LocalStorage;
GNode;
MutationListener;

AnimateAction;
CollisionAction;
ExportOBJAction;
MechanicalAction;
MechanicalVopAction;
MechanicalVPrintAction;
PrintAction;
PropagateEventAction;
UpdateMappingAction;
VisualAction;
VisualComputeBBoxAction;
XMLPrintAction;

Simulation;
}

/// Automate-based multithread scheduler.
/// (current Components::Thread namespace)
namespace AutomateScheduler
{
Automate;
CPU;
Node;
...
}

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
