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
*/

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

/// Object
/// Specifies how pointer to generic objects and context are handled, as
/// well as the basic functionnalities of all objects (name, fields, ...).
namespace Object
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

} // namespace Framework

/// Implementation of components
namespace Components
{

/// Coordinates and their derivatives
MechanicalObject;

/// Shapes, independently of the coordinates
namespace Topology
{
MeshTopology;
GridTopology;
RegularGridTopology;
TrimmedRegularGridTopology;
}

/// Solve ODE equation (and can call Solver to perform)
namespace ODESolver
{
EulerSolver;
RungeKutta4Solver;
EulerImplicitSolver;
}

/// Numerical solver
namespace NumericalSolver
{
StaticSolver;
CgSolver;
MatrixSolver;

LCP; // this class already exist ! => maybe we should change it to LCPSolver
QPSolver; //hypothetical new class
LagrangeMultiplierSolver; // hypothetical new class
}


/// Mappings
namespace Mapping
{
BarycentricMapping;
IdentityMapping;
RigidMapping;
ImplicitSurfaceMapping;
}

/// Components used to modify the context
namespace Context
{
Gravity;
CoordinateSystem;
}

namespace Deformable
{
MechanicalObject<Vec3Types>; // instantiation of MechanicalObject on Vec3 types
// DiagonalMass<Vec3Types>; // instantiation of DiagonalMass on Vec3 types
// UniformMass<Vec3Types>; // instantiation of UniformMass on Vec3 types

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

/// Weight and inertia
namespace Mass
{
DiagonalMass;
UniformMass;
}

}

/// Components used only for rigid bodies
namespace Rigid
{
MechanicalObject<RigidTypes>; // instantiation of MechanicalObject on rigid types
//      DiagonalMass<RigidTypes>; // instantiation of DiagonalMass on rigid types
//      UniformMass<RigidTypes>; // instantiation of UniformMass on rigid types
RigidConstraintSolver; // hypothetical rigid body constraint solver (christian : ???)

namespace Mass
{
UniformMass;	// renamed RigidMass ?
DiagonalMass;
}

namespace Springs
{
6DSpringForceField;	// 6D spring force field (for haptic for instance)
AngularForceField;  // 1D (angular) for some articulations
}
}

/// Components used only for fluids
namespace Fluid
{
SpatialGridContainer;

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

/// Force-based interactions
namespace InteractionForce
{
BarycentricPenalityContact;
RepulsiveSpringForceField;
ExternalForceField;
PlaneForceField;
}

/// Constraints
namespace Constraint
{
FixedConstraint;
BoxConstraint;
OscillatorConstraint;
LagrangianMultiplierConstraint;
LagrangianMultiplierFixedConstraint;
LagrangianMultiplierAttachConstraint;
LagrangianMultiplierContactConstraint;

UnilateralConstraint; // hypothetical new class
CoulombFrictionConstraint; // hypothetical new class
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

class BaseNode : BaseContext;
class BaseSimulation; // TODO: define from interface of Graph::Simulation
class BaseScheduler; // TODO: define from interface of Graph::ActionScheduler

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
