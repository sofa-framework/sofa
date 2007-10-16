// items hierarchically listed in alphabetical order, except at top level

Abstract  // No DataType. Address no other Sofa namespace.
{
    BasicConstraint;
    BasicForceField;
    BasicMapping;
    BasicMass;
    BasicMechanicalModel;
    BasicTopology;
    BehaviorModel;
    Event;
    InteractionForceField;
    KeypressedEvent;
    Mapping;
    Mass;
    OdeSolver;
    TopologicalMapping;
    Topology;
    VisualModel;
}

Components  // Address only ::Abstract
{
    Collision{ // more structure needed here ?
        ContactManagerSofa;
        ContinuousIntersections{}
        DiscreteIntersections{}

        BroadPhaseDetection;
        BruteForceDetection;
        CollisionElementIterator;
        CollisionGroupManager;
        CollisionModel;
        Contact;
        ContactManager;
        ContinuousIntersection;
        Detection;
        DetectionOutput;
        Pipeline;
        PipelineSofa;
        TCollisionElementIterator;
    }
    GL{}
    Topology{
        TopologyAlgorithms;
        TopologyChange;
        TopologyContainer;
        TopologyModifier;
    }

    BarycentricMapping;
    CgImplicitSolver;
    Constraint;
    Context;
    DiagonalMass;
    EmptyClass;
    ExternalForceField;
    FixedConstraint;
    ForceField;
    Gravity;
    GridTopology;
    MechanicalMapping;
    MechanicalObject;
    MechanicalObjectLoader;
}

// Data structure and scheduling. Address only ::Abstract
Graph
{
    Encoding{}
    Thread{} // where else ?
    Tree{ // run-time model for a tree structure
        XML{}

        Action;
        ActionScheduler;
        AnimateAction;
        CollisionAction;
        CollisionGroupManagerSofa;
        ExportOBJAction;
        GNode;
        LocalStorage;
        MechanicalAction;
        MechanicalVopAction;
        MechanicalVPrintAction;
        MutationListener;
        PrintAction;
        PropagateEventAction;
        Simulation;
        UpdateMappingAction;
        VisualAction;
        VisualComputeBBoxAction;
        XMLPrintAction;
    }
    // in the future, why not a run-time model for a network structure here.

    // valid in any data structure (tree,network) :
    Base;
    BaseContext;
    BaseNode;
    BaseObject;
    ContextObject;
    Creator;
    DataField;
    Factory;
    FnDispatcher;
    FieldBase;
    FieldContainer;
    MultiVector;
    SetDirectory;
    SolverMerger;
    TypeInfo;
    XField;
}

Default
{
    ExtVector;
    Frame;
    Image;
    Mesh;
    Quater;
    RigidMass;
    SolidTypes;
    StdVectorTypes;
    Vec;
}

MechanicalObject2d;

MechanicalObject2f;
MechanicalObject3d;
MechanicalObject3f;


