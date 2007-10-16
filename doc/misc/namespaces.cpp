// items hierarchically listed in alphabetical order, except at top level

Abstract
{
    Base;
    BaseContext;
    BaseNode;
    BaseObject;
    BehaviorModel;
    CollisionElementIterator;
    CollisionModel;
    ContextObject;
    Event;
    FieldBase;
    FieldContainer;
    TCollisionElementIterator;
    VisualModel;
}

Core
{
    Encoding{}

    BasicConstraint;
    BasicForceField;
    BasicMapping;
    BasicMass;
    BasicMechanicalModel;
    BasicTopology;
    Constraint;
    Context;
    ForceField;
    GeometryAlgorithms;
    InteractionForceField;
    MappedModel;
    Mapping;
    Mass;
    MechanicalMapping;
    MechanicalObject;
    MultiVector;
    OdeSolver;
    TopologicalMapping;
    Topology;
    TopologyAlgorithms;
    TopologyChange;
    TopologyContainer;
    TopologyModifier;
    XField;
}


Components
{
    Collision {
        BroadPhaseDetection;
        CollisionGroupManager;
        Contact;
        ContactManager;
        Detection;
        DetectionOutput;
        Pipeline;
    }
    Common{
        // just a selection
        Creator;
        DataField;
        ExtVector;
        Factory;
        FnDispatcher;
        Frame;
        Image;
        Mesh;
        Quater;
        RigidMass;
        SetDirectory;
        SolidTypes;
        StdVectorTypes;
        TypeInfo;
        Vec;
    }
    ContinuousIntersections{}
    DiscreteIntersections{}
    GL{}
    Graph{ // just a selection
        Action;
        ActionScheduler;
        AnimateAction;
        CollisionAction;
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
        XMLPrintAction;
    }
    Thread{}
    XML{}

    // Many classes ommited. Tried to select one of each category
    BarycentricMapping;
    BruteForceDetection;
    CgImplicitSolver;
    CollisionGroupManagerSofa;
    ContactManagerSofa;
    ContinuousIntersection;
    DiagonalMass;
    EmptyClass;
    ExternalForceField;
    FixedConstraint;
    Gravity;
    GridTopology;
    KeypressedEvent;
    MechanicalObjectLoader;
    PipelineSofa;
    SolverMerger;
    VisualComputeBBoxAction; // error ?
}

