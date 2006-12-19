/** An example of Sofa simulation shown using the call stack tree.
It simulates a pair of particles linked by a spring, one particle being fixed.
Here is the code corresponding to the scene graph :
        // The graph root node
		groot = new Sofa::Components::Graph::GNode;
		groot->setName( "root" );

		// One solver for all the graph
		Sofa::Components::CGImplicitSolver* solver = new Sofa::Components::CGImplicitSolver;
		solver->f_printLog.setValue(false);
		groot->addObject(solver);

		// Set gravity for all the graph
		Sofa::Components::Gravity* gravity =  new Sofa::Components::Gravity;
		gravity->f_gravity.setValue( Vec3(0,-100,0) );
		groot->addObject(gravity);

		// Spring degrees of freedom
		Sofa::Core::MechanicalObject<MyTypes>* DOF = new Sofa::Core::MechanicalObject<MyTypes>;
		groot->addObject(DOF);
		DOF->resize(2);
		DOF->setName("DOF");
		MyTypes::VecCoord& x = *DOF->getX();
		x[0] = Vec3(0,0,0);
		x[1] = Vec3(0,-10,0);

		// Spring mass
		Sofa::Components::UniformMass<MyTypes,double>* mass = new Sofa::Components::UniformMass<MyTypes,double>(DOF);
        groot->addObject(mass);
        mass->setMass( 1 );

		// Spring constraints
		Sofa::Components::FixedConstraint<MyTypes>* constraints = new Sofa::Components::FixedConstraint<MyTypes>(DOF);
		groot->addObject(constraints);
		constraints->setName("constraints");
		constraints->addConstraint(0);

		// Spring force field
		Sofa::Components::StiffSpringForceField<MyTypes>* spring = new Sofa::Components::StiffSpringForceField<MyTypes>(DOF);
		spring->addSpring(0, 1, 10.f, 1.f, 5);
		groot->addObject(spring);

This file follows the C language syntax and is better seen in a folding C-source editor with comment highlights.
*/

/** User application
*/
GUI::QT::QtViewer::step()
{
    /** Move the ::Sofa:: scene a step forward
    * The data structure is hierarchically traversed by *actions* which trigger specific methods of the *components*.
    * The components are not aware of the data structure. We use an extended tree (technically, a directed acyclic graph), because it eases the generic implementation of efficient simulation approaches.
    * Other structures, such as networks, could be of interrest.
    */
    Simulation::animate(root,dt)
    {
        /** Process the collisions. Set up penalty-based contacts, modify topology, ...
        */
        GNode::execute<CollisionAction>()
        {
            GNode::executeAction(CollisionAction)
            {
                GNode::doExecuteAction(CollisionAction)
                {
                    CollisionAction::processNodeTopDown(GNode)
                    {
                        // ...
                    }
                }
            }
        }

        /** Move forward in time */
        GNode::execute(animateAction)
        {
            GNode::executeAction(animateAction)
            {
                GNode::doExecuteAction(animateAction)
                {
                    /** Notifies the BehaviorModel and OdeSolver components.
                    * The BehavioModel components are stand-alone.
                    * They can be used to model objects designed separately from Sofa.
                    * They are processed top-down.
                    *
                    * When a OdeSolver component is found, it handles its sub-tree.
                    * The traversal of the AnimateAction does not continue through the children,
                    * it jumps to the sibling nodes.
                    */
                    AnimateAction::processNodeTopDown(node)
                    {
                        /** Solve an ODE and move forward in time.
                        * The ODE solver repeatedly uses the mechanical components (Mass, ForceField, Constraint) to update the state of the mechanical system, using actions.
                        */
                        AnimateAction::processSolver(node,solver)
                        {
                            /** MechanicalBeginIntegrationAction
                            * All necessary tasks before integration, such as updating the external forces
                            */
                            GNode::execute(MechanicalBeginIntegrationAction)
                            {
                                GNode::executeAction(Action)
                                {
                                    GNode::doExecuteAction(Action)
                                    {
                                        MechanicalAction::processNodeTopDown(GNode)
                                        {
                                            MechanicalBeginIntegrationAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                            {
                                                MechanicalObject::beginIntegration(dt);
                                            }
                                            Action::for_each(MechanicalAction, GNode, const Sequence, Action::Result (GNode,InteractionForceField)* fonction)
                                            {
                                                MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                            }
                                            Action::for_each(MechanicalAction, GNode, const Sequence, Action::Result (GNode,BasicConstraint)* fonction)
                                            {
                                                MechanicalBeginIntegrationAction::fwdConstraint(GNode,BasicConstraint)
                                                {
                                                    BasicConstraint::getDOFs();
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            /** The ODE solver applies its algorithm
                            * Here, Implicit Euler solved using a filtered conjugate gradient [Baraff&Witkin 98]
                            */
                            CGImplicitSolver::solve(dt)
                            {
                                /** CGImplicitSolver init */

                                /** Reserve auxiliary vectors.
                                The solver allocates the corresponding DOFs (i.e. vectors (f,x, v...) in MechanicalObject)
                                */
                                MultiVector::MultiVector(OdeSolver, VecType)
                                {
                                    OdeSolver::v_alloc(VecType)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalVAllocAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::vAlloc(VecId );
                                                        }
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Result (GNode, InteractionForceField )* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                /** b = f0
                                */
                                OdeSolver::computeForce(b)
                                {
                                    /** First, vector a is set to 0
                                    */
                                    MechanicalResetForceAction::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalResetForceAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setF(VecId);
                                                        MechanicalObject::resetForce();
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, const Sequence & list, Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, const Sequence & list, Result (GNode, BasicConstaint)* fonction)
                                                    {
                                                        MechanicalResetForceAction::fwdConstraint(GNode, BasicConstraint);
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    /** Then, the ForceField components accumulate their contribution
                                    */
                                    MechanicalComputeForceAction::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalComputeForceAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setF(VecId);
                                                        MechanicalObject::accumulateForce(); // external forces
                                                    }
                                                    /** BasicForceField */
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicForceField)* fonction)
                                                    {
                                                        /** UniformMass */
                                                        MechanicalComputeForceAction::fwdForceField(GNode, BasicForceField)
                                                        {
                                                            ForceField::addForce()
                                                            {
                                                                /** Get the state vectors using getContext()->getMechanicalModel()->getF(),
                                                                getContext()->getMechanicalModel()->getX(),
                                                                getContext()->getMechanicalModel()->getV()
                                                                */
                                                                UniformMass::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
                                                                {
                                                                    // weight
                                                                    const double* g = this->getContext()->getLocalGravity().ptr();
                                                                    Deriv theGravity;
                                                                    DataTypes::set
                                                                    ( theGravity, g[0], g[1], g[2]);
                                                                    Deriv mg = theGravity * mass;

                                                                    // velocity-based stuff
                                                                    Core::Context::SpatialVector vframe = getContext()->getVelocityInWorld();
                                                                    Core::Context::Vec3 aframe = getContext()->getVelocityBasedLinearAccelerationInWorld() ;

                                                                    // project back to local frame
                                                                    vframe = getContext()->getPositionInWorld() / vframe;
                                                                    aframe = getContext()->getPositionInWorld().backProjectVector( aframe );

                                                                    // add weight and inertia force
                                                                    for (unsigned int i=0; i<f.size(); i++)
                                                                    {
                                                                        f[i] += mg + Core::inertiaForce(vframe,aframe,mass,x[i],v[i]);
                                                                    }

                                                                }
                                                            }
                                                        }
                                                    }
                                                    /** InteractionForceField */
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        /** StiffSpringForceField */
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField)
                                                        {
                                                            MechanicalComputeForceAction::fwdForceField(GNode, BasicForceField)
                                                            {
                                                                StiffSpringForceField::addForce()
                                                                {
                                                                    assert(this->object1);
                                                                    assert(this->object2);
                                                                    this->dfdx.resize(this->springs.size());
                                                                    VecDeriv& f1 = *this->object1->getF();
                                                                    const VecCoord& p1 = *this->object1->getX();
                                                                    const VecDeriv& v1 = *this->object1->getV();
                                                                    VecDeriv& f2 = *this->object2->getF();
                                                                    const VecCoord& p2 = *this->object2->getX();
                                                                    const VecDeriv& v2 = *this->object2->getV();
                                                                    f1.resize(p1.size());
                                                                    f2.resize(p2.size());
                                                                    m_potentialEnergy = 0;
                                                                    for (unsigned int i=0; i<this->springs.size(); i++)
                                                                    {
                                                                        this->addSpringForce(m_potentialEnergy,f1,p1,v1,f2,p2,v2, i, this->springs[i]);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    /** BasicConstraint */
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicConstraint)* fonction)
                                                    {
                                                        MechanicalComputeForceAction::fwdConstraint(GNode, BasicConstraint)
                                                        {
                                                            /** Nothing is done in this example ! */
                                                            BasicConstraint::getDOFs();
                                                        }

                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                /** dx = v
                                * We need to compute hKv.
                                * Given a displacement, the ForceField components are able to compute the corresponding force variations
                                * This action makes v the current displacement
                                */
                                OdeSolver::propagateDx(v)
                                {
                                    Action::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    /// Make the Dx index refer to the given one (v)
                                                    MechanicalPropagateDxAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setDx(VecId);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField) ;
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicConstraint)* fonction)
                                                    {
                                                        MechanicalPropagateDxAction::fwdConstraint(GNode, BasicConstraint);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                /** f = K dx
                                * Compute the force increment corresponding to the current displacement, and store it in vector df
                                */
                                OdeSolver::computeDf(df)
                                {
                                    MechanicalResetForceAction::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    /** MechanicalResetForceAction */
                                                    MechanicalResetForceAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setF(VecId);
                                                        MechanicalObject::resetForce();
                                                        MechanicalObject::setF(VecId);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicConstraint)* fonction)
                                                    {
                                                        MechanicalResetForceAction::fwdConstraint(GNode, BasicConstraint);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    /** Do nothing */
                                    OdeSolver::finish();
                                    MechanicalComputeDfAction::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    /** MechanicalComputeDfAction */
                                                    MechanicalComputeDfAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setF(VecId v);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField)
                                                        {
                                                            MechanicalComputeDfAction::fwdForceField(GNode, BasicForceField)
                                                            {
                                                                StiffSpringForceField::addDForce();
                                                            }
                                                        }
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicConstraint)* fonction)
                                                    {
                                                        MechanicalComputeDfAction::fwdConstraint(GNode, BasicConstraint);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                /** b = f0 + (h+rs)df/dx v
                                */
                                MultiVector::peq(a, f)
                                {
                                    OdeSolver::v_peq( VecId v, VecId a, double f)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalVOpAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::vOp(VecId v, VecId a, VecId b, double f);
                                                        }
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                MultiVector::clear()
                                {
                                    /** v=0  (?????)
                                    */
                                    OdeSolver::v_clear(v)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalVOpAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::vOp(VecId v, VecId a, VecId b, double f);
                                                        }
                                                        Action::for_each(MechanicalAction, GNode * node=0x01cc3f28, const Sequence & list, Action::Result (GNode,InteractionForceField)* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                OdeSolver::addMdx(VecId res, VecId dx)
                                {
                                    Action::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalAddMDxAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setF(VecId v);
                                                        MechanicalObject::setDx(VecId v);
                                                    }
                                                    MechanicalAddMDxAction::fwdMass(GNode, BasicMass)
                                                    {
                                                        Mass::addMDx()
                                                        {
                                                            /** Get the state vectors using
                                                            *  Mass->MechanicalModel->getF(),
                                                            *  Mass->MechanicalModel->getDx()
                                                            */
                                                            UniformMass::addMDx(VecDeriv& f, const VecDeriv& dx)
                                                            {
                                                                for (unsigned int i=0; i<dx.size(); i++)
                                                                {
                                                                    f[i] += dx[i] * mass;
                                                                }
                                                            }
                                                        }
                                                    }

                                                }
                                            }
                                        }
                                    }
                                }

                                MultiVector::peq(VecId a, double f)
                                {
                                    OdeSolver::v_peq(VecId v, VecId a, double f)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalVOpAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::vOp(VecId v, VecId a, VecId b, double f);
                                                        }
                                                        Action::for_each(MechanicalAction, GNode * node=0x01cc3f28, const Sequence & list, Action::Result (GNode,InteractionForceField)* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                /** b = h(f0 + (h+rs)df/dx v - rd M v)
                                */
                                MultiVector::teq(f)
                                {
                                    OdeSolver::v_teq(VecId v, double f)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalVOpAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::vOp(VecId v, VecId a, VecId b, double f);
                                                        }
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                        }
                                                    }

                                                }
                                            }
                                        }
                                    }
                                }
                                /** b is projected to the constrained space
                                */
                                OdeSolver::projectResponse(dx)
                                {
                                    Action::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalApplyConstraintsAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setDx(VecId v);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                MultiVector::dot(VecId a)
                                {
                                    /** Appel OdeSolver::v_dot(){
                                    *           Action::execute(){
                                    *               ...
                                    *                   MechanicalVDotAction::fwdMechanicalModel(){
                                    *                       MechanicalObject::vDot();
                                    *                   }
                                    *                   Action::for_each(){
                                    *                       MechanicalAction::fwdInteractionForceField();
                                    *                   }
                                    *           }
                                    *   }
                                    */
                                }
                                OdeSolver::v_clear(v)
                                {
                                    Action::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalVOpAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::vOp(VecId v, VecId a, VecId b, double f);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                    }

                                                }
                                            }
                                        }
                                    }
                                }
                                /** v=a
                                */
                                OdeSolver::v_eq(VecId v, VecId a)
                                {
                                    Action::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalVOpAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::vOp(VecId v, VecId a, VecId b, double f);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }





                                /********************************/
                                /** CGImplicitSolver iterations */
                                /********************************/
                                /**... calcul vectoriel... */
                                /** CGImplicitSolver iterations */
                                OdeSolver::propagateDx(VecId dx)
                                {
                                    Action::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalPropagateDxAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setDx(VecId v);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicConstraint)* fonction)
                                                    {
                                                        MechanicalPropagateDxAction::fwdConstraint(GNode, BasicConstraint);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                /** CGImplicitSolver iterations */
                                OdeSolver::computeDf(VecId df)
                                {
                                    MechanicalResetForceAction::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    /** MechanicalResetForceAction */
                                                    MechanicalResetForceAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setF(VecId v);
                                                        MechanicalObject::resetForce();
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicConstraint)* fonction)
                                                    {
                                                        MechanicalResetForceAction::fwdConstraint(GNode, BasicConstraint);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    /** Do nothing */
                                    OdeSolver::finish();
                                    MechanicalComputeDfAction::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {

                                                    /** MechanicalComputeDfAction */
                                                    MechanicalComputeDfAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setF(VecId v);
                                                        BasicMechanicalModel::accumulateDf()
                                                        {
                                                            //empty
                                                        }

                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicForceField)* fonction)
                                                    {
                                                        MechanicalComputeDfAction::fwdForceField(GNode, BasicForceField);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField)
                                                        {
                                                            MechanicalComputeDfAction::fwdForceField(GNode,BasicForceField)
                                                            {
                                                                StiffSpringForceField::addDForce();
                                                            }
                                                        }
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicConstraint)* fonction)
                                                    {
                                                        MechanicalComputeDfAction::fwdConstraint(GNode ,BasicConstraint);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                /** ... calcul vectoriel... */
                                OdeSolver::projectResponse(VecId dx)
                                {
                                    Action::execute(BaseContext)
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalApplyConstraintsAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setDx(VecId v);
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                /** ... calcul vectoriel... */
                                /** Compute the value of the new positions x and new velocities v
                                *  Apply the solution
                                *  pos = pos + h vel
                                */
                                MultiVector::peq(a, f)
                                {
                                    OdeSolver::v_peq(VecId v, VecId a, double f)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalVOpAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::vOp(VecId v, VecId a, VecId b, double f);
                                                        }
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }



                                /** ... calcul vectoriel... */

                                /****************/
                                /** Free memory */
                                /****************/
                                MultiVector::~MultiVector()
                                {
                                    OdeSolver::v_free(VecId v)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalVFreeAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::vFree(VecId v);
                                                        }
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            /** Set x and v as current positions and velocities */
                            OdeSolver::propagatePositionAndVelocity(t, x, v)
                            {
                                Action::execute(BaseContext)
                                {
                                    GNode::executeAction(Action)
                                    {
                                        GNode::doExecuteAction(Action)
                                        {
                                            MechanicalPropagatePositionAndVelocityAction::processNodeTopDown(GNode)
                                            {
                                                /** Copy context values from parent node, and apply local changes */
                                                node->updateContext()
                                                {
                                                    Gravity::apply()
                                                    {
                                                        getContext()->setGravityInWorld( f_gravity.getValue() );
                                                    }
                                                }
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalPropagatePositionAndVelocityAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setX(VecId v);
                                                        MechanicalObject::setV(VecId v);
                                                    }
                                                    for_each(MechanicalAction, GNode, const Sequence & list, Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                    }
                                                    for_each(MechanicalAction, GNode, const Sequence & list, Result (GNode, BasicConstraint)* fonction)
                                                    {
                                                        /** Filter the positions and velocities to match the constraints */
                                                        MechanicalPropagatePositionAndVelocityAction::fwdConstraint(GNode, BasicConstraint)
                                                        {
                                                            // for example constraint a position in a plane or a fixed dot
                                                            Constraint::projectPosition();
                                                            // see also  Constraint::projectVelocity();
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            /**  MechanicalEndIntegrationAction
                            *  Ending the action : execute(MechanicalEndIntegrationAction(getDt()))
                            */
                            GNode::execute(MechanicalEndIntegrationAction)
                            {
                                GNode::executeAction(Action)
                                {
                                    GNode::doExecuteAction(Action)
                                    {
                                        MechanicalAction::processNodeTopDown(GNode)
                                        {
                                            MechanicalEndIntegrationAction::fwdMechanicalModel(GNode,BasicMechanicalModel)
                                            {
                                                Core::MechanicalObject::endIntegration(double);
                                            }
                                            Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                            {
                                                MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    /** nothing to do bottom-up */
                    Action::processNodeBottomUp(GNode)
                    {}
                }
            }
        }

        /** Update the other aspects: visual, haptics, ...
        */
        GNode::execute<UpdateMappingAction>()
        {}

        /** redraw
        */
        GNode::execute(VisualUpdateAction)
        {
            GNode::executeAction(VisualUpdateAction)
            {
                GNode::doExecuteAction(Action)
                {
                    VisualAction::processNodeTopDown(GNode)
                    {
                        Action::for_each(VisualAction, GNode, Sequence<VisualModel>, void (GNode, VisualModel)* fonction)
                        {
                            VisualUpdateAction::processVisualModel(GNode, VisualModel)
                            {
                                /** OpenGL rendering of the viewable components
                                */
                                VisualModel::draw();
                            }
                        }
                    }
                }
            }
        }
    }
}
