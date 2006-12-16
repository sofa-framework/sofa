/** An example of Sofa simulation shown using the call stack tree.
It simulates a pair of particles linked by a spring, one particle being fixed.

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

        /** Solve an ODE and move forward in time
        * The ODE solver repeatedly uses the mechanical components (Mass, ForceField, Constraint) to update the state of the mechanical system.
        */
        GNode::execute(animateAction)
        {
            GNode::executeAction(animateAction)
            {
                GNode::doExecuteAction(animateAction)
                {
                    /** Notifies the BehaviorModel and OdeSolver components */
                    AnimateAction::processNodeTopDown(node)
                    {
                        /** When a OdeSolver component is found, it handles its sub-tree.
                        * The traversal of the AnimateAction does not continue through the children. It jumps to the sibling nodes.
                        */
                        AnimateAction::processSolver(node,solver)
                        {
                            /** The ODE solver applies its algorithm
                            * Here, Implicit Euler solved using a filtered conjugate gradient [Baraff&Witkin 98]
                            */
                            CGImplicitSolver::solve(dt)(dt)
                            {
                                /** The right-hand term of the equation is stored in auxiliary state vector b.*/
                                {
                                    /** b = f0 */
                                    OdeSolver::computeForce(b)
                                    {
                                        /** First, vector a is set to 0
                                        */
                                        MechanicalResetForceAction::MechanicalResetForceAction(VecId);
                                        Action::execute(BaseContext); // MechanicalResetForceAction
                                        /** Then, the ForceField components accumulate their contribution
                                         */
                                        MechanicalComputeForceAction::MechanicalComputeForceAction(VecId);
                                        Action::execute(BaseContext)
                                        {
                                            // MechanicalComputeForceAction
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicForceField)* fonction)
                                                        {
                                                            MechanicalComputeForceAction::fwdForceField(GNode, BasicForceField)
                                                            {
                                                                /** Internal forces (weight, springs, FEM, ...)
                                                                */
                                                                ForceField::addForce()
                                                                {
                                                                    UniformMass::addForce(f, x, v);
                                                                }
                                                            }
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField)
                                                            {
                                                                /** Interaction forces (penalties, ...)
                                                                 */
                                                                MechanicalComputeForceAction::fwdForceField(GNode, BasicForceField)
                                                                {
                                                                    StiffSpringForceField::addForce();
                                                                }
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
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                        {
                                                            /** Not complete. Documentation to be continued here
                                                            */
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField) ;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    /** df = K dx
                                    * Compute the force increment corresponding to the current displacement, and store it in vector df
                                    */
                                    OdeSolver::computeDf(df)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField) ;
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
                                        OdeSolver::v_peq( v, a, f)
                                        {
                                            Action::execute(BaseContext)
                                            {
                                                GNode::executeAction(Action)
                                                {
                                                    GNode::doExecuteAction(Action)
                                                    {
                                                        MechanicalAction::processNodeTopDown(GNode)
                                                        {
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
                                    /** b = h(f0 + (h+rs)df/dx v - rd M v)
                                    */
                                    MultiVector::teq(f)
                                    {
                                        OdeSolver::v_teq(v, f)
                                        {
                                            Action::execute(BaseContext)
                                            {
                                                GNode::executeAction(Action)
                                                {
                                                    GNode::doExecuteAction(Action)
                                                    {
                                                        MechanicalAction::processNodeTopDown(GNode)
                                                        {
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
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
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
                                    OdeSolver::v_eq(v, a)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
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
                                /** The solution of the equation system is iteratively refined using the filtered conjugate gradient algorithm */
                                {
                                    OdeSolver::propagateDx(dx)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    OdeSolver::computeDf(df)
                                    {
                                        Action::execute(BaseContext)
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField) ;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
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
                                /** Compute the value of the new positions x and new velocities v */
                                {
                                    // vel = vel + x
                                    MultiVector::peq(a, f)
                                    {
                                        OdeSolver::v_peq(v, a, f)
                                        {
                                            Action::execute(BaseContext)
                                            {
                                                GNode::executeAction(Action)
                                                {
                                                    GNode::doExecuteAction(Action)
                                                    {
                                                        MechanicalAction::processNodeTopDown(GNode)
                                                        {
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
                                    // pos = pos + h vel
                                    MultiVector::peq(a, f)
                                    {
                                        OdeSolver::v_peq(v, a, f)
                                        {
                                            Action::execute(BaseContext)
                                            {
                                                GNode::executeAction(Action)
                                                {
                                                    GNode::doExecuteAction(Action)
                                                    {
                                                        MechanicalAction::processNodeTopDown(GNode)
                                                        {
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
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    /** Copy context values from parent node, and apply local changes */
                                                    node->updateContext();
                                                    /** Set x and v as current positions and velocities */
                                                    MechanicalPropagatePositionAndVelocityAction::fwdMechanicalModel(GNode,BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setX(x);
                                                        MechanicalObject::setV(v);
                                                    }
                                                    /** Filter the positions and velocities to match the constraints */
                                                    for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicConstraint)* fonction)
                                                    {
                                                        MechanicalPropagatePositionAndVelocityAction::fwdConstraint(GNode, BasicConstraint)
                                                        {
                                                            Constraint::projectVelocity();
                                                            Constraint::projectPosition();
                                                        }
                                                    }
                                                }
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

