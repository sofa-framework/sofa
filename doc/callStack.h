/*
Sofa call stack tree - How the different concepts are piled up ?
This file tends to respect the C language syntax and uses "{" and "}" to build the tree nodes.
*/

GUI::QT::QtViewer::step()
{
    Simulation::animate(root,dt)
    {
        GNode::execute<CollisionAction>()
        {
            GNode::executeAction(CollisionAction)
            {
                GNode::doExecuteAction(CollisionAction)
                {
                    CollisionAction::processNodeTopDown(GNode);
                }
            }
        }
        GNode::execute(animateAction)
        {
            GNode::executeAction(animateAction)
            {
                GNode::doExecuteAction(animateAction)
                {
                    AnimateAction::processNodeTopDown(node)
                    {
                        AnimateAction::processSolver(node,solver)
                        {
                            CGImplicitSolver::solve(dt)(dt)
                            {
                                //     CGImplicitSolver init
                                {
                                    // b = f0
                                    OdeSolver::computeForce(a)
                                    {
                                        MechanicalResetForceAction::MechanicalResetForceAction(VecId);
                                        Action::execute(BaseContext); // MechanicalResetForceAction
                                        MechanicalComputeForceAction::MechanicalComputeForceAction(VecId);
                                        Action::execute(BaseContext)  // MechanicalComputeForceAction
                                        {
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
                                                                ForceField::addForce()
                                                                {
                                                                    UniformMass::addForce(f, x, v);
                                                                }
                                                            }
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField)
                                                            {
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
                                    // dx = v
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
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField) ;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // f = df/dx v
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
                                    // b = f0 + (h+rs)df/dx v
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
                                    // b = h(f0 + (h+rs)df/dx v - rd M v)
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
                                    // b is projected to the constrained space
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
                                // CGImplicitSolver iterations
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
                                // Apply the solution
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
                                                    node->updateContext();
                                                    //mechanicalModel
                                                    MechanicalPropagatePositionAndVelocityAction::fwdMechanicalModel(GNode,BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setX(v);
                                                        MechanicalObject::setV(v);
                                                    }
                                                    //constraint
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
                    Action::processNodeBottomUp(GNode);
                }
            }
        }
        GNode::execute<UpdateMappingAction>()
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
                                UniformMass<::update();
                            }
                        }
                    }
                }
            }
        }
    }
}
}
