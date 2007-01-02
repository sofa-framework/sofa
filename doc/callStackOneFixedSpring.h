/** An example of Sofa simulation shown using the call stack tree.
This file follows the C language syntax and is better seen in a folding C-source editor with comment highlights.

It simulates a pair of particles linked by a spring, one particle being fixed (see ../src/CallStack/oneFixedSpring/).
*/


Simulation::init(GNode)
{
    GNode::execute<InitAction>()
    {
        GNode::executeAction(InitAction)
        {
            GNode::doExecuteAction(InitAction)
            {
                InitAction::processNodeTopDown(GNode)
                {
                    GNode::initialize()
                    {
                        GNode::updateContext()
                        {
                            if( getParent() != NULL )
                            {
                                copyContext(*parent);
                            }
                            // Apply local modifications to the context
                            if (getLogTime()) // False
                            {
                                for( unsigned i=0; i<contextObject.size(); ++i )
                                {
                                    contextObject[i]->init();
                                    contextObject[i]->apply();
                                }
                            }
                            else // Pass around here
                            {
                                // In this example, contextObject is only composed with gravity
                                for( unsigned i=0; i<contextObject.size(); ++i )
                                {
                                    // BaseObject::init();
                                    contextObject[i]->init();
                                    // Gravity::apply();
                                    contextObject[i]->apply();
                                }
                            }
                        }
                    }

                }
                for(GNode::Sequence<BaseObject>::iterator i=node->object.begin(),
                    iend=node->object.end();
                    i!=iend;
                    i++ )
                {
                    (*i)->init() // The different calls "init()" while the iterations
                    {
                        // 1st and 2nd loop;
                        BaseObject::init();
                        //3th loop
                        UniformMass::init()
                        {
                            ForceField::init()
                            {
                                BaseObject::init();
                            }
                        }
                        //4th loop
                        FixedConstraint::init()
                        {
                            Constraint::init()
                            {
                                BaseObject::init();
                                // Init its MechanicalModel
                                mmodel = dynamic_cast< MechanicalModel<DataTypes>* >(getContext()->getMechanicalModel());
                            }
                        }
                        //4th loop
                        StiffSpringForceField::init()
                        {
                            SpringForceField::init()
                            {
                                BaseObject::init();
                            }
                        }
                    }
                }
            }
        }
    }
}

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
                                                MechanicalObject::beginIntegration(dt)
                                                {
                                                    this->f = this->internalForces;
                                                }
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
                                CGImplicitSolver* group = this;

                                /** Reserve auxiliary vectors.
                                * The solver allocates the corresponding DOFs (i.e. vectors (f,x, v...) in MechanicalObject)
                                */
                                MultiVector pos(this, VecId::position())
                                {
                                    OdeSolver::v_alloc(VecType t)
                                    {
                                        VecId v(t, vectors[t].alloc());
                                        MechanicalVAllocAction(v).execute( getContext() )
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalVAllocAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::vAlloc(VecId v)
                                                            {
                                                                if (v.type == V_COORD && v.index >= V_FIRST_DYNAMIC_INDEX)
                                                                {
                                                                    VecCoord* vec = getVecCoord(v.index);
                                                                    vec->resize(vsize);
                                                                }
                                                                else if (v.type == V_DERIV && v.index >= V_FIRST_DYNAMIC_INDEX)
                                                                {
                                                                    VecDeriv* vec = getVecDeriv(v.index);
                                                                    vec->resize(vsize);
                                                                }
                                                                else
                                                                {
                                                                    std::cerr << "Invalid alloc operation ("<<v<<")\n";
                                                                    return;
                                                                }
                                                                //vOp(v); // clear vector
                                                            }
                                                        }
                                                        Action::for_each(MechanicalAction, GNode, Sequence, Result (GNode, InteractionForceField )* fonction)
                                                        {
                                                            MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        return v;
                                    }
                                }
                                MultiVector vel(this, VecId::velocity());
                                MultiVector f(this, VecId::force());
                                MultiVector b(this, V_DERIV);
                                MultiVector p(this, V_DERIV);
                                MultiVector q(this, V_DERIV);
                                MultiVector q2(this, V_DERIV);
                                MultiVector r(this, V_DERIV);
                                MultiVector x(this, V_DERIV);

                                double h = dt;
                                bool printLog = f_printLog.getValue();

                                // compute the right-hand term of the equation system
                                /** b = f0
                                */
                                computeForce(b)
                                {
                                    /** First, vector a is set to 0
                                    */
                                    MechanicalResetForceAction(b).execute( getContext() )
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalResetForceAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setF(VecId v)
                                                        {
                                                            if (v.type == V_DERIV)
                                                            {
                                                                this->v = getVecDeriv(v.index);
                                                            }
                                                            else
                                                            {
                                                                std::cerr << "Invalid setV operation ("<<v<<")\n";
                                                            }
                                                        }
                                                        MechanicalObject::resetForce()
                                                        {
                                                            VecDeriv& f= *getF()  { return f;  };
                                                            for( unsigned i=0; i<f.size(); ++i )
                                                                f[i] = Deriv();
                                                        }
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

                                    /** Do nothing */
                                    OdeSolver::finish();
                                    /** Then, the ForceField components accumulate their contribution
                                    */
                                    MechanicalComputeForceAction(b).execute( getContext() )
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalComputeForceAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setF(VecId v)
                                                        {
                                                            if (v.type == V_DERIV)
                                                            {
                                                                this->f = getVecDeriv(v.index);
                                                            }
                                                            else
                                                            {
                                                                std::cerr << "Invalid setF operation ("<<v<<")\n";
                                                            }
                                                        }
                                                        // external forces
                                                        MechanicalObject::accumulateForce()
                                                        {
                                                            if (!this->externalForces->empty())
                                                            {
                                                                for (unsigned int i=0; i < this->externalForces->size(); i++)
                                                                    (*this->f)[i] += (*this->externalForces)[i];
                                                            }
                                                        }
                                                    }
                                                    /** BasicForceField */
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, BasicForceField)* fonction)
                                                    {
                                                        /** UniformMass */
                                                        MechanicalComputeForceAction::fwdForceField(GNode, BasicForceField)
                                                        {
                                                            ForceField::addForce()
                                                            {
                                                                /** Get the state vectors using getContext()->getMechanicalModel()->getF()  { return f;  },
                                                                getContext()->getMechanicalModel()->getX()  { f_X->beginEdit(); return x;  },
                                                                getContext()->getMechanicalModel()->getV()  { f_V->beginEdit(); return v;  }
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
                                                                    VecDeriv& f1 = *this->object1->getF()  { return f;  };
                                                                    const VecCoord& p1 = *this->object1->getX()  { f_X->beginEdit(); return x;  };
                                                                    const VecDeriv& v1 = *this->object1->getV()  { f_V->beginEdit(); return v;  };
                                                                    VecDeriv& f2 = *this->object2->getF()  { return f;  };
                                                                    const VecCoord& p2 = *this->object2->getX()  { f_X->beginEdit(); return x;  };
                                                                    const VecDeriv& v2 = *this->object2->getV()  { f_V->beginEdit(); return v;  };
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
                                propagateDx(vel)
                                {
                                    MechanicalPropagateDxAction(vel).execute( getContext() )
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
                                                        MechanicalObject::setDx(VecId v)
                                                        {
                                                            if (v.type == V_DERIV)
                                                            {
                                                                this->dx = getVecDeriv(v.index);
                                                            }
                                                            else
                                                            {
                                                                std::cerr << "Invalid setDx operation ("<<v<<")\n";
                                                            }
                                                        }
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
                                /** f = df/dx v
                                * Compute the force increment corresponding to the current displacement, and store it in vector df
                                */
                                computeDf(f)
                                {
                                    MechanicalResetForceAction(f).execute( getContext() )
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
                                                        MechanicalObject::setF(VecId v)
                                                        {
                                                            if (v.type == V_DERIV)
                                                            {
                                                                this->f = getVecDeriv(v.index);
                                                            }
                                                            else
                                                            {
                                                                std::cerr << "Invalid setF operation ("<<v<<")\n";
                                                            }
                                                        }
                                                        MechanicalObject::resetForce()
                                                        {
                                                            VecDeriv& f= *getF()  { return f;  };
                                                            for( unsigned i=0; i<f.size(); ++i )
                                                                f[i] = Deriv();
                                                        }
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
                                    MechanicalComputeDfAction(f).execute( getContext() )
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
                                                        MechanicalObject::setF(VecId v)
                                                        {
                                                            if (v.type == V_DERIV)
                                                            {
                                                                this->f = getVecDeriv(v.index);
                                                            }
                                                            else
                                                            {
                                                                std::cerr << "Invalid setF operation ("<<v<<")\n";
                                                            }
                                                        }
                                                    }
                                                    Action::for_each(MechanicalAction, GNode, Sequence, Action::Result (GNode, InteractionForceField)* fonction)
                                                    {
                                                        MechanicalAction::fwdInteractionForceField(GNode, InteractionForceField)
                                                        {
                                                            MechanicalComputeDfAction::fwdForceField(GNode, BasicForceField)
                                                            {
                                                                StiffSpringForceField::addDForce()
                                                                {
                                                                    VecDeriv& f1  = *this->object1->getF()  { return f;  };
                                                                    const VecCoord& p1 = *this->object1->getX()  { f_X->beginEdit(); return x;  };
                                                                    const VecDeriv& dx1 = *this->object1->getDx() { return dx; }
                                                                    VecDeriv& f2  = *this->object2->getF()  { return f;  };
                                                                    const VecCoord& p2 = *this->object2->getX()  { f_X->beginEdit(); return x;  };
                                                                    const VecDeriv& dx2 = *this->object2->getDx() { return dx; }
                                                                    f1.resize(dx1.size());
                                                                    f2.resize(dx2.size());
                                                                    //cerr<<"StiffSpringForceField<DataTypes>::addDForce, dx1 = "<<dx1<<endl;
                                                                    //cerr<<"StiffSpringForceField<DataTypes>::addDForce, df1 before = "<<f1<<endl;
                                                                    for (unsigned int i=0; i<this->springs.size(); i++)
                                                                    {
                                                                        this->addSpringDForce(f1,p1,dx1,f2,p2,dx2, i, this->springs[i]);
                                                                    }
                                                                    //cerr<<"StiffSpringForceField<DataTypes>::addDForce, df1 = "<<f1<<endl;
                                                                    //cerr<<"StiffSpringForceField<DataTypes>::addDForce, df2 = "<<f2<<endl;
                                                                }
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
                                b.peq(f,h+f_rayleighStiffness.getValue())
                                {
                                    OdeSolver::v_peq( VecId v, VecId a, double f)
                                    {
                                        MechanicalVOpAction(v,v,a,f).execute( getContext() )
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalVOpAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::vOp(VecId v, VecId a, VecId b, double f)
                                                            {
                                                                if(v.isNull())
                                                                {
                                                                    // ERROR
                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                    return;
                                                                }
                                                                if (a.isNull())
                                                                {
                                                                    if (b.isNull())
                                                                    {
                                                                        // v = 0
                                                                        if (v.type == V_COORD)
                                                                        {
                                                                            VecCoord* vv = getVecCoord(v.index);
                                                                            vv->resize(this->vsize);
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = Coord();
                                                                        }
                                                                        else
                                                                        {
                                                                            VecDeriv* vv = getVecDeriv(v.index);
                                                                            vv->resize(this->vsize);
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = Deriv();
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        if (b.type != v.type)
                                                                        {
                                                                            // ERROR
                                                                            std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                            return;
                                                                        }
                                                                        if (v == b)
                                                                        {
                                                                            // v *= f
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] *= (Real)f;
                                                                            }
                                                                            else
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] *= (Real)f;
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            // v = b*f
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                VecCoord* vb = getVecCoord(b.index);
                                                                                vv->resize(vb->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = (*vb)[i] * (Real)f;
                                                                            }
                                                                            else
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(vb->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = (*vb)[i] * (Real)f;
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                                else
                                                                {
                                                                    if (a.type != v.type)
                                                                    {
                                                                        // ERROR
                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                        return;
                                                                    }
                                                                    if (b.isNull())
                                                                    {
                                                                        // v = a
                                                                        if (v.type == V_COORD)
                                                                        {
                                                                            VecCoord* vv = getVecCoord(v.index);
                                                                            VecCoord* va = getVecCoord(a.index);
                                                                            vv->resize(va->size());
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = (*va)[i];
                                                                        }
                                                                        else
                                                                        {
                                                                            VecDeriv* vv = getVecDeriv(v.index);
                                                                            VecDeriv* va = getVecDeriv(a.index);
                                                                            vv->resize(va->size());
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = (*va)[i];
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        if (v == a)
                                                                        {
                                                                            if (f==1.0)
                                                                            {
                                                                                // v += b
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // v += b*f
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            if (f==1.0)
                                                                            {
                                                                                // v = a+b
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    VecCoord* va = getVecCoord(a.index);
                                                                                    vv->resize(va->size());
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* va = getVecDeriv(a.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(va->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // v = a+b*f
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    VecCoord* va = getVecCoord(a.index);
                                                                                    vv->resize(va->size());
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* va = getVecDeriv(a.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(va->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
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


                                if (f_rayleighMass.getValue() != 0.0)
                                {
                                    f.clear()
                                    {
                                        /** v=0  (?????)
                                        */
                                        OdeSolver::v_clear(v)
                                        {
                                            MechanicalVOpAction(v).execute( getContext() )
                                            {
                                                GNode::executeAction(Action)
                                                {
                                                    GNode::doExecuteAction(Action)
                                                    {
                                                        MechanicalAction::processNodeTopDown(GNode)
                                                        {
                                                            MechanicalVOpAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                            {
                                                                MechanicalObject::vOp(VecId v, VecId a, VecId b, double f)
                                                                {
                                                                    if(v.isNull())
                                                                    {
                                                                        // ERROR
                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                        return;
                                                                    }
                                                                    if (a.isNull())
                                                                    {
                                                                        if (b.isNull())
                                                                        {
                                                                            // v = 0
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                vv->resize(this->vsize);
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = Coord();
                                                                            }
                                                                            else
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                vv->resize(this->vsize);
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = Deriv();
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            if (b.type != v.type)
                                                                            {
                                                                                // ERROR
                                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                return;
                                                                            }
                                                                            if (v == b)
                                                                            {
                                                                                // v *= f
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] *= (Real)f;
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] *= (Real)f;
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // v = b*f
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    VecCoord* vb = getVecCoord(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] = (*vb)[i] * (Real)f;
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] = (*vb)[i] * (Real)f;
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        if (a.type != v.type)
                                                                        {
                                                                            // ERROR
                                                                            std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                            return;
                                                                        }
                                                                        if (b.isNull())
                                                                        {
                                                                            // v = a
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                VecCoord* va = getVecCoord(a.index);
                                                                                vv->resize(va->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = (*va)[i];
                                                                            }
                                                                            else
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* va = getVecDeriv(a.index);
                                                                                vv->resize(va->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = (*va)[i];
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            if (v == a)
                                                                            {
                                                                                if (f==1.0)
                                                                                {
                                                                                    // v += b
                                                                                    if (v.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                                        if (b.type == V_COORD)
                                                                                        {
                                                                                            VecCoord* vb = getVecCoord(b.index);
                                                                                            vv->resize(vb->size());
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                                (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            VecDeriv* vb = getVecDeriv(b.index);
                                                                                            vv->resize(vb->size());
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                                (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                    }
                                                                                    else if (b.type == V_DERIV)
                                                                                    {
                                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        // ERROR
                                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                        return;
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    // v += b*f
                                                                                    if (v.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                                        if (b.type == V_COORD)
                                                                                        {
                                                                                            VecCoord* vb = getVecCoord(b.index);
                                                                                            vv->resize(vb->size());
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                                (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            VecDeriv* vb = getVecDeriv(b.index);
                                                                                            vv->resize(vb->size());
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                                (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                    }
                                                                                    else if (b.type == V_DERIV)
                                                                                    {
                                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        // ERROR
                                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                        return;
                                                                                    }
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                if (f==1.0)
                                                                                {
                                                                                    // v = a+b
                                                                                    if (v.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                                        VecCoord* va = getVecCoord(a.index);
                                                                                        vv->resize(va->size());
                                                                                        if (b.type == V_COORD)
                                                                                        {
                                                                                            VecCoord* vb = getVecCoord(b.index);
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                            {
                                                                                                (*vv)[i] = (*va)[i];
                                                                                                (*vv)[i] += (*vb)[i];
                                                                                            }
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            VecDeriv* vb = getVecDeriv(b.index);
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                            {
                                                                                                (*vv)[i] = (*va)[i];
                                                                                                (*vv)[i] += (*vb)[i];
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                    else if (b.type == V_DERIV)
                                                                                    {
                                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                                        VecDeriv* va = getVecDeriv(a.index);
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(va->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        // ERROR
                                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                        return;
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    // v = a+b*f
                                                                                    if (v.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                                        VecCoord* va = getVecCoord(a.index);
                                                                                        vv->resize(va->size());
                                                                                        if (b.type == V_COORD)
                                                                                        {
                                                                                            VecCoord* vb = getVecCoord(b.index);
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                            {
                                                                                                (*vv)[i] = (*va)[i];
                                                                                                (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                            }
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            VecDeriv* vb = getVecDeriv(b.index);
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                            {
                                                                                                (*vv)[i] = (*va)[i];
                                                                                                (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                    else if (b.type == V_DERIV)
                                                                                    {
                                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                                        VecDeriv* va = getVecDeriv(a.index);
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(va->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        // ERROR
                                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                        return;
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
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
                                    addMdx(f,vel)
                                    {
                                        MechanicalAddMDxAction(f,vel).execute( getContext() )
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalAddMDxAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::setF(VecId v)
                                                            {
                                                                if (v.type == V_DERIV)
                                                                {
                                                                    this->f = getVecDeriv(v.index);
                                                                }
                                                                else
                                                                {
                                                                    std::cerr << "Invalid setF operation ("<<v<<")\n";
                                                                }
                                                            }
                                                            MechanicalObject::setDx(VecId v)
                                                            {
                                                                if (v.type == V_DERIV)
                                                                {
                                                                    this->dx = getVecDeriv(v.index);
                                                                }
                                                                else
                                                                {
                                                                    std::cerr << "Invalid setDx operation ("<<v<<")\n";
                                                                }
                                                            }
                                                        }
                                                        MechanicalAddMDxAction::fwdMass(GNode, BasicMass)
                                                        {
                                                            Mass::addMDx()
                                                            {
                                                                /** Get the state vectors using
                                                                *  Mass->MechanicalModel->getF()  { return f;  },
                                                                *  Mass->MechanicalModel->getDx() const { return dx; }
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
                                    b.peq(f,-f_rayleighMass.getValue())
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
                                                                MechanicalObject::vOp(VecId v, VecId a, VecId b, double f)
                                                                {
                                                                    if(v.isNull())
                                                                    {
                                                                        // ERROR
                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                        return;
                                                                    }
                                                                    if (a.isNull())
                                                                    {
                                                                        if (b.isNull())
                                                                        {
                                                                            // v = 0
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                vv->resize(this->vsize);
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = Coord();
                                                                            }
                                                                            else
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                vv->resize(this->vsize);
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = Deriv();
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            if (b.type != v.type)
                                                                            {
                                                                                // ERROR
                                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                return;
                                                                            }
                                                                            if (v == b)
                                                                            {
                                                                                // v *= f
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] *= (Real)f;
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] *= (Real)f;
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // v = b*f
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    VecCoord* vb = getVecCoord(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] = (*vb)[i] * (Real)f;
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] = (*vb)[i] * (Real)f;
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        if (a.type != v.type)
                                                                        {
                                                                            // ERROR
                                                                            std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                            return;
                                                                        }
                                                                        if (b.isNull())
                                                                        {
                                                                            // v = a
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                VecCoord* va = getVecCoord(a.index);
                                                                                vv->resize(va->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = (*va)[i];
                                                                            }
                                                                            else
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* va = getVecDeriv(a.index);
                                                                                vv->resize(va->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = (*va)[i];
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            if (v == a)
                                                                            {
                                                                                if (f==1.0)
                                                                                {
                                                                                    // v += b
                                                                                    if (v.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                                        if (b.type == V_COORD)
                                                                                        {
                                                                                            VecCoord* vb = getVecCoord(b.index);
                                                                                            vv->resize(vb->size());
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                                (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            VecDeriv* vb = getVecDeriv(b.index);
                                                                                            vv->resize(vb->size());
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                                (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                    }
                                                                                    else if (b.type == V_DERIV)
                                                                                    {
                                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        // ERROR
                                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                        return;
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    // v += b*f
                                                                                    if (v.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                                        if (b.type == V_COORD)
                                                                                        {
                                                                                            VecCoord* vb = getVecCoord(b.index);
                                                                                            vv->resize(vb->size());
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                                (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            VecDeriv* vb = getVecDeriv(b.index);
                                                                                            vv->resize(vb->size());
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                                (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                    }
                                                                                    else if (b.type == V_DERIV)
                                                                                    {
                                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        // ERROR
                                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                        return;
                                                                                    }
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                if (f==1.0)
                                                                                {
                                                                                    // v = a+b
                                                                                    if (v.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                                        VecCoord* va = getVecCoord(a.index);
                                                                                        vv->resize(va->size());
                                                                                        if (b.type == V_COORD)
                                                                                        {
                                                                                            VecCoord* vb = getVecCoord(b.index);
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                            {
                                                                                                (*vv)[i] = (*va)[i];
                                                                                                (*vv)[i] += (*vb)[i];
                                                                                            }
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            VecDeriv* vb = getVecDeriv(b.index);
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                            {
                                                                                                (*vv)[i] = (*va)[i];
                                                                                                (*vv)[i] += (*vb)[i];
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                    else if (b.type == V_DERIV)
                                                                                    {
                                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                                        VecDeriv* va = getVecDeriv(a.index);
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(va->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        // ERROR
                                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                        return;
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    // v = a+b*f
                                                                                    if (v.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                                        VecCoord* va = getVecCoord(a.index);
                                                                                        vv->resize(va->size());
                                                                                        if (b.type == V_COORD)
                                                                                        {
                                                                                            VecCoord* vb = getVecCoord(b.index);
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                            {
                                                                                                (*vv)[i] = (*va)[i];
                                                                                                (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                            }
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            VecDeriv* vb = getVecDeriv(b.index);
                                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                            {
                                                                                                (*vv)[i] = (*va)[i];
                                                                                                (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                    else if (b.type == V_DERIV)
                                                                                    {
                                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                                        VecDeriv* va = getVecDeriv(a.index);
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(va->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        // ERROR
                                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                        return;
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
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
                                }
                                /** b = h(f0 + (h+rs)df/dx v - rd M v)
                                */
                                b.teq(h)
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
                                                            MechanicalObject::vOp(VecId v, VecId a, VecId b, double f)
                                                            {
                                                                if(v.isNull())
                                                                {
                                                                    // ERROR
                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                    return;
                                                                }
                                                                if (a.isNull())
                                                                {
                                                                    if (b.isNull())
                                                                    {
                                                                        // v = 0
                                                                        if (v.type == V_COORD)
                                                                        {
                                                                            VecCoord* vv = getVecCoord(v.index);
                                                                            vv->resize(this->vsize);
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = Coord();
                                                                        }
                                                                        else
                                                                        {
                                                                            VecDeriv* vv = getVecDeriv(v.index);
                                                                            vv->resize(this->vsize);
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = Deriv();
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        if (b.type != v.type)
                                                                        {
                                                                            // ERROR
                                                                            std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                            return;
                                                                        }
                                                                        if (v == b)
                                                                        {
                                                                            // v *= f
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] *= (Real)f;
                                                                            }
                                                                            else
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] *= (Real)f;
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            // v = b*f
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                VecCoord* vb = getVecCoord(b.index);
                                                                                vv->resize(vb->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = (*vb)[i] * (Real)f;
                                                                            }
                                                                            else
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(vb->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = (*vb)[i] * (Real)f;
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                                else
                                                                {
                                                                    if (a.type != v.type)
                                                                    {
                                                                        // ERROR
                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                        return;
                                                                    }
                                                                    if (b.isNull())
                                                                    {
                                                                        // v = a
                                                                        if (v.type == V_COORD)
                                                                        {
                                                                            VecCoord* vv = getVecCoord(v.index);
                                                                            VecCoord* va = getVecCoord(a.index);
                                                                            vv->resize(va->size());
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = (*va)[i];
                                                                        }
                                                                        else
                                                                        {
                                                                            VecDeriv* vv = getVecDeriv(v.index);
                                                                            VecDeriv* va = getVecDeriv(a.index);
                                                                            vv->resize(va->size());
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = (*va)[i];
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        if (v == a)
                                                                        {
                                                                            if (f==1.0)
                                                                            {
                                                                                // v += b
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // v += b*f
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            if (f==1.0)
                                                                            {
                                                                                // v = a+b
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    VecCoord* va = getVecCoord(a.index);
                                                                                    vv->resize(va->size());
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* va = getVecDeriv(a.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(va->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // v = a+b*f
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    VecCoord* va = getVecCoord(a.index);
                                                                                    vv->resize(va->size());
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* va = getVecDeriv(a.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(va->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
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
                                projectResponse(b)
                                {
                                    MechanicalApplyConstraintsAction(b).execute( getContext() )
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalApplyConstraintsAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::setDx(VecId v)
                                                        {
                                                            if (v.type == V_DERIV)
                                                            {
                                                                this->dx = getVecDeriv(v.index);
                                                            }
                                                            else
                                                            {
                                                                std::cerr << "Invalid setDx operation ("<<v<<")\n";
                                                            }
                                                        }
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

                                double normb = sqrt(b.dot(b))
                                {
                                    /** pile simplifiée */
                                    OdeSolver::v_dot(VecId a, VecId b)
                                    {
                                        MechanicalVDotAction(a,b,&result).execute( getContext() )
                                        {
                                            ...
                                            MechanicalVDotAction::fwdMechanicalModel()
                                            {
                                                MechanicalObject::vDot()
                                                {
                                                    double r = 0.0;
                                                    if (a.type == V_COORD && b.type == V_COORD)
                                                    {
                                                        VecCoord* va = getVecCoord(a.index);
                                                        VecCoord* vb = getVecCoord(b.index);
                                                        for (unsigned int i=0; i<va->size(); i++)
                                                            r += (*va)[i] * (*vb)[i];
                                                    }
                                                    else if (a.type == V_DERIV && b.type == V_DERIV)
                                                    {
                                                        VecDeriv* va = getVecDeriv(a.index);
                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                        for (unsigned int i=0; i<va->size(); i++)
                                                            r += (*va)[i] * (*vb)[i];
                                                    }
                                                    else
                                                    {
                                                        std::cerr << "Invalid dot operation ("<<a<<','<<b<<")\n";
                                                    }
                                                    return r;
                                                }
                                            }
                                            Action::for_each()
                                            {
                                                MechanicalAction::fwdInteractionForceField();
                                            }
                                        }
                                    }
                                }

                                // -- solve the system using a conjugate gradient solution
                                double rho, rho_1=0, alpha, beta;
                                v_clear( x )
                                {
                                    MechanicalVOpAction(v).execute( getContext() )
                                    {
                                        GNode::executeAction(Action)
                                        {
                                            GNode::doExecuteAction(Action)
                                            {
                                                MechanicalAction::processNodeTopDown(GNode)
                                                {
                                                    MechanicalVOpAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                    {
                                                        MechanicalObject::vOp(VecId v, VecId a, VecId b, double f)
                                                        {
                                                            if(v.isNull())
                                                            {
                                                                // ERROR
                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                return;
                                                            }
                                                            if (a.isNull())
                                                            {
                                                                if (b.isNull())
                                                                {
                                                                    // v = 0
                                                                    if (v.type == V_COORD)
                                                                    {
                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                        vv->resize(this->vsize);
                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                            (*vv)[i] = Coord();
                                                                    }
                                                                    else
                                                                    {
                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                        vv->resize(this->vsize);
                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                            (*vv)[i] = Deriv();
                                                                    }
                                                                }
                                                                else
                                                                {
                                                                    if (b.type != v.type)
                                                                    {
                                                                        // ERROR
                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                        return;
                                                                    }
                                                                    if (v == b)
                                                                    {
                                                                        // v *= f
                                                                        if (v.type == V_COORD)
                                                                        {
                                                                            VecCoord* vv = getVecCoord(v.index);
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] *= (Real)f;
                                                                        }
                                                                        else
                                                                        {
                                                                            VecDeriv* vv = getVecDeriv(v.index);
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] *= (Real)f;
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        // v = b*f
                                                                        if (v.type == V_COORD)
                                                                        {
                                                                            VecCoord* vv = getVecCoord(v.index);
                                                                            VecCoord* vb = getVecCoord(b.index);
                                                                            vv->resize(vb->size());
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = (*vb)[i] * (Real)f;
                                                                        }
                                                                        else
                                                                        {
                                                                            VecDeriv* vv = getVecDeriv(v.index);
                                                                            VecDeriv* vb = getVecDeriv(b.index);
                                                                            vv->resize(vb->size());
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = (*vb)[i] * (Real)f;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                            else
                                                            {
                                                                if (a.type != v.type)
                                                                {
                                                                    // ERROR
                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                    return;
                                                                }
                                                                if (b.isNull())
                                                                {
                                                                    // v = a
                                                                    if (v.type == V_COORD)
                                                                    {
                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                        VecCoord* va = getVecCoord(a.index);
                                                                        vv->resize(va->size());
                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                            (*vv)[i] = (*va)[i];
                                                                    }
                                                                    else
                                                                    {
                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                        VecDeriv* va = getVecDeriv(a.index);
                                                                        vv->resize(va->size());
                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                            (*vv)[i] = (*va)[i];
                                                                    }
                                                                }
                                                                else
                                                                {
                                                                    if (v == a)
                                                                    {
                                                                        if (f==1.0)
                                                                        {
                                                                            // v += b
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                if (b.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vb = getVecCoord(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                }
                                                                            }
                                                                            else if (b.type == V_DERIV)
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(vb->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] += (*vb)[i];
                                                                            }
                                                                            else
                                                                            {
                                                                                // ERROR
                                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                return;
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            // v += b*f
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                if (b.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vb = getVecCoord(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                }
                                                                            }
                                                                            else if (b.type == V_DERIV)
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(vb->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] += (*vb)[i]*(Real)f;
                                                                            }
                                                                            else
                                                                            {
                                                                                // ERROR
                                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                return;
                                                                            }
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        if (f==1.0)
                                                                        {
                                                                            // v = a+b
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                VecCoord* va = getVecCoord(a.index);
                                                                                vv->resize(va->size());
                                                                                if (b.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vb = getVecCoord(b.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                }
                                                                            }
                                                                            else if (b.type == V_DERIV)
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* va = getVecDeriv(a.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(va->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                {
                                                                                    (*vv)[i] = (*va)[i];
                                                                                    (*vv)[i] += (*vb)[i];
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // ERROR
                                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                return;
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            // v = a+b*f
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                VecCoord* va = getVecCoord(a.index);
                                                                                vv->resize(va->size());
                                                                                if (b.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vb = getVecCoord(b.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                }
                                                                            }
                                                                            else if (b.type == V_DERIV)
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* va = getVecDeriv(a.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(va->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                {
                                                                                    (*vv)[i] = (*va)[i];
                                                                                    (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // ERROR
                                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                return;
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
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
                                /** Initial residual
                                */
                                v_eq(r,b)
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
                                                        MechanicalObject::vOp(VecId v, VecId a, VecId b, double f)
                                                        {
                                                            if(v.isNull())
                                                            {
                                                                // ERROR
                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                return;
                                                            }
                                                            if (a.isNull())
                                                            {
                                                                if (b.isNull())
                                                                {
                                                                    // v = 0
                                                                    if (v.type == V_COORD)
                                                                    {
                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                        vv->resize(this->vsize);
                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                            (*vv)[i] = Coord();
                                                                    }
                                                                    else
                                                                    {
                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                        vv->resize(this->vsize);
                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                            (*vv)[i] = Deriv();
                                                                    }
                                                                }
                                                                else
                                                                {
                                                                    if (b.type != v.type)
                                                                    {
                                                                        // ERROR
                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                        return;
                                                                    }
                                                                    if (v == b)
                                                                    {
                                                                        // v *= f
                                                                        if (v.type == V_COORD)
                                                                        {
                                                                            VecCoord* vv = getVecCoord(v.index);
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] *= (Real)f;
                                                                        }
                                                                        else
                                                                        {
                                                                            VecDeriv* vv = getVecDeriv(v.index);
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] *= (Real)f;
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        // v = b*f
                                                                        if (v.type == V_COORD)
                                                                        {
                                                                            VecCoord* vv = getVecCoord(v.index);
                                                                            VecCoord* vb = getVecCoord(b.index);
                                                                            vv->resize(vb->size());
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = (*vb)[i] * (Real)f;
                                                                        }
                                                                        else
                                                                        {
                                                                            VecDeriv* vv = getVecDeriv(v.index);
                                                                            VecDeriv* vb = getVecDeriv(b.index);
                                                                            vv->resize(vb->size());
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = (*vb)[i] * (Real)f;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                            else
                                                            {
                                                                if (a.type != v.type)
                                                                {
                                                                    // ERROR
                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                    return;
                                                                }
                                                                if (b.isNull())
                                                                {
                                                                    // v = a
                                                                    if (v.type == V_COORD)
                                                                    {
                                                                        VecCoord* vv = getVecCoord(v.index);
                                                                        VecCoord* va = getVecCoord(a.index);
                                                                        vv->resize(va->size());
                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                            (*vv)[i] = (*va)[i];
                                                                    }
                                                                    else
                                                                    {
                                                                        VecDeriv* vv = getVecDeriv(v.index);
                                                                        VecDeriv* va = getVecDeriv(a.index);
                                                                        vv->resize(va->size());
                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                            (*vv)[i] = (*va)[i];
                                                                    }
                                                                }
                                                                else
                                                                {
                                                                    if (v == a)
                                                                    {
                                                                        if (f==1.0)
                                                                        {
                                                                            // v += b
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                if (b.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vb = getVecCoord(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                }
                                                                            }
                                                                            else if (b.type == V_DERIV)
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(vb->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] += (*vb)[i];
                                                                            }
                                                                            else
                                                                            {
                                                                                // ERROR
                                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                return;
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            // v += b*f
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                if (b.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vb = getVecCoord(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                }
                                                                            }
                                                                            else if (b.type == V_DERIV)
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(vb->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] += (*vb)[i]*(Real)f;
                                                                            }
                                                                            else
                                                                            {
                                                                                // ERROR
                                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                return;
                                                                            }
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        if (f==1.0)
                                                                        {
                                                                            // v = a+b
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                VecCoord* va = getVecCoord(a.index);
                                                                                vv->resize(va->size());
                                                                                if (b.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vb = getVecCoord(b.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                }
                                                                            }
                                                                            else if (b.type == V_DERIV)
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* va = getVecDeriv(a.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(va->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                {
                                                                                    (*vv)[i] = (*va)[i];
                                                                                    (*vv)[i] += (*vb)[i];
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // ERROR
                                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                return;
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            // v = a+b*f
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                VecCoord* va = getVecCoord(a.index);
                                                                                vv->resize(va->size());
                                                                                if (b.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vb = getVecCoord(b.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                }
                                                                            }
                                                                            else if (b.type == V_DERIV)
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* va = getVecDeriv(a.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(va->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                {
                                                                                    (*vv)[i] = (*va)[i];
                                                                                    (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // ERROR
                                                                                std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                return;
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
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

                                unsigned nb_iter;
                                const char* endcond = "iterations";
                                for( nb_iter=1; nb_iter<=f_maxIter.getValue(); nb_iter++ )
                                {
                                    rho = r.dot(r);

                                    if( nb_iter==1 )
                                        p = r; //z;
                                    else
                                    {
                                        beta = rho / rho_1;
                                        p *= beta
                                        {
                                            /** MultiVector : operator *= overloaded */
                                            teq(f);
                                        }
                                        p += r
                                        {
                                            /** MultiVector : operator += overloaded */
                                            peq(f);
                                        }
                                    }

                                    // matrix-vector product
                                    // dx = p
                                    propagateDx(p)
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
                                                            MechanicalObject::setDx(VecId v)
                                                            {
                                                                if (v.type == V_DERIV)
                                                                {
                                                                    this->dx = getVecDeriv(v.index);
                                                                }
                                                                else
                                                                {
                                                                    std::cerr << "Invalid setDx operation ("<<v<<")\n";
                                                                }
                                                            }
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
                                    // q = df/dx p
                                    computeDf(q)
                                    {
                                        MechanicalResetForceAction(q).execute( getContext() )
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
                                                            MechanicalObject::setF(VecId v)
                                                            {
                                                                if (v.type == V_DERIV)
                                                                {
                                                                    this->f = getVecDeriv(v.index);
                                                                }
                                                                else
                                                                {
                                                                    std::cerr << "Invalid setF operation ("<<v<<")\n";
                                                                }
                                                            }
                                                            MechanicalObject::resetForce()
                                                            {
                                                                VecDeriv& f= *getF()  { return f;  };
                                                                for( unsigned i=0; i<f.size(); ++i )
                                                                    f[i] = Deriv();
                                                            }
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
                                        MechanicalComputeDfAction(q).execute( getContext() )
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
                                                            MechanicalObject::setF(VecId v)
                                                            {
                                                                if (v.type == V_DERIV)
                                                                {
                                                                    this->f = getVecDeriv(v.index);
                                                                }
                                                                else
                                                                {
                                                                    std::cerr << "Invalid setF operation ("<<v<<")\n";
                                                                }
                                                            }
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
                                    // q = -h(h+rs) df/dx p
                                    q *= -h*(h+f_rayleighStiffness.getValue())
                                    {
                                        /** MultiVector : operator *= overloaded */
                                        teq(f);
                                    }

                                    // apply global Rayleigh damping
                                    if (f_rayleighMass.getValue()==0.0)
                                        addMdx( q, p);           // q = Mp -h(h+rs) df/dx p
                                    else
                                    {
                                        q2.clear();
                                        addMdx( q2, p);
                                        q.peq(q2,(1+h*f_rayleighMass.getValue())); // q = Mp -h(h+rs) df/dx p +hr Mp  =  (M + dt(rd M + rs K) + dt2 K) dx
                                    }

                                    // filter the product to take the constraints into account
                                    // q is projected to the constrained space
                                    projectResponse(q)
                                    {
                                        MechanicalApplyConstraintsAction(q).execute( getContext() )
                                        {
                                            GNode::executeAction(Action)
                                            {
                                                GNode::doExecuteAction(Action)
                                                {
                                                    MechanicalAction::processNodeTopDown(GNode)
                                                    {
                                                        MechanicalApplyConstraintsAction::fwdMechanicalModel(GNode, BasicMechanicalModel)
                                                        {
                                                            MechanicalObject::setDx(VecId v)
                                                            {
                                                                if (v.type == V_DERIV)
                                                                {
                                                                    this->dx = getVecDeriv(v.index);
                                                                }
                                                                else
                                                                {
                                                                    std::cerr << "Invalid setDx operation ("<<v<<")\n";
                                                                }
                                                            }
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

                                    double den = p.dot(q);


                                    if( fabs(den)<f_smallDenominatorThreshold.getValue() )
                                    {
                                        endcond = "threshold";
                                        break;
                                    }
                                    alpha = rho/den;
                                    x.peq(p,alpha);                 // x = x + alpha p
                                    r.peq(q,-alpha);                // r = r - alpha r

                                    double normr = sqrt(r.dot(r));
                                    if (normr/normb <= f_tolerance.getValue())
                                    {
                                        endcond = "tolerance";
                                        break;
                                    }
                                    rho_1 = rho;
                                }
                                // x is the solution of the system

                                // apply the solution
                                vel.peq( x );                       // vel = vel + x
                                /** Compute the value of the new positions x and new velocities v
                                *  Apply the solution
                                *  pos = pos + h vel
                                */
                                pos.peq( vel, h )
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
                                                            MechanicalObject::vOp(VecId v, VecId a, VecId b, double f)
                                                            {
                                                                if(v.isNull())
                                                                {
                                                                    // ERROR
                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                    return;
                                                                }
                                                                if (a.isNull())
                                                                {
                                                                    if (b.isNull())
                                                                    {
                                                                        // v = 0
                                                                        if (v.type == V_COORD)
                                                                        {
                                                                            VecCoord* vv = getVecCoord(v.index);
                                                                            vv->resize(this->vsize);
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = Coord();
                                                                        }
                                                                        else
                                                                        {
                                                                            VecDeriv* vv = getVecDeriv(v.index);
                                                                            vv->resize(this->vsize);
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = Deriv();
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        if (b.type != v.type)
                                                                        {
                                                                            // ERROR
                                                                            std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                            return;
                                                                        }
                                                                        if (v == b)
                                                                        {
                                                                            // v *= f
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] *= (Real)f;
                                                                            }
                                                                            else
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] *= (Real)f;
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            // v = b*f
                                                                            if (v.type == V_COORD)
                                                                            {
                                                                                VecCoord* vv = getVecCoord(v.index);
                                                                                VecCoord* vb = getVecCoord(b.index);
                                                                                vv->resize(vb->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = (*vb)[i] * (Real)f;
                                                                            }
                                                                            else
                                                                            {
                                                                                VecDeriv* vv = getVecDeriv(v.index);
                                                                                VecDeriv* vb = getVecDeriv(b.index);
                                                                                vv->resize(vb->size());
                                                                                for (unsigned int i=0; i<vv->size(); i++)
                                                                                    (*vv)[i] = (*vb)[i] * (Real)f;
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                                else
                                                                {
                                                                    if (a.type != v.type)
                                                                    {
                                                                        // ERROR
                                                                        std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                        return;
                                                                    }
                                                                    if (b.isNull())
                                                                    {
                                                                        // v = a
                                                                        if (v.type == V_COORD)
                                                                        {
                                                                            VecCoord* vv = getVecCoord(v.index);
                                                                            VecCoord* va = getVecCoord(a.index);
                                                                            vv->resize(va->size());
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = (*va)[i];
                                                                        }
                                                                        else
                                                                        {
                                                                            VecDeriv* vv = getVecDeriv(v.index);
                                                                            VecDeriv* va = getVecDeriv(a.index);
                                                                            vv->resize(va->size());
                                                                            for (unsigned int i=0; i<vv->size(); i++)
                                                                                (*vv)[i] = (*va)[i];
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        if (v == a)
                                                                        {
                                                                            if (f==1.0)
                                                                            {
                                                                                // v += b
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // v += b*f
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        vv->resize(vb->size());
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(vb->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                        }
                                                                        else
                                                                        {
                                                                            if (f==1.0)
                                                                            {
                                                                                // v = a+b
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    VecCoord* va = getVecCoord(a.index);
                                                                                    vv->resize(va->size());
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i];
                                                                                        }
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* va = getVecDeriv(a.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(va->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i];
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                            else
                                                                            {
                                                                                // v = a+b*f
                                                                                if (v.type == V_COORD)
                                                                                {
                                                                                    VecCoord* vv = getVecCoord(v.index);
                                                                                    VecCoord* va = getVecCoord(a.index);
                                                                                    vv->resize(va->size());
                                                                                    if (b.type == V_COORD)
                                                                                    {
                                                                                        VecCoord* vb = getVecCoord(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        VecDeriv* vb = getVecDeriv(b.index);
                                                                                        for (unsigned int i=0; i<vv->size(); i++)
                                                                                        {
                                                                                            (*vv)[i] = (*va)[i];
                                                                                            (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                        }
                                                                                    }
                                                                                }
                                                                                else if (b.type == V_DERIV)
                                                                                {
                                                                                    VecDeriv* vv = getVecDeriv(v.index);
                                                                                    VecDeriv* va = getVecDeriv(a.index);
                                                                                    VecDeriv* vb = getVecDeriv(b.index);
                                                                                    vv->resize(va->size());
                                                                                    for (unsigned int i=0; i<vv->size(); i++)
                                                                                    {
                                                                                        (*vv)[i] = (*va)[i];
                                                                                        (*vv)[i] += (*vb)[i]*(Real)f;
                                                                                    }
                                                                                }
                                                                                else
                                                                                {
                                                                                    // ERROR
                                                                                    std::cerr << "Invalid vOp operation ("<<v<<','<<a<<','<<b<<','<<f<<")\n";
                                                                                    return;
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
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
                                if (f_velocityDamping.getValue()!=0.0)
                                    vel *= exp(-h*f_velocityDamping.getValue())
                                {
                                    /** MultiVector : operator *= overloaded */
                                    teq(f);
                                }

                                /** Free memory */
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
                                                            MechanicalObject::vFree(VecId v)
                                                            {
                                                                if (v.type == V_COORD && v.index >= V_FIRST_DYNAMIC_INDEX)
                                                                {
                                                                    VecCoord* vec = getVecCoord(v.index);
                                                                    vec->resize(0);
                                                                }
                                                                else if (v.type == V_DERIV && v.index >= V_FIRST_DYNAMIC_INDEX)
                                                                {
                                                                    VecDeriv* vec = getVecDeriv(v.index);
                                                                    vec->resize(0);
                                                                }
                                                                else
                                                                {
                                                                    std::cerr << "Invalid free operation ("<<v<<")\n";
                                                                    return;
                                                                }
                                                            }
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
                                MechanicalPropagatePositionAndVelocityAction(t,x,v).execute( getContext() )
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
                                                        MechanicalObject::setX(VecId v)
                                                        {
                                                            if (v.type == V_COORD)
                                                            {
                                                                this->x = getVecCoord(v.index);
                                                            }
                                                            else
                                                            {
                                                                std::cerr << "Invalid setX operation ("<<v<<")\n";
                                                            }
                                                        }
                                                        MechanicalObject::setV(VecId v)
                                                        {
                                                            if (v.type == V_DERIV)
                                                            {
                                                                this->f = getVecDeriv(v.index);
                                                            }
                                                            else
                                                            {
                                                                std::cerr << "Invalid setF operation ("<<v<<")\n";
                                                            }
                                                        }
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
                                                            Constraint::projectPosition()
                                                            {
                                                                /** Empty */
                                                            }
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
                                                MechanicalObject::endIntegration(double)
                                                {
                                                    this->f = this->externalForces;
                                                    this->externalForces->clear();
                                                }
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
        GNode::execute<UpdateMappingAction>() {}

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
