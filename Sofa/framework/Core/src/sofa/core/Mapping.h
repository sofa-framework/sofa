/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/core/BaseMapping.h>
#include <sofa/core/State.h>

namespace sofa::core
{

/**
*  \brief Specialized interface to convert a model state of type TIn to a model state of type TOut.
* This is basically a sofa::core::BaseMapping with given input and output types.
*
*
*/
template <class TIn, class TOut>
class Mapping : public BaseMapping
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(Mapping,TIn,TOut), BaseMapping);

    /// Input Data Type
    typedef TIn In;
    /// Output Data Type
    typedef TOut Out;

    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

protected:
    /// Input Model, also called parent
    SingleLink<Mapping<In,Out>, State< In >, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> fromModel;
    /// Output Model, also called child
    SingleLink<Mapping<In,Out>, State< Out >, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> toModel;
public:

    /// by default rest position are NOT propagated to mapped dofs.
    /// In some cases, rest pos is needed for mapped dofs (generally when this dof is used to compute mechanics).
    /// In that case, Data applyRestPosition must be setted to true for all the mappings until the desired dof.
    Data<bool> f_applyRestPosition;
protected:
    /// Constructor, taking input and output models as parameters.
    ///
    /// Note that if you do not specify these models here, you must call
    /// setModels with non-nullptr value before the intialization (i.e. before
    /// init() is called).
    Mapping(State< In >* from=nullptr, State< Out >* to=nullptr);
    /// Destructor
    ~Mapping() override;
public:
    /// Specify the input and output models.
    virtual void setModels(State< In > * from, State< Out >* to);
    /// If the type is compatible set the input model and return true, otherwise do nothing and return false.
    bool setFrom( BaseState* from ) override;
    /// If the type is compatible set the output model and return true, otherwise do nothing and return false.
    bool setTo( BaseState* to ) override;

    /// Set the path to the objects mapped in the scene graph
    void setPathInputObject(const std::string &o) {fromModel.setPath(o);}
    void setPathOutputObject(const std::string &o) {toModel.setPath(o);}

    /// Return the pointer to the input model.
    State< In >* getFromModel();
    /// Return the pointer to the output model.
    State< Out >* getToModel();

    /// Return the pointer to the input model.
    type::vector<BaseState*> getFrom() override;
    /// Return the pointer to the output model.
    type::vector<BaseState*> getTo() override;

    /// Apply ///
    /// Apply the mapping to position vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    void apply (const MechanicalParams* mparams, MultiVecCoordId outPos, ConstMultiVecCoordId inPos ) override;

    /// This method must be reimplemented by all mappings.
    virtual void apply( const MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in)= 0;

    /// ApplyJ ///
    /// Apply the mapping to derived (velocity, displacement) vectors.
    /// $ out = J in $
    /// where J is the tangent operator (the linear approximation) of the mapping
    void applyJ(const MechanicalParams* mparams, MultiVecDerivId outVel, ConstMultiVecDerivId inVel ) override;

    /// This method must be reimplemented by all mappings.
    virtual void applyJ( const MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in) = 0;

    /// ApplyJT (Force)///
    /// Apply the reverse mapping to force vectors.
    /// $ out += J^t in $
    /// where J is the tangent operator (the linear approximation) of the mapping
    void applyJT(const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce ) override;

    /// This method must be reimplemented by all mappings.
    virtual void applyJT( const MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in) = 0;

    /// ApplyDJT (Force)///
    /// Apply the change of force due to the nonlinearity of the mapping and the last propagated displacement. Also called geometric stiffness.
    /// The default implementation does nothing, assuming a linear mapping.
    ///
    /// This method computes
    /// \f$ f_p += dJ^t f_c \f$, where \f$ f_p \f$ is the parent force and  \f$ f_c \f$ is the child force.
    /// where J is the tangent operator (the linear approximation) of the mapping
    /// The child force is accessed in the child state using mparams->readF() .  This requires that the child force vector is used by the solver to compute the force \f$ f(x,v)\f$ corresponding to the current positions and velocities, and not to store auxiliary values.
    /// The displacement is accessed in the parent state using mparams->readDx() .
    /// This method generally corresponds to a symmetric stiffness matrix, but with rotations (which are not a commutative group) it is not the case.
    /// Since some solvers (including the Conjugate Gradient) require symmetric matrices, a flag is set in the MechanicalParams to say if symmetric matrices are required. If so, non-symmetric geometric stiffness should not be applied.
    void applyDJT(const MechanicalParams* /*mparams = */ , MultiVecDerivId /*parentForce*/, ConstMultiVecDerivId  /*childForce*/ ) override;

    /// ApplyJT (Constraint)///
    void applyJT(const ConstraintParams* cparams, MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst ) override;

    /// This method must be reimplemented by all mappings if they need to support constraints.
    virtual void applyJT( const ConstraintParams* /* mparams */, InDataMatrixDeriv& /* out */, const OutDataMatrixDeriv& /* in */)
    {
        msg_error() << "This mapping does not support certain constraints because Mapping::applyJT( const ConstraintParams* , InDataMatrixDeriv&, const OutDataMatrixDeriv&) is not overloaded.";
    }

    /// computeAccFromMapping
    /// Compute the acceleration of the child, based on the acceleration and the velocity of the parent.
    /// Let \f$ v_c = J v_p \f$ be the velocity of the child given the velocity of the parent, then the acceleration is \f$ a_c = J a_p + dJ v_p \f$.
    /// The second term is null in linear mappings, otherwise it encodes the acceleration due to the change of mapping at constant parent velocity.
    /// For instance, in a rigid mapping with angular velocity\f$ w \f$,  the second term is $ w^(w^rel_pos) $
    void computeAccFromMapping(const MechanicalParams* mparams, MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc ) override;

    /// This method must be reimplemented by all mappings if they need to support composite accelerations
    virtual void computeAccFromMapping(const MechanicalParams* /* mparams */, OutDataVecDeriv& /* accOut */, const InDataVecDeriv& /* vIn */, const InDataVecDeriv& /* accIn */)
    {
    }

    /// Propagate positions and velocities to the output
    void init() override;

    ///<TO REMOVE>  FF:why would we remove this, is there any alternative function ?
    // Useful ?
    /// Get the source (upper) model.
    virtual type::vector<behavior::BaseMechanicalState*> getMechFrom() override;

    /// Get the destination (lower, mapped) model.
    virtual type::vector<behavior::BaseMechanicalState*> getMechTo() override;

    //Create a matrix for mapped mechanical objects
    //If the two mechanical objects is identical, create a new stiffness matrix for this mapped objects
    //If the two mechanical objects is different, create a new interaction matrix
    sofa::linearalgebra::BaseMatrix* createMappedMatrix(const behavior::BaseMechanicalState* state1, const behavior::BaseMechanicalState* state2, func_createMappedMatrix) override;

    /// Disable the mapping to get the original coordinates of the mapped model.
    ///
    /// It is for instance used in RigidMapping to get the local coordinates of the object.
    void disable() override;

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the input and output attributes and check
    /// if they are compatible with the input and output model types of this
    /// mapping.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        State<In>* stin = nullptr;
        State<Out>* stout = nullptr;

        std::string inPath, outPath;

        if (arg->getAttribute("input"))
            inPath = arg->getAttribute("input");
        else
            inPath = "@../";

        context->findLinkDest(stin, inPath, nullptr);

        if (arg->getAttribute("output"))
            outPath = arg->getAttribute("output");
        else
            outPath = "@./";

        context->findLinkDest(stout, outPath, nullptr);

        if (stin == nullptr)
        {
            arg->logError("Data attribute 'input' does not point to a mechanical state of data type '"+std::string(In::Name())+"' and none can be found in the parent node context.");
            return false;
        }

        if (stout == nullptr)
        {
            arg->logError("Data attribute 'output' does not point to a mechanical state of data type '"+std::string(Out::Name())+"' and none can be found in the parent node context.");
            return false;
        }

        if (dynamic_cast<BaseObject*>(stin) == dynamic_cast<BaseObject*>(stout))
        {
            // we should refuse to create mappings with the same input and output model, which may happen if a State object is missing in the child node
            arg->logError("Both the input and the output point to the same mechanical state ('"+stin->getName()+"').");
            return false;
        }

        return BaseMapping::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    ///
    /// This implementation read the input and output attributes to
    /// find the input and output models of this mapping.
    template<class T>
    static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>();

        if (context)
            context->addObject(obj);

        if (arg)
        {
            std::string inPath, outPath;
            if (arg->getAttribute("input"))
                inPath = arg->getAttribute("input");
            else
                inPath = "@../";

            if (arg->getAttribute("output"))
                outPath = arg->getAttribute("output");
            else
                outPath = "@./";

            obj->fromModel.setPath( inPath );
            obj->toModel.setPath( outPath );

            obj->parse(arg);
        }

        return obj;
    }

    template<class T>
    static std::string shortName(const T* ptr = nullptr, objectmodel::BaseObjectDescription* arg = nullptr)
    {
        std::string name = Inherit1::shortName(ptr, arg);
        sofa::helper::replaceAll(name, "Mapping", "Map");
        return name;
    }

};


#if !defined(SOFA_CORE_MAPPING_CPP)

extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec2Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types >;

extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec1Types >;

extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types >;

extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6Types, sofa::defaulttype::Vec6Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6Types, sofa::defaulttype::Vec3Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6Types, sofa::defaulttype::Vec1Types >;

// Rigid templates
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Rigid2Types >;

extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec6Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec3Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec1Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Rigid3Types >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Rigid3Types >;

// cross templates
#endif
} // namespace sofa::core
