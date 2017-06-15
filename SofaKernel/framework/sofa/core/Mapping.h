/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_MAPPING_H
#define SOFA_CORE_MAPPING_H

#include <sofa/core/BaseMapping.h>
#include <sofa/core/State.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace core
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
    /// setModels with non-NULL value before the intialization (i.e. before
    /// init() is called).
    Mapping(State< In >* from=NULL, State< Out >* to=NULL);
    /// Destructor
    virtual ~Mapping();
public:
    /// Specify the input and output models.
    virtual void setModels(State< In > * from, State< Out >* to);
    /// If the type is compatible set the input model and return true, otherwise do nothing and return false.
    virtual bool setFrom( BaseState* from );
    /// If the type is compatible set the output model and return true, otherwise do nothing and return false.
    virtual bool setTo( BaseState* to );

    /// Set the path to the objects mapped in the scene graph
    void setPathInputObject(const std::string &o) {fromModel.setPath(o);}
    void setPathOutputObject(const std::string &o) {toModel.setPath(o);}

    /// Return the pointer to the input model.
    State< In >* getFromModel();
    /// Return the pointer to the output model.
    State< Out >* getToModel();

    /// Return the pointer to the input model.
    helper::vector<BaseState*> getFrom();
    /// Return the pointer to the output model.
    helper::vector<BaseState*> getTo();

    /// Apply ///
    /// Apply the mapping to position vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    virtual void apply (const MechanicalParams* mparams, MultiVecCoordId outPos, ConstMultiVecCoordId inPos ) ;

    /// This method must be reimplemented by all mappings.
    virtual void apply( const MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in)= 0;

    /// ApplyJ ///
    /// Apply the mapping to derived (velocity, displacement) vectors.
    /// $ out = J in $
    /// where J is the tangent operator (the linear approximation) of the mapping
    virtual void applyJ(const MechanicalParams* mparams, MultiVecDerivId outVel, ConstMultiVecDerivId inVel );

    /// This method must be reimplemented by all mappings.
    virtual void applyJ( const MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in) = 0;

    /// ApplyJT (Force)///
    /// Apply the reverse mapping to force vectors.
    /// $ out += J^t in $
    /// where J is the tangent operator (the linear approximation) of the mapping
    virtual void applyJT(const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce );

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
    virtual void applyDJT(const MechanicalParams* /*mparams = MechanicalParams::defaultInstance()*/ , MultiVecDerivId /*parentForce*/, ConstMultiVecDerivId  /*childForce*/ );

    /// ApplyJT (Constraint)///
    virtual void applyJT(const ConstraintParams* cparams, MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst );

    /// This method must be reimplemented by all mappings if they need to support constraints.
    virtual void applyJT( const ConstraintParams* /* mparams */, InDataMatrixDeriv& /* out */, const OutDataMatrixDeriv& /* in */)
    {
        serr << "This mapping does not support certain constraints because Mapping::applyJT( const ConstraintParams* , InDataMatrixDeriv&, const OutDataMatrixDeriv&) is not overloaded." << sendl;
    }

    /// computeAccFromMapping
    /// Compute the acceleration of the child, based on the acceleration and the velocity of the parent.
    /// Let \f$ v_c = J v_p \f$ be the velocity of the child given the velocity of the parent, then the acceleration is \f$ a_c = J a_p + dJ v_p \f$.
    /// The second term is null in linear mappings, otherwise it encodes the acceleration due to the change of mapping at constant parent velocity.
    /// For instance, in a rigid mapping with angular velocity\f$ w \f$,  the second term is $ w^(w^rel_pos) $
    virtual void computeAccFromMapping(const MechanicalParams* mparams, MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc );

    /// This method must be reimplemented by all mappings if they need to support composite accelerations
    virtual void computeAccFromMapping(const MechanicalParams* /* mparams */, OutDataVecDeriv& /* accOut */, const InDataVecDeriv& /* vIn */, const InDataVecDeriv& /* accIn */)
    {
    }

    /// Propagate positions and velocities to the output
    virtual void init();

    ///<TO REMOVE>  FF:why would we remove this, is there any alternative function ?
    // Useful ?
    /// Get the source (upper) model.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechFrom();

    /// Get the destination (lower, mapped) model.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechTo();

    //Create a matrix for mapped mechanical objects
    //If the two mechanical objects is identical, create a new stiffness matrix for this mapped objects
    //If the two mechanical objects is different, create a new interaction matrix
    virtual sofa::defaulttype::BaseMatrix* createMappedMatrix(const behavior::BaseMechanicalState* state1, const behavior::BaseMechanicalState* state2, func_createMappedMatrix);

    /// Disable the mapping to get the original coordinates of the mapped model.
    ///
    /// It is for instance used in RigidMapping to get the local coordinates of the object.
    virtual void disable();

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the input and output attributes and check
    /// if they are compatible with the input and output model types of this
    /// mapping.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        State<In>* stin = NULL;
        State<Out>* stout = NULL;

        std::string inPath, outPath;

        if (arg->getAttribute("input"))
            inPath = arg->getAttribute("input");
        else
            inPath = "@../";

        context->findLinkDest(stin, inPath, NULL);

        if (arg->getAttribute("output"))
            outPath = arg->getAttribute("output");
        else
            outPath = "@./";

        context->findLinkDest(stout, outPath, NULL);

        if (stin == NULL)
        {
//            This warning seems irrelevant, as it is raised multiple times while the creation works fine (Francois Faure, Feb. 2012)
//            context->serr << "Cannot create "<<className(obj)<<" as input model "<< inPath << " is missing or invalid." << context->sendl;
            return false;
        }

        if (stout == NULL)
        {
//            This warning seems irrelevant, as it is raised multiple times OutDataVecCoord& out, const InDataVecCoord& in)while the creation works fine (Francois Faure, Feb. 2012)
//            context->serr << "Cannot create "<<className(obj)<<" as output model "<< outPath << " is missing or invalid." << context->sendl;
            return false;
        }

        if (static_cast<BaseObject*>(stin) == static_cast<BaseObject*>(stout))
        {
            // we should refuse to create mappings with the same input and output model, which may happen if a State object is missing in the child node
            context->serr << "Creation of " << className(obj) << " mapping failed because the same object \"" << stin->getName() << "\" is linked as both input and output." << context->sendl;
            context->serr << "  Maybe a MechanicalObject should be added before this mapping." << context->sendl;
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

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const Mapping<TIn, TOut>* = NULL);


    template<class T>
    static std::string shortName(const T* ptr = NULL, objectmodel::BaseObjectDescription* arg = NULL)
    {
        std::string name = Inherit1::shortName(ptr, arg);
        sofa::helper::replaceAll(name, "Mapping", "Map");
        return name;
    }


protected:

    typedef BaseMapping::ForceMask ForceMask;
    /// keep an eye on the dof masks (if the dofs are Mechanical)
    ForceMask *maskFrom, *maskTo;

    /// Useful when the mapping is applied only on a subset of parent dofs.
    /// It is automatically called by applyJT.
    ///
    /// That way, we can optimize Jacobian sparsity.
    /// Every Dofs are inserted by default. The mappings using only a subset of dofs should only insert these dofs in the mask.
    virtual void updateForceMask();

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_CORE_MAPPING_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2dTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6dTypes, sofa::defaulttype::Vec6dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2dTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2dTypes, sofa::defaulttype::Rigid2dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2dTypes, sofa::defaulttype::ExtVec2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::ExtVec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec6dTypes >;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2fTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6fTypes, sofa::defaulttype::Vec6fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2fTypes, sofa::defaulttype::ExtVec2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2fTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2fTypes, sofa::defaulttype::Rigid2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec6fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2dTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec2fTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6dTypes, sofa::defaulttype::Vec6fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6fTypes, sofa::defaulttype::Vec6dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec6fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2dTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2fTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2dTypes, sofa::defaulttype::Rigid2fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid2fTypes, sofa::defaulttype::Rigid2dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3dTypes >;
extern template class SOFA_CORE_API Mapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::ExtVec3dTypes >;
#endif
#endif

extern template class SOFA_CORE_API Mapping< sofa::defaulttype::LaparoscopicRigidTypes, sofa::defaulttype::RigidTypes >;

#endif

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_MAPPING_H
