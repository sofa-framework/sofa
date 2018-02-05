/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_CORE_MULTIMAPPING_H
#define SOFA_CORE_MULTIMAPPING_H

#include <sofa/core/BaseMapping.h>
#include <sofa/core/State.h>
#include <sofa/core/core.h>
#include <sofa/core/VecId.h>

namespace sofa
{

namespace core
{

/**
 *  \brief Specialized interface to describe many to many mapping.
 *   All the input must have the same type, and all the output must have the same type. See also class Multi2Mapping.
 */

template <class TIn, class TOut>
class MultiMapping : public BaseMapping
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(MultiMapping,TIn,TOut), BaseMapping);

    /// Input Model Type
    typedef TIn In;
    /// Output Model Type
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
    /// Input Models container. New inputs are added through addInputModel(In* ).
    typedef MultiLink<MultiMapping<In,Out>, State< In >, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkFromModels;
    typedef typename LinkFromModels::Container VecFromModels;
    LinkFromModels fromModels;
    //helper::vector<State<In>*> fromModels;
    /// Output Model container. New outputs are added through addOutputModel( Ou* )
    typedef MultiLink<MultiMapping<In,Out>, State< Out >, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkToModels;
    typedef typename LinkToModels::Container VecToModels;
    LinkToModels toModels;
    //helper::vector<State<Out>*> toModels;

public:

    Data<bool> f_applyRestPosition; ///< @todo document this
protected:

    /// Constructor
    MultiMapping();
    /// Destructor
    virtual ~MultiMapping() {}
public:

    void addInputModel(BaseState* model, const std::string& path = "" );
    void addOutputModel(BaseState* model, const std::string& path = "" );

    /// Return the reference to fromModels.
    const VecFromModels& getFromModels();
    /// Return reference to toModels.
    const VecToModels& getToModels();

    /// Return a container of input models statically casted as BaseObject*
    helper::vector<BaseState* > getFrom() override;
    /// Return container of output model statically casted as BaseObject*.
    helper::vector<BaseState* > getTo() override;

    /// Get the source (upper) model.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechFrom() override;

    /// Get the destination (lower, mapped) model.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechTo() override;

    /// Apply ///
    /// Apply the mapping to position vectors.
    ///
    /// If the Mapping can be represented as a matrix \f$ J \f$, this method computes
    /// \f$ out = J in \f$
    virtual void apply (const MechanicalParams* mparams, MultiVecCoordId outPos, ConstMultiVecCoordId inPos ) override;

    /// This method must be reimplemented by all mappings.
    /// InPos and OutPos by default contains VecIds of type V_COORD.
    /// The size of InPos vector is the same as the number of fromModels.
    /// The size of OutPos vector is the same as the number of OutModels.
    virtual void apply(const MechanicalParams* mparams, const helper::vector<OutDataVecCoord*>& dataVecOutPos, const helper::vector<const InDataVecCoord*>& dataVecInPos) = 0;

    /// ApplyJ ///
    /// Apply the mapping to derived (velocity, displacement) vectors.
    /// \f$ out = J in \f$
    /// where \f$ J \f$ is the tangent operator (the linear approximation) of the mapping
    virtual void applyJ (const MechanicalParams* mparams, MultiVecDerivId outVel, ConstMultiVecDerivId inVel ) override;

    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJ(const MechanicalParams* mparams, const helper::vector<OutDataVecDeriv*>& dataVecOutVel, const helper::vector<const InDataVecDeriv*>& dataVecInVel) = 0;

    /// ApplyJT (Force)///
    /// Apply the reverse mapping to force vectors.
    /// \f$ out += J^t in \f$
    /// where \f$ J \f$ is the tangent operator (the linear approximation) of the mapping
    virtual void applyJT (const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce ) override;

    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJT(const MechanicalParams* mparams, const helper::vector<InDataVecDeriv*>& dataVecOutForce, const helper::vector<const OutDataVecDeriv*>& dataVecInForce) = 0;

    /// ApplyJT (Constraint)///
    virtual void applyJT(const ConstraintParams* cparams, MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst ) override
    {
        helper::vector<InDataMatrixDeriv*> matOutConst;
        getMatInDeriv(inConst, matOutConst);
        helper::vector<const OutDataMatrixDeriv*> matInConst;
        getConstMatOutDeriv(outConst, matInConst);

        this->applyJT(cparams, matOutConst, matInConst);
    }
    /// This method must be reimplemented by all mappings if they need to support constraints.
    virtual void applyJT( const ConstraintParams* /* cparams */, const helper::vector< InDataMatrixDeriv* >& /* dataMatOutConst */, const helper::vector< const OutDataMatrixDeriv* >& /* dataMatInConst */ )
    {
        serr << "This mapping does not support certain constraints since MultiMapping::applyJT( const ConstraintParams*, const helper::vector< InDataMatrixDeriv* >& , const helper::vector< const OutDataMatrixDeriv* >&  ) is not overloaded" << sendl;
    }

    /// computeAccFromMapping
    virtual void computeAccFromMapping(const MechanicalParams* mparams, MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc) override
    {
        helper::vector<OutDataVecDeriv*> vecOutAcc;
        getVecOutDeriv(outAcc, vecOutAcc);
        helper::vector<const InDataVecDeriv*> vecInVel;
        getConstVecInDeriv(inVel, vecInVel);
        helper::vector<const InDataVecDeriv*> vecInAcc;
        getConstVecInDeriv(inAcc, vecInAcc);

        this->computeAccFromMapping(mparams, vecOutAcc, vecInVel, vecInAcc);
    }
    /// This method must be reimplemented by all mappings if they need to support composite accelerations
    virtual void computeAccFromMapping(
        const MechanicalParams* /* mparams */, const helper::vector< OutDataVecDeriv*>&  /* dataVecOutAcc */,
        const helper::vector<const InDataVecDeriv*>& /* dataVecInVel */,
        const helper::vector<const InDataVecDeriv*>& /* dataVecInAcc */)
    {
    }


    virtual void init() override;

    ///<TO REMOVE>
    /// Apply the mapping to position and velocity vectors.
    ///
    /// This method call the internal apply(helper::vector<VecId>& InPos, helper::vector<VecId>& OutPos)
    /// and applyJ(helper::vector<VecId>& InDeriv, helper::vector<VecId>& OutDeriv) methods.
    //virtual void updateMapping();

    /// Disable the mapping to get the original coordinates of the mapped model.
    ///
    /// It is for instance used in RigidMapping to get the local coordinates of the object.
    virtual void disable() override;



    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const MultiMapping<TIn, TOut>* = NULL);

    template<class T>
    static std::string shortName(const T* ptr = NULL, objectmodel::BaseObjectDescription* arg = NULL)
    {
        std::string name = Inherit1::shortName(ptr, arg);
        sofa::helper::replaceAll(name, "Mapping", "Map");
        return name;
    }

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes and check
    /// if they are compatible with the input and output models types of this
    /// mapping.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        std::string input  = arg->getAttribute("input","");
        if( input.empty() || !LinkFromModels::CheckPaths( input, context ) ) return false;
        std::string output = arg->getAttribute("output","");
        if( output.empty() || !LinkToModels::CheckPaths( output, context ) ) return false;
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
            obj->parse(arg);
        }

        return obj;
    }

protected:

    void getVecInCoord     (const MultiVecCoordId id,         helper::vector<      InDataVecCoord* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].write()); }
    void getConstVecInCoord(const ConstMultiVecCoordId id,    helper::vector<const InDataVecCoord* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].read());  }
    void getVecInDeriv      (const MultiVecDerivId id,         helper::vector<      InDataVecDeriv* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].write()); }
    void getConstVecInDeriv (const ConstMultiVecDerivId id,    helper::vector<const InDataVecDeriv* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].read());  }
    void getMatInDeriv      (const MultiMatrixDerivId id,      helper::vector<      InDataMatrixDeriv* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].write()); }
    void getConstMatInDeriv (const ConstMultiMatrixDerivId id, helper::vector<const InDataMatrixDeriv* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].read());  }

    void getVecOutCoord     (const MultiVecCoordId id,         helper::vector<      OutDataVecCoord* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].write());    }
    void getConstVecOutCoord(const ConstMultiVecCoordId id,    helper::vector<const OutDataVecCoord* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].read());     }
    void getVecOutDeriv     (const MultiVecDerivId id,         helper::vector<      OutDataVecDeriv* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].write());    }
    void getConstVecOutDeriv(const ConstMultiVecDerivId id,    helper::vector<const OutDataVecDeriv* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].read());     }
    void getMatOutDeriv     (const MultiMatrixDerivId id,      helper::vector<      OutDataMatrixDeriv* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].write()); }
    void getConstMatOutDeriv(const ConstMultiMatrixDerivId id, helper::vector<const OutDataMatrixDeriv* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].read());  }

    /// Useful when the mapping is applied only on a subset of parent dofs.
    /// It is automatically called by applyJT.
    ///
    /// That way, we can optimize Jacobian sparsity.
    /// Every Dofs are inserted by default. The mappings using only a subset of dofs should only insert these dofs in the mask.
    virtual void updateForceMask() override;

    /// keep pointers on the masks
    helper::vector<helper::StateMask*> maskFrom, maskTo;

};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_CORE_MULTIMAPPING_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec1dTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec2dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec6dTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3dTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3dTypes, defaulttype::Vec6dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3dTypes, defaulttype::Rigid3dTypes >;
#endif


#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec1fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec2fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec6fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3fTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3fTypes, defaulttype::Vec6fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3fTypes, defaulttype::Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec1dTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec1fTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3dTypes, defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Vec3fTypes, defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3dTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3fTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3dTypes, defaulttype::Vec6fTypes >;
extern template class SOFA_CORE_API MultiMapping< defaulttype::Rigid3fTypes, defaulttype::Vec6dTypes >;
#endif
#endif

#endif


} // namespace core

} // namespace sofa

#endif

