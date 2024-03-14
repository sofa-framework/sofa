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
#include <sofa/core/config.h>
#include <sofa/helper/fwd.h>
#include <sofa/core/State.h>

namespace sofa::core
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
    //type::vector<State<In>*> fromModels;
    /// Output Model container. New outputs are added through addOutputModel( Ou* )
    typedef MultiLink<MultiMapping<In,Out>, State< Out >, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkToModels;
    typedef typename LinkToModels::Container VecToModels;
    LinkToModels toModels;
    //type::vector<State<Out>*> toModels;

public:

    Data<bool> f_applyRestPosition; ///< @todo document this
protected:

    /// Constructor
    MultiMapping();
    /// Destructor
    ~MultiMapping() override {}
public:

    void addInputModel(BaseState* model, const std::string& path = "" );
    void addOutputModel(BaseState* model, const std::string& path = "" );

    /// Return the reference to fromModels.
    const VecFromModels& getFromModels();
    /// Return reference to toModels.
    const VecToModels& getToModels();

    /// Return a container of input models statically casted as BaseObject*
    type::vector<BaseState* > getFrom() override;
    /// Return container of output model statically casted as BaseObject*.
    type::vector<BaseState* > getTo() override;

    /// Get the source (upper) model.
    virtual type::vector<behavior::BaseMechanicalState*> getMechFrom() override;

    /// Get the destination (lower, mapped) model.
    virtual type::vector<behavior::BaseMechanicalState*> getMechTo() override;

    /// Apply ///
    /// Apply the mapping to position vectors.
    ///
    /// If the Mapping can be represented as a matrix \f$ J \f$, this method computes
    /// \f$ out = J in \f$
    void apply (const MechanicalParams* mparams, MultiVecCoordId outPos, ConstMultiVecCoordId inPos ) override;

    /// This method must be reimplemented by all mappings.
    /// InPos and OutPos by default contains VecIds of type V_COORD.
    /// The size of InPos vector is the same as the number of fromModels.
    /// The size of OutPos vector is the same as the number of OutModels.
    virtual void apply(const MechanicalParams* mparams, const type::vector<OutDataVecCoord*>& dataVecOutPos, const type::vector<const InDataVecCoord*>& dataVecInPos) = 0;

    /// ApplyJ ///
    /// Apply the mapping to derived (velocity, displacement) vectors.
    /// \f$ out = J in \f$
    /// where \f$ J \f$ is the tangent operator (the linear approximation) of the mapping
    void applyJ (const MechanicalParams* mparams, MultiVecDerivId outVel, ConstMultiVecDerivId inVel ) override;

    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJ(const MechanicalParams* mparams, const type::vector<OutDataVecDeriv*>& dataVecOutVel, const type::vector<const InDataVecDeriv*>& dataVecInVel) = 0;

    /// ApplyJT (Force)///
    /// Apply the reverse mapping to force vectors.
    /// \f$ out += J^t in \f$
    /// where \f$ J \f$ is the tangent operator (the linear approximation) of the mapping
    void applyJT (const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce ) override;

    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJT(const MechanicalParams* mparams, const type::vector<InDataVecDeriv*>& dataVecOutForce, const type::vector<const OutDataVecDeriv*>& dataVecInForce) = 0;

    /// ApplyJT (Constraint)///
    void applyJT(const ConstraintParams* cparams, MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst ) override
    {
        type::vector<InDataMatrixDeriv*> matOutConst;
        getMatInDeriv(inConst, matOutConst);
        type::vector<const OutDataMatrixDeriv*> matInConst;
        getConstMatOutDeriv(outConst, matInConst);

        this->applyJT(cparams, matOutConst, matInConst);
    }
    /// This method must be reimplemented by all mappings if they need to support constraints.
    virtual void applyJT( const ConstraintParams* /* cparams */, const type::vector< InDataMatrixDeriv* >& /* dataMatOutConst */, const type::vector< const OutDataMatrixDeriv* >& /* dataMatInConst */ )
    {
        msg_error() << "This mapping does not support certain constraints since MultiMapping::applyJT( const ConstraintParams*, const type::vector< InDataMatrixDeriv* >& , const type::vector< const OutDataMatrixDeriv* >&  ) is not overloaded";
    }

    /// computeAccFromMapping
    void computeAccFromMapping(const MechanicalParams* mparams, MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc) override
    {
        type::vector<OutDataVecDeriv*> vecOutAcc;
        getVecOutDeriv(outAcc, vecOutAcc);
        type::vector<const InDataVecDeriv*> vecInVel;
        getConstVecInDeriv(inVel, vecInVel);
        type::vector<const InDataVecDeriv*> vecInAcc;
        getConstVecInDeriv(inAcc, vecInAcc);

        this->computeAccFromMapping(mparams, vecOutAcc, vecInVel, vecInAcc);
    }
    /// This method must be reimplemented by all mappings if they need to support composite accelerations
    virtual void computeAccFromMapping(
        const MechanicalParams* /* mparams */, const type::vector< OutDataVecDeriv*>&  /* dataVecOutAcc */,
        const type::vector<const InDataVecDeriv*>& /* dataVecInVel */,
        const type::vector<const InDataVecDeriv*>& /* dataVecInAcc */)
    {
    }


    void init() override;

    ///<TO REMOVE>
    /// Apply the mapping to position and velocity vectors.
    ///
    /// This method call the internal apply(type::vector<VecId>& InPos, type::vector<VecId>& OutPos)
    /// and applyJ(type::vector<VecId>& InDeriv, type::vector<VecId>& OutDeriv) methods.
    //virtual void updateMapping();

    /// Disable the mapping to get the original coordinates of the mapped model.
    ///
    /// It is for instance used in RigidMapping to get the local coordinates of the object.
    void disable() override;


    template<class T>
    static std::string shortName(const T* ptr = nullptr, objectmodel::BaseObjectDescription* arg = nullptr)
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
        if (input.empty()) {
            arg->logError("The 'input' data attribute is empty. It should contain a valid path "
                          "to one or more mechanical states of type '" + std::string(TIn::Name()) + "'.");
            return false;

        } else if (!PathResolver::CheckPaths(context, LinkFromModels::DestType::GetClass(), input)) {
            arg->logError("The 'input' data attribute does not contain a valid path to one or more mechanical "
                          "states of type '" + std::string(TIn::Name()) + "'.");
            return false;
        }

        std::string output = arg->getAttribute("output","");
        if (output.empty()) {
            arg->logError("The 'output' data attribute is empty. It should contain a valid path "
                          "to one or more mechanical states. of type '" + std::string(TOut::Name()) + "'.");
            return false;
        } else if (!PathResolver::CheckPaths(context, LinkToModels::DestType::GetClass(), output)) {
            arg->logError("The 'output' data attribute does not contain a valid path to one or more mechanical "
                          "states of type '" + std::string(TOut::Name()) + "'.");
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
            obj->parse(arg);
        }

        return obj;
    }

protected:

    void getVecInCoord     (const MultiVecCoordId id,         type::vector<      InDataVecCoord* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].write()); }
    void getConstVecInCoord(const ConstMultiVecCoordId id,    type::vector<const InDataVecCoord* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].read());  }
    void getVecInDeriv      (const MultiVecDerivId id,         type::vector<      InDataVecDeriv* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].write()); }
    void getConstVecInDeriv (const ConstMultiVecDerivId id,    type::vector<const InDataVecDeriv* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].read());  }
    void getMatInDeriv      (const MultiMatrixDerivId id,      type::vector<      InDataMatrixDeriv* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].write()); }
    void getConstMatInDeriv (const ConstMultiMatrixDerivId id, type::vector<const InDataMatrixDeriv* > &v) const
    {   for (unsigned int i=0; i<fromModels.size(); ++i) v.push_back(id[fromModels.get(i)].read());  }

    void getVecOutCoord     (const MultiVecCoordId id,         type::vector<      OutDataVecCoord* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].write());    }
    void getConstVecOutCoord(const ConstMultiVecCoordId id,    type::vector<const OutDataVecCoord* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].read());     }
    void getVecOutDeriv     (const MultiVecDerivId id,         type::vector<      OutDataVecDeriv* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].write());    }
    void getConstVecOutDeriv(const ConstMultiVecDerivId id,    type::vector<const OutDataVecDeriv* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].read());     }
    void getMatOutDeriv     (const MultiMatrixDerivId id,      type::vector<      OutDataMatrixDeriv* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].write()); }
    void getConstMatOutDeriv(const ConstMultiMatrixDerivId id, type::vector<const OutDataMatrixDeriv* > &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels.get(i)].read());  }
};


#if !defined(SOFA_CORE_MULTIMAPPING_CPP)

extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types >;
extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec1Types >;
extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types >;
extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types >;
extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec2Types >;
extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types >;
extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Vec6Types, sofa::defaulttype::Vec1Types >;
extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec1Types >;
extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec3Types >;
extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec6Types >;
extern template class SOFA_CORE_API MultiMapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Rigid3Types >;

#endif
} // namespace sofa::core

