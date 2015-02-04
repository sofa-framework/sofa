/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
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
    helper::vector<BaseState* > getFrom();
    /// Return container of output model statically casted as BaseObject*.
    helper::vector<BaseState* > getTo();

    /// Get the source (upper) model.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechFrom();

    /// Get the destination (lower, mapped) model.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechTo();

    /// Apply ///
    /// Apply the mapping to position vectors.
    ///
    /// If the Mapping can be represented as a matrix \f$ J \f$, this method computes
    /// \f$ out = J in \f$
    virtual void apply (const MechanicalParams* mparams /* PARAMS FIRST  = MechanicalParams::defaultInstance()*/, MultiVecCoordId outPos, ConstMultiVecCoordId inPos );

    /// This method must be reimplemented by all mappings.
    /// InPos and OutPos by default contains VecIds of type V_COORD.
    /// The size of InPos vector is the same as the number of fromModels.
    /// The size of OutPos vector is the same as the number of OutModels.
    virtual void apply(const MechanicalParams* mparams /* PARAMS FIRST */, const helper::vector<OutDataVecCoord*>& dataVecOutPos, const helper::vector<const InDataVecCoord*>& dataVecInPos)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        //Not optimized at all...
        helper::vector<OutVecCoord*> vecOutPos;
        for(unsigned int i=0; i<dataVecOutPos.size(); i++)
            vecOutPos.push_back(dataVecOutPos[i]->beginEdit(mparams));

        helper::vector<const InVecCoord*> vecInPos;
        for(unsigned int i=0; i<dataVecInPos.size(); i++)
            vecInPos.push_back(&dataVecInPos[i]->getValue(mparams));

        this->apply(vecOutPos, vecInPos);

        //Really Not optimized at all...
        for(unsigned int i=0; i<dataVecOutPos.size(); i++)
            dataVecOutPos[i]->endEdit(mparams);

    }
    /// Compat Method
    /// @deprecated
    virtual void apply(const helper::vector<OutVecCoord*>&  /* outPos */, const helper::vector<const InVecCoord*>& /* inPos */) {}
#endif //SOFA_DEPRECATE_OLD_API

    /// ApplyJ ///
    /// Apply the mapping to derived (velocity, displacement) vectors.
    /// \f$ out = J in \f$
    /// where \f$ J \f$ is the tangent operator (the linear approximation) of the mapping
    virtual void applyJ (const MechanicalParams* mparams /* PARAMS FIRST  = MechanicalParams::defaultInstance()*/, MultiVecDerivId outVel, ConstMultiVecDerivId inVel );

    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJ(const MechanicalParams* mparams /* PARAMS FIRST */, const helper::vector<OutDataVecDeriv*>& dataVecOutVel, const helper::vector<const InDataVecDeriv*>& dataVecInVel)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        //Not optimized at all...
        helper::vector<OutVecDeriv*> vecOutVel;
        for(unsigned int i=0; i<dataVecOutVel.size(); i++)
            vecOutVel.push_back(dataVecOutVel[i]->beginEdit(mparams));

        helper::vector<const InVecDeriv*> vecInVel;
        for(unsigned int i=0; i<dataVecInVel.size(); i++)
            vecInVel.push_back(&dataVecInVel[i]->getValue(mparams));

        this->applyJ(vecOutVel, vecInVel);

        //Really Not optimized at all...
        for(unsigned int i=0; i<dataVecOutVel.size(); i++)
            dataVecOutVel[i]->endEdit(mparams);

    }
    /// Compat Method
    /// @deprecated
    virtual void applyJ(const helper::vector< OutVecDeriv*>& /* outDeriv */, const helper::vector<const InVecDeriv*>& /* inDeriv */) {}
#endif //SOFA_DEPRECATE_OLD_API

    /// ApplyJT (Force)///
    /// Apply the reverse mapping to force vectors.
    /// \f$ out += J^t in \f$
    /// where \f$ J \f$ is the tangent operator (the linear approximation) of the mapping
    virtual void applyJT (const MechanicalParams* mparams /* PARAMS FIRST  = MechanicalParams::defaultInstance()*/, MultiVecDerivId inForce, ConstMultiVecDerivId outForce );

    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJT(const MechanicalParams* mparams /* PARAMS FIRST */, const helper::vector<InDataVecDeriv*>& dataVecOutForce, const helper::vector<const OutDataVecDeriv*>& dataVecInForce)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        //Not optimized at all...
        helper::vector<InVecDeriv*> vecOutForce;
        for(unsigned int i=0; i<dataVecOutForce.size(); i++)
            vecOutForce.push_back(dataVecOutForce[i]->beginEdit(mparams));

        helper::vector<const OutVecDeriv*> vecInForce;
        for(unsigned int i=0; i<dataVecInForce.size(); i++)
            vecInForce.push_back(&dataVecInForce[i]->getValue(mparams));

        this->applyJT(vecOutForce, vecInForce);

        //Really Not optimized at all...
        for(unsigned int i=0; i<dataVecOutForce.size(); i++)
            dataVecOutForce[i]->endEdit(mparams);

    }
    /// Compat Method
    /// @deprecated
    virtual void applyJT(const helper::vector< InVecDeriv*>& /* outDeriv */, const helper::vector<const OutVecDeriv*>& /* inDeriv */) {}
#endif //SOFA_DEPRECATE_OLD_API

    /// ApplyJT (Constraint)///
    virtual void applyJT(const ConstraintParams* cparams /* PARAMS FIRST */, MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst )
    {
        helper::vector<InDataMatrixDeriv*> matOutConst;
        getMatInDeriv(inConst, matOutConst);
        helper::vector<const OutDataMatrixDeriv*> matInConst;
        getConstMatOutDeriv(outConst, matInConst);

        this->applyJT(cparams /* PARAMS FIRST */, matOutConst, matInConst);
    }
    /// This method must be reimplemented by all mappings if they need to support constraints.
    virtual void applyJT( const ConstraintParams* cparams /* PARAMS FIRST */, const helper::vector< InDataMatrixDeriv* >& dataMatOutConst, const helper::vector< const OutDataMatrixDeriv* >& dataMatInConst )
#ifdef SOFA_DEPRECATE_OLD_API
    {
        serr << "This mapping does not support constraints since MultiMapping::applyJT( const ConstraintParams*, const helper::vector< InDataMatrixDeriv* >& , const helper::vector< const OutDataMatrixDeriv* >&  ) is not overloaded" << sendl;
    }
#else
    {
        //Not optimized at all...
        helper::vector<InMatrixDeriv*> matOutConst;
        for(unsigned int i=0; i<dataMatOutConst.size(); i++)
            matOutConst.push_back(dataMatOutConst[i]->beginEdit(cparams));

        helper::vector<const OutMatrixDeriv*> matInConst;
        for(unsigned int i=0; i<dataMatInConst.size(); i++)
            matInConst.push_back(&dataMatInConst[i]->getValue(cparams));

        this->applyJT(matOutConst, matInConst);

        //Really Not optimized at all...
        for(unsigned int i=0; i<dataMatOutConst.size(); i++)
            dataMatOutConst[i]->endEdit(cparams);
    }
    /// Compat Method
    /// @deprecated
    virtual void applyJT( const helper::vector< InMatrixDeriv* >& /*outConst*/ , const helper::vector< const OutMatrixDeriv* >& /*inConst*/ )
    {
//        serr << "This mapping does not support constraints since MultiMapping::applyJT( const helper::vector< InMatrixDeriv* >& , const helper::vector< const OutMatrixDeriv* >& ) is not overloaded." << sendl;
    }
#endif //SOFA_DEPRECATE_OLD_API

    /// computeAccFromMapping
    virtual void computeAccFromMapping(const MechanicalParams* mparams /* PARAMS FIRST  = MechanicalParams::defaultInstance()*/, MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc )
    {
        helper::vector<OutDataVecDeriv*> vecOutAcc;
        getVecOutDeriv(outAcc, vecOutAcc);
        helper::vector<const InDataVecDeriv*> vecInVel;
        getConstVecInDeriv(inVel, vecInVel);
        helper::vector<const InDataVecDeriv*> vecInAcc;
        getConstVecInDeriv(inAcc, vecInAcc);

        this->computeAccFromMapping(mparams /* PARAMS FIRST */, vecOutAcc, vecInVel, vecInAcc);
    }
    /// This method must be reimplemented by all mappings if they need to support composite accelerations
    virtual void computeAccFromMapping(
        const MechanicalParams* mparams /* PARAMS FIRST */, const helper::vector< OutDataVecDeriv*>& dataVecOutAcc,
        const helper::vector<const InDataVecDeriv*>& dataVecInVel,
        const helper::vector<const InDataVecDeriv*>& dataVecInAcc)
#ifdef SOFA_DEPRECATE_OLD_API
    {
    }
#else
    {
        //Not optimized at all...
        helper::vector<OutVecDeriv*> vecOutAcc;
        for(unsigned int i=0; i<dataVecOutAcc.size(); i++)
            vecOutAcc.push_back(dataVecOutAcc[i]->beginEdit(mparams));

        helper::vector<const InVecDeriv*> vecInVel;
        for(unsigned int i=0; i<dataVecInVel.size(); i++)
            vecInVel.push_back(&dataVecInVel[i]->getValue(mparams));
        helper::vector<const InVecDeriv*> vecInAcc;
        for(unsigned int i=0; i<dataVecInAcc.size(); i++)
            vecInAcc.push_back(&dataVecInAcc[i]->getValue(mparams));

        this->computeAccFromMapping(vecOutAcc, vecInVel, vecInAcc);

        //Really Not optimized at all...
        for(unsigned int i=0; i<dataVecOutAcc.size(); i++)
            dataVecOutAcc[i]->endEdit(mparams);
    }
    /// Compat Method
    /// @deprecated
    virtual void computeAccFromMapping( const helper::vector<typename Out::VecDeriv*>& /*outDx*/,
            const helper::vector<const typename In::VecDeriv*>& /*inV */,
            const helper::vector<const typename In::VecDeriv*>& /*inDx */ )
    {
    }
#endif //SOFA_DEPRECATE_OLD_API

    virtual void init();

    ///<TO REMOVE>
    /// Apply the mapping to position and velocity vectors.
    ///
    /// This method call the internal apply(helper::vector<VecId>& InPos, helper::vector<VecId>& OutPos)
    /// and applyJ(helper::vector<VecId>& InDeriv, helper::vector<VecId>& OutDeriv) methods.
    //virtual void updateMapping();

    /// Disable the mapping to get the original coordinates of the mapped model.
    ///
    /// It is for instance used in RigidMapping to get the local coordinates of the object.
    virtual void disable();



    virtual std::string getTemplateName() const
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
        std::string output = arg->getAttribute("output","");
        if (input.empty() || !LinkFromModels::CheckPaths(input, context)) {
            std::cerr << "[" << arg->getAttribute("name") << "(" << arg->getAttribute("type") << ")]: " << "bad input" << std::endl;
            return false;
        }
        if (output.empty() || !LinkToModels::CheckPaths(output, context)) {
            std::cerr << "[" << arg->getAttribute("name") << "(" << arg->getAttribute("type") << ")]: " << "bad output" << std::endl;
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

};

} // namespace core

} // namespace sofa

#endif

