/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_MULTI2MAPPING_H
#define SOFA_CORE_MULTI2MAPPING_H

#include <sofa/core/BaseMapping.h>
#include <sofa/core/core.h>
#include <sofa/core/VecId.h>


namespace sofa
{

namespace core
{

/**
 *  \brief Specialized interface to describe many to many mapping.
 *   The inputs can be of two different types, while all the outputs must be of the same type.
 */

template <class TIn1, class TIn2, class TOut>
class Multi2Mapping : public BaseMapping
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE3(Multi2Mapping,TIn1, TIn2,TOut), BaseMapping);

    /// Input Model Type
    typedef TIn1 In1;
    typedef TIn2 In2;
    /// Output Model Type
    typedef TOut Out;

    typedef typename In1::VecCoord In1VecCoord;
    typedef typename In1::VecDeriv In1VecDeriv;
    typedef typename In1::MatrixDeriv In1MatrixDeriv;
    typedef Data<In1VecCoord> In1DataVecCoord;
    typedef Data<In1VecDeriv> In1DataVecDeriv;
    typedef Data<In1MatrixDeriv> In1DataMatrixDeriv;
    typedef typename In2::VecCoord In2VecCoord;
    typedef typename In2::VecDeriv In2VecDeriv;
    typedef typename In2::MatrixDeriv In2MatrixDeriv;
    typedef Data<In2VecCoord> In2DataVecCoord;
    typedef Data<In2VecDeriv> In2DataVecDeriv;
    typedef Data<In2MatrixDeriv> In2DataMatrixDeriv;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

    typedef MultiLink<Multi2Mapping<In1,In2,Out>, State< In1 >, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkFromModels1;
    typedef typename LinkFromModels1::Container VecFromModels1;
    typedef MultiLink<Multi2Mapping<In1,In2,Out>, State< In2 >, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkFromModels2;
    typedef typename LinkFromModels2::Container VecFromModels2;
    typedef MultiLink<Multi2Mapping<In1,In2,Out>, State< Out >, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkToModels;
    typedef typename LinkToModels::Container VecToModels;

protected:
    /// Input Models container. New inputs are added through addInputModel(In* ).
    LinkFromModels1 fromModels1;
    LinkFromModels2 fromModels2;
    LinkToModels toModels;

public:

    Data<bool> f_applyRestPosition; ///< @todo document this
protected:

    /// Constructor
    Multi2Mapping();
    /// Destructor
    virtual ~Multi2Mapping() {}
public:

    virtual void addInputModel1(State<In1>*, const std::string& path = "");
    virtual void addInputModel2(State<In2>*, const std::string& path = "");
    virtual void addOutputModel(State<Out>*, const std::string& path = "");

    /// Return the reference to fromModels (In1).
    const VecFromModels1& getFromModels1();
    /// Return the reference to fromModels (In2).
    const VecFromModels2& getFromModels2();
    /// Return reference to toModels.
    const VecToModels& getToModels();

    /// Return a container of input models statically casted as BaseObject*
    helper::vector<BaseState*> getFrom() override;
    /// Return container of output model statically casted as BaseObject*.
    helper::vector<BaseState*> getTo() override;

    /// Get the source (upper) model.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechFrom() override;

    /// Get the destination (lower, mapped) model.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechTo() override;

    /// Apply ///
    /// Apply the mapping to position vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    virtual void apply (const MechanicalParams* mparams, MultiVecCoordId outPos, ConstMultiVecCoordId inPos ) override
    {
        helper::vector<OutDataVecCoord*> vecOutPos;
        getVecOutCoord(outPos, vecOutPos);
        helper::vector<const In1DataVecCoord*> vecIn1Pos;
        getConstVecIn1Coord(inPos, vecIn1Pos);
        helper::vector<const In2DataVecCoord*> vecIn2Pos;
        getConstVecIn2Coord(inPos, vecIn2Pos);

        this->apply(mparams, vecOutPos, vecIn1Pos, vecIn2Pos);

#ifdef SOFA_USE_MASK
        this->m_forceMaskNewStep = true;
#endif
    }
    /// This method must be reimplemented by all mappings.
    /// InPos and OutPos by default contains VecIds of type V_COORD.
    /// The size of InPos vector is the same as the number of fromModels.
    /// The size of OutPos vector is the same as the number of OutModels.
    virtual void apply(
        const MechanicalParams* mparams, const helper::vector<OutDataVecCoord*>& dataVecOutPos,
        const helper::vector<const In1DataVecCoord*>& dataVecIn1Pos ,
        const helper::vector<const In2DataVecCoord*>& dataVecIn2Pos) = 0;

    /// ApplyJ ///
    /// This method computes
    /// $ out = J in $
    /// where J is the tangent operator (the linear approximation) of the mapping
    virtual void applyJ (const MechanicalParams* mparams, MultiVecDerivId outVel, ConstMultiVecDerivId inVel ) override
    {
        helper::vector<OutDataVecDeriv*> vecOutVel;
        getVecOutDeriv(outVel, vecOutVel);
        helper::vector<const In1DataVecDeriv*> vecIn1Vel;
        getConstVecIn1Deriv(inVel, vecIn1Vel);
        helper::vector<const In2DataVecDeriv*> vecIn2Vel;
        getConstVecIn2Deriv(inVel, vecIn2Vel);
        this->applyJ(mparams, vecOutVel, vecIn1Vel, vecIn2Vel);
    }
    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJ(
        const MechanicalParams* mparams, const helper::vector< OutDataVecDeriv*>& dataVecOutVel,
        const helper::vector<const In1DataVecDeriv*>& dataVecIn1Vel,
        const helper::vector<const In2DataVecDeriv*>& dataVecIn2Vel)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        //Not optimized at all...
        helper::vector<OutVecDeriv*> vecOutVel;
        for(unsigned int i=0; i<dataVecOutVel.size(); i++)
            vecOutVel.push_back(dataVecOutVel[i]->beginEdit(mparams));

        helper::vector<const In1VecDeriv*> vecIn1Vel;
        for(unsigned int i=0; i<dataVecIn1Vel.size(); i++)
            vecIn1Vel.push_back(&dataVecIn1Vel[i]->getValue(mparams));
        helper::vector<const In2VecDeriv*> vecIn2Vel;
        for(unsigned int i=0; i<dataVecIn2Vel.size(); i++)
            vecIn2Vel.push_back(&dataVecIn2Vel[i]->getValue(mparams));
        this->applyJ(vecOutVel, vecIn1Vel, vecIn2Vel);

        //Really Not optimized at all...
        for(unsigned int i=0; i<dataVecOutVel.size(); i++)
            dataVecOutVel[i]->endEdit(mparams);
    }
    /// Compat Method
    /// @deprecated
    virtual void applyJ(const helper::vector< OutVecDeriv*>& /* outDeriv */,
            const helper::vector<const In1VecDeriv*>& /* inDeriv1 */,
            const helper::vector<const In2VecDeriv*>& /* inDeriv2 */) {}
#endif //SOFA_DEPRECATE_OLD_API

    /// ApplyJT (Force)///
    /// Apply the mapping to Force vectors.
    virtual void applyJT (const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce ) override
    {
        helper::vector<In1DataVecDeriv*> vecOut1Force;
        getVecIn1Deriv(inForce, vecOut1Force);
        helper::vector<In2DataVecDeriv*> vecOut2Force;
        getVecIn2Deriv(inForce, vecOut2Force);

        helper::vector<const OutDataVecDeriv*> vecInForce;
        getConstVecOutDeriv(outForce, vecInForce);
        this->applyJT(mparams, vecOut1Force, vecOut2Force, vecInForce);

#ifdef SOFA_USE_MASK
        if( this->m_forceMaskNewStep )
        {
            this->m_forceMaskNewStep = false;
            updateForceMask();
        }
#endif
    }
    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJT(
        const MechanicalParams* mparams, const helper::vector< In1DataVecDeriv*>& dataVecOut1Force,
        const helper::vector< In2DataVecDeriv*>& dataVecOut2Force,
        const helper::vector<const OutDataVecDeriv*>& dataVecInForce) = 0;

    /// ApplyJT (Constraint)///
    virtual void applyJT(const ConstraintParams* cparams, MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst ) override
    {
        helper::vector<In1DataMatrixDeriv*> matOut1Const;
        getMatIn1Deriv(inConst, matOut1Const);
        helper::vector<In2DataMatrixDeriv*> matOut2Const;
        getMatIn2Deriv(inConst, matOut2Const);

        helper::vector<const OutDataMatrixDeriv*> matInConst;
        getConstMatOutDeriv(outConst, matInConst);
        this->applyJT(cparams, matOut1Const, matOut2Const, matInConst);
    }
    /// This method must be reimplemented by all mappings if they need to support constraints.
    virtual void applyJT(
        const ConstraintParams* /* cparams */, const helper::vector< In1DataMatrixDeriv*>& /* dataMatOut1Const */ ,
        const helper::vector< In2DataMatrixDeriv*>&  /* dataMatOut2Const */,
        const helper::vector<const OutDataMatrixDeriv*>& /* dataMatInConst */)
    {
        serr << "This mapping does not support constraint because Multi2Mapping::applyJT(const ConstraintParams*, const helper::vector< In1DataMatrixDeriv*>&, const helper::vector< In2DataMatrixDeriv*>&, const helper::vector<const OutDataMatrixDeriv*>&) is not overloaded." << sendl;
    }

    /// computeAccFromMapping
    virtual void computeAccFromMapping(const MechanicalParams* mparams, MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc ) override
    {
        helper::vector<OutDataVecDeriv*> vecOutAcc;
        getVecOutDeriv(outAcc, vecOutAcc);

        helper::vector<const In1DataVecDeriv*> vecIn1Vel;
        getConstVecIn1Deriv(inVel, vecIn1Vel);
        helper::vector<const In1DataVecDeriv*> vecIn1Acc;
        getConstVecIn1Deriv(inAcc, vecIn1Acc);

        helper::vector<const In2DataVecDeriv*> vecIn2Vel;
        getConstVecIn2Deriv(inVel, vecIn2Vel);
        helper::vector<const In2DataVecDeriv*> vecIn2Acc;
        getConstVecIn2Deriv(inAcc, vecIn2Acc);

        this->computeAccFromMapping(mparams, vecOutAcc, vecIn1Vel, vecIn2Vel,vecIn1Acc, vecIn2Acc);
    }
    /// This method must be reimplemented by all mappings if they need to support composite accelerations
    virtual void computeAccFromMapping(
        const MechanicalParams* /* mparams */, const helper::vector< OutDataVecDeriv*>& /* dataVecOutAcc */,
        const helper::vector<const In1DataVecDeriv*>& /* dataVecIn1Vel */,
        const helper::vector<const In2DataVecDeriv*>& /* dataVecIn2Vel */,
        const helper::vector<const In1DataVecDeriv*>& /* dataVecIn1Acc */,
        const helper::vector<const In2DataVecDeriv*>& /* dataVecIn2Acc */)
    {
    }

    virtual void init() override;

    /// Disable the mapping to get the original coordinates of the mapped model.
    ///
    /// It is for instance used in RigidMapping to get the local coordinates of the object.
    virtual void disable() override;

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const Multi2Mapping<TIn1,TIn2, TOut>* = NULL);

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes and check
    /// if they are compatible with the input and output models types of this
    /// mapping.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        std::string input1 = arg->getAttribute("input1","");
        std::string input2 = arg->getAttribute("input2","");
        std::string output = arg->getAttribute("output","");
        if (!input1.empty() && !LinkFromModels1::CheckPaths(input1, context))
            return false;
        if (!input2.empty() && !LinkFromModels2::CheckPaths(input2, context))
            return false;
        if (output.empty() || !LinkToModels::CheckPaths(output, context))
            return false;

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
            obj->parse(arg);

        return obj;
    }


protected:
    void getVecIn1Coord     (const MultiVecCoordId id,         helper::vector<      In1DataVecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].write()); }
    void getConstVecIn1Coord(const ConstMultiVecCoordId id,    helper::vector<const In1DataVecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].read());  }
    void getVecIn1Deriv     (const MultiVecDerivId id,         helper::vector<      In1DataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].write()); }
    void getConstVecIn1Deriv(const ConstMultiVecDerivId id,    helper::vector<const In1DataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].read());  }
    void getMatIn1Deriv     (const MultiMatrixDerivId id,      helper::vector<      In1DataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].write()); }
    void getConstMatIn1Deriv(const ConstMultiMatrixDerivId id, helper::vector<const In1DataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].read());  }

    void getVecIn2Coord     (const MultiVecCoordId id,         helper::vector<      In2DataVecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].write()); }
    void getConstVecIn2Coord(const ConstMultiVecCoordId id,    helper::vector<const In2DataVecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].read());  }
    void getVecIn2Deriv     (const MultiVecDerivId id,         helper::vector<      In2DataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].write()); }
    void getConstVecIn2Deriv(const ConstMultiVecDerivId id,    helper::vector<const In2DataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].read());  }
    void getMatIn2Deriv     (const MultiMatrixDerivId id,      helper::vector<      In2DataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].write()); }
    void getConstMatIn2Deriv(const ConstMultiMatrixDerivId id, helper::vector<const In2DataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].read());  }

    void getVecOutCoord     (const MultiVecCoordId id,         helper::vector<      OutDataVecCoord*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].write());      }
    void getConstVecOutCoord(const ConstMultiVecCoordId id,    helper::vector<const OutDataVecCoord*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].read());       }
    void getVecOutDeriv     (const MultiVecDerivId id,         helper::vector<      OutDataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].write());      }
    void getConstVecOutDeriv(const ConstMultiVecDerivId id,    helper::vector<const OutDataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].read());       }
    void getMatOutDeriv     (const MultiMatrixDerivId id,      helper::vector<      OutDataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].write());      }
    void getConstMatOutDeriv(const ConstMultiMatrixDerivId id, helper::vector<const OutDataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].read());       }


    /// Useful when the mapping is applied only on a subset of parent dofs.
    /// It is automatically called by applyJT.
    ///
    /// That way, we can optimize Jacobian sparsity.
    /// Every Dofs are inserted by default. The mappings using only a subset of dofs should only insert these dofs in the mask.
    virtual void updateForceMask() override;

    /// keep pointers on the masks
    helper::vector<helper::StateMask*> maskFrom1, maskFrom2, maskTo;
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_CORE_MULTI2MAPPING_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1dTypes, defaulttype::Rigid3dTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Rigid3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Rigid3dTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Vec3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1dTypes, defaulttype::Vec3dTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1dTypes, defaulttype::Rigid3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1dTypes, defaulttype::Vec1dTypes, defaulttype::Rigid3dTypes >;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3fTypes, defaulttype::Rigid3fTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3fTypes, defaulttype::Rigid3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1fTypes, defaulttype::Rigid3fTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3fTypes, defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1fTypes, defaulttype::Vec3fTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1fTypes, defaulttype::Rigid3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1fTypes, defaulttype::Vec1fTypes, defaulttype::Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Vec3fTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Vec3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Rigid3fTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Rigid3dTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Rigid3fTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3fTypes, defaulttype::Rigid3dTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3fTypes, defaulttype::Rigid3fTypes, defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3fTypes, defaulttype::Rigid3dTypes, defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Vec3fTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Vec3fTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Rigid3fTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Rigid3dTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3dTypes, defaulttype::Rigid3fTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3fTypes, defaulttype::Rigid3dTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3fTypes, defaulttype::Rigid3fTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3fTypes, defaulttype::Rigid3dTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1fTypes, defaulttype::Rigid3fTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1fTypes, defaulttype::Rigid3dTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1dTypes, defaulttype::Rigid3fTypes, defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1fTypes, defaulttype::Rigid3dTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1dTypes, defaulttype::Rigid3fTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1dTypes, defaulttype::Rigid3dTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1fTypes, defaulttype::Vec3fTypes, defaulttype::Rigid3dTypes >;
#endif
#endif

#endif

} // namespace core

} // namespace sofa

#endif

