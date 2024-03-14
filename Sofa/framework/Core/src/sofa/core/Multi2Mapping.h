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
#include <sofa/core/PathResolver.h>
#include <sofa/core/config.h>
#include <sofa/core/State.h>

namespace sofa::core
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
    ~Multi2Mapping() override {}
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
    type::vector<BaseState*> getFrom() override;
    /// Return container of output model statically casted as BaseObject*.
    type::vector<BaseState*> getTo() override;

    /// Get the source (upper) model.
    virtual type::vector<behavior::BaseMechanicalState*> getMechFrom() override;

    /// Get the destination (lower, mapped) model.
    virtual type::vector<behavior::BaseMechanicalState*> getMechTo() override;

    /// Apply ///
    /// Apply the mapping to position vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    void apply (const MechanicalParams* mparams, MultiVecCoordId outPos, ConstMultiVecCoordId inPos ) override;

    /// This method must be reimplemented by all mappings.
    /// InPos and OutPos by default contains VecIds of type V_COORD.
    /// The size of InPos vector is the same as the number of fromModels.
    /// The size of OutPos vector is the same as the number of OutModels.
    virtual void apply(
        const MechanicalParams* mparams, const type::vector<OutDataVecCoord*>& dataVecOutPos,
        const type::vector<const In1DataVecCoord*>& dataVecIn1Pos ,
        const type::vector<const In2DataVecCoord*>& dataVecIn2Pos) = 0;

    /// ApplyJ ///
    /// This method computes
    /// $ out = J in $
    /// where J is the tangent operator (the linear approximation) of the mapping
    void applyJ (const MechanicalParams* mparams, MultiVecDerivId outVel, ConstMultiVecDerivId inVel ) override;

    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJ(
        const MechanicalParams*, const type::vector< OutDataVecDeriv*>& dataVecOutVel,
        const type::vector<const In1DataVecDeriv*>& dataVecIn1Vel,
        const type::vector<const In2DataVecDeriv*>& dataVecIn2Vel)
    {
        //Not optimized at all...
        type::vector<OutVecDeriv*> vecOutVel;
        for(unsigned int i=0; i<dataVecOutVel.size(); i++)
            vecOutVel.push_back(dataVecOutVel[i]->beginEdit());

        type::vector<const In1VecDeriv*> vecIn1Vel;
        for(unsigned int i=0; i<dataVecIn1Vel.size(); i++)
            vecIn1Vel.push_back(&dataVecIn1Vel[i]->getValue());
        type::vector<const In2VecDeriv*> vecIn2Vel;
        for(unsigned int i=0; i<dataVecIn2Vel.size(); i++)
            vecIn2Vel.push_back(&dataVecIn2Vel[i]->getValue());
        this->applyJ(vecOutVel, vecIn1Vel, vecIn2Vel);

        //Really Not optimized at all...
        for(unsigned int i=0; i<dataVecOutVel.size(); i++)
            dataVecOutVel[i]->endEdit();
    }
    /// Compat Method
    /// @deprecated
    virtual void applyJ(const type::vector< OutVecDeriv*>& /* outDeriv */,
            const type::vector<const In1VecDeriv*>& /* inDeriv1 */,
            const type::vector<const In2VecDeriv*>& /* inDeriv2 */) {}

    /// ApplyJT (Force)///
    /// Apply the mapping to Force vectors.
    void applyJT (const MechanicalParams* mparams, MultiVecDerivId inForce, ConstMultiVecDerivId outForce ) override;

    /// This method must be reimplemented by all mappings.
    /// InDeriv and OutDeriv by default contains VecIds of type V_DERIV.
    /// The size of InDeriv vector is the same as the number of fromModels.
    /// The size of OutDeriv vector is the same as the number of OutModels.
    virtual void applyJT(
        const MechanicalParams* mparams, const type::vector< In1DataVecDeriv*>& dataVecOut1Force,
        const type::vector< In2DataVecDeriv*>& dataVecOut2Force,
        const type::vector<const OutDataVecDeriv*>& dataVecInForce) = 0;

    /// ApplyJT (Constraint)///
    void applyJT(const ConstraintParams* cparams, MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst ) override;

    /// This method must be reimplemented by all mappings if they need to support constraints.
    virtual void applyJT(
        const ConstraintParams* /* cparams */, const type::vector< In1DataMatrixDeriv*>& /* dataMatOut1Const */ ,
        const type::vector< In2DataMatrixDeriv*>&  /* dataMatOut2Const */,
        const type::vector<const OutDataMatrixDeriv*>& /* dataMatInConst */)
    {
        msg_error() << "This mapping does not support constraint because Multi2Mapping::applyJT(const ConstraintParams*, const type::vector< In1DataMatrixDeriv*>&, const type::vector< In2DataMatrixDeriv*>&, const type::vector<const OutDataMatrixDeriv*>&) is not overloaded.";
    }

    /// computeAccFromMapping
    void computeAccFromMapping(const MechanicalParams* mparams, MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc ) override;

    /// This method must be reimplemented by all mappings if they need to support composite accelerations
    virtual void computeAccFromMapping(
        const MechanicalParams* /* mparams */, const type::vector< OutDataVecDeriv*>& /* dataVecOutAcc */,
        const type::vector<const In1DataVecDeriv*>& /* dataVecIn1Vel */,
        const type::vector<const In2DataVecDeriv*>& /* dataVecIn2Vel */,
        const type::vector<const In1DataVecDeriv*>& /* dataVecIn1Acc */,
        const type::vector<const In2DataVecDeriv*>& /* dataVecIn2Acc */)
    {
    }

    void init() override;

    /// Disable the mapping to get the original coordinates of the mapped model.
    ///
    /// It is for instance used in RigidMapping to get the local coordinates of the object.
    void disable() override;

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes and check
    /// if they are compatible with the input and output models types of this
    /// mapping.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        static const sofa::type::fixed_array<std::tuple<const char*, const char*, const ::sofa::core::objectmodel::BaseClass*>, 3> attributes {
            std::make_tuple("input1", In1::Name(), LinkFromModels1::DestType::GetClass()),
            std::make_tuple("input2", In2::Name(), LinkFromModels2::DestType::GetClass()),
            std::make_tuple("output", Out::Name(), LinkToModels::DestType::GetClass())
        };

        bool error = false;
        for (const auto& [attribute, dataType, classDescription] : attributes)
        {
            const std::string attributeStr = arg->getAttribute(attribute,"");
            if (!attributeStr.empty() && !PathResolver::CheckPaths(context, classDescription, attributeStr))
            {
                arg->logError("Data attribute '" + std::string(attribute) + "' does not point to a "
                            "mechanical state of data type '" + std::string(dataType) + "'");
                error = true;
            }
        }

        const std::string attributeStr = arg->getAttribute("output","");
        if (attributeStr.empty())
        {
            arg->logError("Data attribute 'output' is empty, but it is required to set it. "
                        "Set this attribute to a mechanical state of data type '" + std::string(Out::Name()) + "'");
            error = true;
        }

        return !error && BaseMapping::canCreate(obj, context, arg);
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
    void getVecIn1Coord     (const MultiVecCoordId id,         type::vector<      In1DataVecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].write()); }
    void getConstVecIn1Coord(const ConstMultiVecCoordId id,    type::vector<const In1DataVecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].read());  }
    void getVecIn1Deriv     (const MultiVecDerivId id,         type::vector<      In1DataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].write()); }
    void getConstVecIn1Deriv(const ConstMultiVecDerivId id,    type::vector<const In1DataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].read());  }
    void getMatIn1Deriv     (const MultiMatrixDerivId id,      type::vector<      In1DataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].write()); }
    void getConstMatIn1Deriv(const ConstMultiMatrixDerivId id, type::vector<const In1DataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels1.size(); ++i) v.push_back(id[fromModels1[i]].read());  }

    void getVecIn2Coord     (const MultiVecCoordId id,         type::vector<      In2DataVecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].write()); }
    void getConstVecIn2Coord(const ConstMultiVecCoordId id,    type::vector<const In2DataVecCoord*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].read());  }
    void getVecIn2Deriv     (const MultiVecDerivId id,         type::vector<      In2DataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].write()); }
    void getConstVecIn2Deriv(const ConstMultiVecDerivId id,    type::vector<const In2DataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].read());  }
    void getMatIn2Deriv     (const MultiMatrixDerivId id,      type::vector<      In2DataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].write()); }
    void getConstMatIn2Deriv(const ConstMultiMatrixDerivId id, type::vector<const In2DataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<fromModels2.size(); ++i) v.push_back(id[fromModels2[i]].read());  }

    void getVecOutCoord     (const MultiVecCoordId id,         type::vector<      OutDataVecCoord*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].write());      }
    void getConstVecOutCoord(const ConstMultiVecCoordId id,    type::vector<const OutDataVecCoord*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].read());       }
    void getVecOutDeriv     (const MultiVecDerivId id,         type::vector<      OutDataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].write());      }
    void getConstVecOutDeriv(const ConstMultiVecDerivId id,    type::vector<const OutDataVecDeriv*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].read());       }
    void getMatOutDeriv     (const MultiMatrixDerivId id,      type::vector<      OutDataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].write());      }
    void getConstMatOutDeriv(const ConstMultiMatrixDerivId id, type::vector<const OutDataMatrixDeriv*> &v) const
    {   for (unsigned int i=0; i<toModels.size(); ++i)  v.push_back(id[toModels[i]].read());       }

};


#if !defined(SOFA_CORE_MULTI2MAPPING_CPP)

extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1Types, defaulttype::Rigid3Types, defaulttype::Rigid3Types >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3Types, defaulttype::Rigid3Types, defaulttype::Vec3Types >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3Types, defaulttype::Rigid3Types, defaulttype::Rigid3Types >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3Types, defaulttype::Vec3Types, defaulttype::Vec3Types >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec3Types, defaulttype::Vec1Types, defaulttype::Vec3Types >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1Types, defaulttype::Vec3Types, defaulttype::Rigid3Types >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1Types, defaulttype::Rigid3Types, defaulttype::Vec3Types >;
extern template class SOFA_CORE_API Multi2Mapping< defaulttype::Vec1Types, defaulttype::Vec1Types, defaulttype::Rigid3Types >;




#endif
} // namespace sofa::core

