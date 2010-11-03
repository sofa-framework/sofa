/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_MAPPING_H
#define SOFA_CORE_MAPPING_H

#include <sofa/core/BaseMapping.h>
#include <sofa/core/State.h>
#include <sofa/core/objectmodel/ObjectRef.h>

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace core
{

/**
*  \brief Specialized interface to convert a model of type TIn to an other model of type TOut
*
*  This Interface is used for the Mappings. A Mapping can convert one model to an other.
*  For example, we can have a mapping from a BehaviorModel to a VisualModel.
*
*/

template <class TIn, class TOut>
class Mapping : public BaseMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(Mapping,TIn,TOut), BaseMapping);

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
    /// Input Model
    State< In >* fromModel;
    /// Output Model
    State< Out >* toModel;
public:
    /// Name of the Input Model
    //Data< std::string > object1;
    objectmodel::DataObjectRef m_inputObject;
    /// Name of the Output Model
    //Data< std::string > object2;
    objectmodel::DataObjectRef m_outputObject;

    Data<bool> f_checkJacobian;

    /// Constructor, taking input and output models as parameters.
    ///
    /// Note that if you do not specify these models here, you must called
    /// setModels with non-NULL value before the intialization (i.e. before
    /// init() is called).
    Mapping(State< In >* from=NULL, State< Out >* to=NULL);
    /// Destructor
    virtual ~Mapping();

    /// Specify the input and output models.
    virtual void setModels(State< In > * from, State< Out >* to);

    /// Set the path to the objects mapped in the scene graph
    void setPathInputObject(const std::string &o) {m_inputObject.setValue(o);}
    void setPathOutPutObject(const std::string &o) {m_outputObject.setValue(o);}

    /// Return the pointer to the input model.
    State< In >* getFromModel();
    /// Return the pointer to the output model.
    State< Out >* getToModel();

    /// Return the pointer to the input model.
    helper::vector<BaseState*> getFrom();
    /// Return the pointer to the output model.
    helper::vector<BaseState*> getTo();

    /// Apply ///
    /// Apply the mapping on position vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    virtual void apply (MultiVecCoordId outPos, ConstMultiVecCoordId inPos, const MechanicalParams* mparams = MechanicalParams::defaultInstance() ) ;

    /// This method must be reimplemented by all mappings.
    virtual void apply( OutDataVecCoord& out, const InDataVecCoord& in, const MechanicalParams* /* mparams */)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        this->apply(*out.beginEdit(), in.getValue());
        out.endEdit();
    }
    /// Compat Method
    /// @deprecated
    virtual void apply( OutVecCoord& /* out */, const InVecCoord& /* in */) { };
#endif //SOFA_DEPRECATE_OLD_API

    /// ApplyJ ///
    /// Apply the mapping on derived (velocity, displacement) vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    virtual void applyJ(MultiVecDerivId outVel, ConstMultiVecDerivId inVel, const MechanicalParams* mparams = MechanicalParams::defaultInstance() );

    /// This method must be reimplemented by all mappings.
    virtual void applyJ( OutDataVecDeriv& out, const InDataVecDeriv& in, const MechanicalParams* /* mparams */)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        this->applyJ(*out.beginEdit(), in.getValue());
        out.endEdit();
    }
    /// Compat Method
    /// @deprecated
    virtual void applyJ( OutVecDeriv& /* out */, const InVecDeriv& /* in */) { };
#endif //SOFA_DEPRECATE_OLD_API

    /// ApplyJT (Force)///
    /// Apply the reverse mapping on force vectors.
    ///
    /// If the MechanicalMapping can be represented as a matrix J, this method computes
    /// $ out += J^t in $
    virtual void applyJT(MultiVecDerivId inForce, ConstMultiVecDerivId outForce, const MechanicalParams* mparams = MechanicalParams::defaultInstance() );

    /// This method must be reimplemented by all mappings.
    virtual void applyJT( InDataVecDeriv& out, const OutDataVecDeriv& in, const MechanicalParams* /* mparams */)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        this->applyJT(*out.beginEdit(), in.getValue());
        out.endEdit();
    }
    /// Compat Method
    /// @deprecated
    virtual void applyJT( InVecDeriv& /* out */, const OutVecDeriv& /* in */) { };
#endif //SOFA_DEPRECATE_OLD_API

    /// ApplyJT (Constraint)///
    virtual void applyJT(MultiMatrixDerivId inConst, ConstMultiMatrixDerivId outConst, const ConstraintParams* cparams = ConstraintParams::defaultInstance() )
    {
        if(this->fromModel && this->toModel)
        {
            InDataMatrixDeriv* out = inConst[fromModel].write();
            const OutDataMatrixDeriv* in = outConst[toModel].read();
            if(out && in)
            {
                if (this->isMechanical() && this->f_checkJacobian.getValue())
                    checkApplyJT(*out->beginEdit(), in->getValue(), this->getJ());
                else
                    this->applyJT(*out, *in, cparams);
            }
        }
    }
    /// This method must be reimplemented by all mappings if they need to support constraints.
    virtual void applyJT( InDataMatrixDeriv& out, const OutDataMatrixDeriv& in, const ConstraintParams* /* mparams */)
#ifdef SOFA_DEPRECATE_OLD_API
    {
        serr << "This mapping does not support constraints" << sendl;
    }
#else
    {
        this->applyJT(*out.beginEdit(), in.getValue());
        out.endEdit();
    }
    /// Compat Method
    /// @deprecated
    virtual void applyJT( InMatrixDeriv& /*out*/, const OutMatrixDeriv& /*in*/ )
    {
        serr << "This mapping does not support constraints" << sendl;
    }
#endif //SOFA_DEPRECATE_OLD_API

    /// computeAccFromMapping
    /// If the mapping input has a rotation velocity, it computes the subsequent acceleration
    /// created by the derivative terms
    /// $ a_out = w^(w^rel_pos)	$
    virtual void computeAccFromMapping(MultiVecDerivId outAcc, ConstMultiVecDerivId inVel, ConstMultiVecDerivId inAcc, const MechanicalParams* mparams = MechanicalParams::defaultInstance() )
    {
        if(this->fromModel && this->toModel)
        {
            OutDataVecDeriv* out = outAcc[toModel].write();
            const InDataVecDeriv* inV = inVel[fromModel].read();
            const InDataVecDeriv* inA = inAcc[fromModel].read();
            if(out && inV && inA)
                this->computeAccFromMapping(*out, *inV, *inA, mparams);
        }
    }
    /// This method must be reimplemented by all mappings if they need to support composite accelerations
    virtual void computeAccFromMapping(OutDataVecDeriv& accOut, const InDataVecDeriv& vIn, const InDataVecDeriv& accIn, const MechanicalParams* /* mparams */)
#ifdef SOFA_DEPRECATE_OLD_API
    {

    }
#else
    {
        this->computeAccFromMapping(*accOut.beginEdit(), vIn.getValue(), accIn.getValue());
        accOut.endEdit();
    }
    /// Compat Method
    /// @deprecated
    virtual void computeAccFromMapping( OutVecDeriv& /*acc_out*/, const InVecDeriv& /*v_in*/, const InVecDeriv& /*acc_in*/)
    {}
#endif //SOFA_DEPRECATE_OLD_API

    virtual void init();

    ///<TO REMOVE>
    // Useful ?
    /// Get the source (upper) model.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechFrom();

    /// Get the destination (lower, mapped) model.
    virtual helper::vector<behavior::BaseMechanicalState*> getMechTo();

    ///<TO REMOVE>
    /// Apply the mapping to position and velocity vectors.
    ///
    /// This method call the internal apply(Out::VecCoord&,const In::VecCoord&)
    /// and applyJ(Out::VecDeriv&,const In::VecDeriv&) methods.
    //virtual void updateMapping();

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

#ifndef SOFA_DEPRECATE_OLD_API
        ////Deprecated check
        Base* bobjInput = NULL;
        Base* bobjOutput = NULL;

        //Input
        if (arg->getAttribute("object1",NULL) == NULL && arg->getAttribute("input",NULL) == NULL)
            bobjInput = arg->findObject("../..");

        if (arg->getAttribute("object1",NULL) != NULL)
            bobjInput = sofa::core::objectmodel::ObjectRef::parseFromXMLPath("object1", arg);

        if (BaseObject* bo = dynamic_cast< BaseObject* >(bobjInput))
            stin = dynamic_cast< State<In>* >(bo);

        else if (core::objectmodel::BaseContext* bc = dynamic_cast< core::objectmodel::BaseContext* >(bobjInput))
            stin = dynamic_cast< State<In>* >(bc->getState());

        //Output
        if (arg->getAttribute("object2",NULL) == NULL && arg->getAttribute("output",NULL) == NULL)
            bobjOutput = arg->findObject("..");

        if (arg->getAttribute("object2",NULL) != NULL)
            bobjOutput = sofa::core::objectmodel::ObjectRef::parseFromXMLPath("object2", arg);

        if (BaseObject* bo = dynamic_cast< BaseObject* >(bobjOutput))
            stout = dynamic_cast< State<Out>* >(bo);

        else if (core::objectmodel::BaseContext* bc = dynamic_cast< core::objectmodel::BaseContext* >(bobjOutput))
            stout = dynamic_cast< State<Out>* >(bc->getState());
        /////

        if (stin == NULL || stout == NULL)
#endif // SOFA_DEPRECATE_OLD_API
        {
            stin = sofa::core::objectmodel::ObjectRef::parse< State<In> >("input", arg);
            stout = sofa::core::objectmodel::ObjectRef::parse< State<Out> >("output", arg);
        }

        if (stin == NULL)
        {
            //context->serr << "Cannot create "<<className(obj)<<" as object1 is missing or invalid." << context->sendl;
            return false;
        }

        if (stout == NULL)
        {
            //context->serr << "Cannot create "<<className(obj)<<" as object2 is missing or invalid." << context->sendl;
            return false;
        }

        return BaseMapping::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    ///
    /// This implementation read the input and output attributes to
    /// find the input and output models of this mapping.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
#ifdef SOFA_SMP_NUMA
        if(context&&context->getProcessor()!=-1)
        {
            obj = new(numa_alloc_onnode(sizeof(T),context->getProcessor()/2)) T(
                (arg?dynamic_cast< State< In >* >(arg->findObject(arg->getAttribute("object1","../.."))):NULL),
                (arg?dynamic_cast< State< Out >* >(arg->findObject(arg->getAttribute("object2",".."))):NULL));
        }
        else
#endif
        {
            State<In>* stin = NULL;
            State<Out>* stout = NULL;

#ifndef SOFA_DEPRECATE_OLD_API
            ////Deprecated check
            std::string object1Path;
            std::string object2Path;

            Base* bobjInput = NULL;
            Base* bobjOutput = NULL;

            //Input
            if(arg->getAttribute("object1",NULL) == NULL && arg->getAttribute("input",NULL) == NULL)
            {
                object1Path = "..";
                context->serr << "Deprecated use of implicit value for input" << context->sendl;
                context->serr << "Use now : input=\"@" << object1Path << "\" "<< context->sendl;
                bobjInput = arg->findObject("../..");
            }

            if(arg->getAttribute("object1",NULL) != NULL)
            {
                object1Path = sofa::core::objectmodel::ObjectRef::convertFromXMLPathToSofaScenePath(arg->getAttribute("object1",NULL));
                context->serr << "Deprecated use of attribute " << "object1" << context->sendl;
                context->serr << "Use now : input=\"@"
                        << object1Path
                        << "\""<< context->sendl;
                bobjInput = sofa::core::objectmodel::ObjectRef::parseFromXMLPath("object1", arg);
            }

            if (BaseObject* bo = dynamic_cast< BaseObject* >(bobjInput))
            {
                stin = dynamic_cast< State<In>* >(bo);
            }
            else if (core::objectmodel::BaseContext* bc = dynamic_cast< core::objectmodel::BaseContext* >(bobjInput))
            {
                stin = dynamic_cast< State<In>* >(bc->getState());
            }

            //Output
            if(arg->getAttribute("object2",NULL) == NULL && arg->getAttribute("output",NULL) == NULL)
            {
                object2Path = ".";
                context->serr << "Deprecated use of implicit value for output" << context->sendl;
                context->serr << "Use now : output=\"@" << object2Path << "\" "<< context->sendl;
                bobjOutput = arg->findObject("..");
            }

            if(arg->getAttribute("object2",NULL) != NULL)
            {
                object2Path = sofa::core::objectmodel::ObjectRef::convertFromXMLPathToSofaScenePath(arg->getAttribute("object2",NULL));
                context->serr << "Deprecated use of attribute " << "object2" << context->sendl;
                context->serr << "Use now : output=\"@"
                        << object2Path
                        << "\""<< context->sendl;
                bobjOutput = sofa::core::objectmodel::ObjectRef::parseFromXMLPath("object2", arg);
            }

            if (BaseObject* bo = dynamic_cast< BaseObject* >(bobjOutput))
            {
                stout = dynamic_cast< State<Out>* >(bo);
            }
            else if (core::objectmodel::BaseContext* bc = dynamic_cast< core::objectmodel::BaseContext* >(bobjOutput))
            {
                stout = dynamic_cast< State<Out>* >(bc->getState());
            }

            /////

            if(stin == NULL && stout == NULL)
#endif // SOFA_DEPRECATE_OLD_API
            {
                stin = sofa::core::objectmodel::ObjectRef::parse< State<In> >("input", arg);
                stout = sofa::core::objectmodel::ObjectRef::parse< State<Out> >("output", arg);
            }

            obj = new T( (arg?stin:NULL), (arg?stout:NULL));
#ifndef SOFA_DEPRECATE_OLD_API
            if (!object1Path.empty())
                obj->m_inputObject.setValue( object1Path );
            if (!object2Path.empty())
                obj->m_outputObject.setValue( object2Path );
#endif // SOFA_DEPRECATE_OLD_API
        }

        if (context)
            context->addObject(obj);

        if (arg)
            obj->parse(arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const Mapping<TIn, TOut>* = NULL);

protected:
    /// If true, display the mapping
    bool getShow() const { return this->getContext()->getShowMappings(); }

    void matrixApplyJ( OutVecDeriv& /* out */, const InVecDeriv& /* in */, const sofa::defaulttype::BaseMatrix* /* J */);
    void matrixApplyJT( InVecDeriv& /* out */, const OutVecDeriv& /* in */, const sofa::defaulttype::BaseMatrix* /* J */);
    void matrixApplyJT( InMatrixDeriv& /* out */, const OutMatrixDeriv& /* in */, const sofa::defaulttype::BaseMatrix* /* J */);
    bool checkApplyJ( OutVecDeriv& /* out */, const InVecDeriv& /* in */, const sofa::defaulttype::BaseMatrix* /* J */);
    bool checkApplyJT( InVecDeriv& /* out */, const OutVecDeriv& /* in */, const sofa::defaulttype::BaseMatrix* /* J */);
    bool checkApplyJT( InMatrixDeriv& /* out */, const OutMatrixDeriv& /* in */, const sofa::defaulttype::BaseMatrix* /* J */);

};

#if defined(WIN32) && !defined(SOFA_BUILD_CORE)

using namespace sofa::defaulttype;

extern template class SOFA_CORE_API Mapping< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_CORE_API Mapping< Rigid3dTypes, Vec3dTypes >;
extern template class SOFA_CORE_API Mapping< Vec3dTypes, ExtVec3fTypes >;

extern template class SOFA_CORE_API Mapping< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< Rigid3fTypes, Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< Vec3fTypes, ExtVec3fTypes >;

extern template class SOFA_CORE_API Mapping< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< Vec3fTypes, Vec3dTypes > ;
extern template class SOFA_CORE_API Mapping< Rigid3dTypes, Vec3fTypes >;
extern template class SOFA_CORE_API Mapping< Rigid3fTypes, Vec3dTypes >;
#endif

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_MAPPING_H
