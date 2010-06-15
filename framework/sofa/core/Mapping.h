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
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MappedModel.h>

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

    /// Input Model Type
    typedef TIn In;
    /// Output Model Type
    typedef TOut Out;

    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;

protected:
    /// Input Model
    In* fromModel;
    /// Output Model
    Out* toModel;
    /// Name of the Input Model
    Data< std::string > object1;
    /// Name of the Output Model
    Data< std::string > object2;
public:
    /// Constructor, taking input and output models as parameters.
    ///
    /// Note that if you do not specify these models here, you must called
    /// setModels with non-NULL value before the intialization (i.e. before
    /// init() is called).
    Mapping(In* from=NULL, Out* to=NULL);
    /// Destructor
    virtual ~Mapping();

    /// Specify the input and output models.
    virtual void setModels(In* from, Out* to);

    /// Set the path to the objects mapped in the scene graph by default object1="../.." and object2=".."
    void setPathObject1(std::string &o) {object1.setValue(o);}
    void setPathObject2(std::string &o) {object2.setValue(o);}

    /// Return the pointer to the input model.
    In* getFromModel();
    /// Return the pointer to the output model.
    Out* getToModel();

    /// Return the pointer to the input model.
    helper::vector<objectmodel::BaseObject*> getFrom();
    /// Return the pointer to the output model.
    helper::vector<objectmodel::BaseObject*> getTo();

    /// Apply the mapping on position vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    ///
    /// This method must be reimplemented by all mappings.
    virtual void apply( OutVecCoord& out, const InVecCoord& in ) = 0;

    /// Apply the mapping on derived (velocity, displacement) vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    ///
    /// This method must be reimplemented by all mappings.
    virtual void applyJ( OutVecDeriv& out, const InVecDeriv& in ) = 0;

    virtual void init();

    /// Apply the mapping to position and velocity vectors.
    ///
    /// This method call the internal apply(Out::VecCoord&,const In::VecCoord&)
    /// and applyJ(Out::VecDeriv&,const In::VecDeriv&) methods.
    virtual void updateMapping();

    /// Disable the mapping to get the original coordinates of the mapped model.
    ///
    /// It is for instance used in RigidMapping to get the local coordinates of the object.
    virtual void disable();

    /// Pre-construction check method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes and check
    /// if they are compatible with the input and output model types of this
    /// mapping.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->findObject(arg->getAttribute("object1","../..")) == NULL)
            context->serr << "Cannot create "<<className(obj)<<" as object1 is missing."<<context->sendl;
        if (arg->findObject(arg->getAttribute("object2","..")) == NULL)
            context->serr << "Cannot create "<<className(obj)<<" as object2 is missing."<<context->sendl;
        if (dynamic_cast<In*>(arg->findObject(arg->getAttribute("object1","../.."))) == NULL)
            return false;
        if (dynamic_cast<Out*>(arg->findObject(arg->getAttribute("object2",".."))) == NULL)
            return false;
        return BaseMapping::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    ///
    /// This implementation read the object1 and object2 attributes to
    /// find the input and output models of this mapping.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
#ifdef SOFA_SMP_NUMA
        if(context&&context->getProcessor()!=-1)
        {

            obj = new(numa_alloc_onnode(sizeof(T),context->getProcessor()/2)) T(
                (arg?dynamic_cast<In*>(arg->findObject(arg->getAttribute("object1","../.."))):NULL),
                (arg?dynamic_cast<Out*>(arg->findObject(arg->getAttribute("object2",".."))):NULL));
        }
        else
#endif
            obj = new T(
                (arg?dynamic_cast<In*>(arg->findObject(arg->getAttribute("object1","../.."))):NULL),
                (arg?dynamic_cast<Out*>(arg->findObject(arg->getAttribute("object2",".."))):NULL));
        if (context) context->addObject(obj);
        if (arg)
        {
            if (arg->getAttribute("object1"))
            {
                obj->object1.setValue( arg->getAttribute("object1") );
                arg->removeAttribute("object1");
            }
            if (arg->getAttribute("object2"))
            {
                obj->object2.setValue( arg->getAttribute("object2") );
                arg->removeAttribute("object2");
            }
            obj->parse(arg);
        }
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }


    static std::string templateName(const Mapping<TIn, TOut>* = NULL);

protected:
    /// If true, display the mapping
    bool getShow() const { return this->getContext()->getShowMappings(); }
};

#if defined(WIN32) && !defined(SOFA_BUILD_CORE)

using namespace sofa::defaulttype;
using namespace sofa::core::behavior;

extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> >;
extern template class SOFA_CORE_API Mapping< MechanicalState<StdRigidTypes<3,double> >, MechanicalState<Vec3dTypes> >;
extern template class SOFA_CORE_API Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> >;
// extern template class SOFA_CORE_API Mapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> >;
extern template class SOFA_CORE_API Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> >;

extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> >;
extern template class SOFA_CORE_API Mapping< MechanicalState<StdRigidTypes<3,float> >, MechanicalState<Vec3fTypes> >;
extern template class SOFA_CORE_API Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> >;
extern template class SOFA_CORE_API Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> >;
// extern template class SOFA_CORE_API Mapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> >;

extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> >;
extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > ;
extern template class SOFA_CORE_API Mapping< MechanicalState<StdRigidTypes<3,double> >, MechanicalState<Vec3fTypes> >;
extern template class SOFA_CORE_API Mapping< MechanicalState<StdRigidTypes<3,float> >, MechanicalState<Vec3dTypes> >;
extern template class SOFA_CORE_API Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> >;
extern template class SOFA_CORE_API Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> >;

// Mech -> Mapped
//extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> >;
//extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> >;
//extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> >;
//extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> >;

// Mech -> ExtMapped
//extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> >;
//extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> >;
//extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> >;
//extern template class SOFA_CORE_API Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> >;

#endif

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_MAPPING_H
