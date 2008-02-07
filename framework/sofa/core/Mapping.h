/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_MAPPING_H
#define SOFA_CORE_MAPPING_H

#include <sofa/core/BaseMapping.h>

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
    /// Input Model Type
    typedef TIn In;
    /// Output Model Type
    typedef TOut Out;

protected:
    /// Input Model
    In* fromModel;
    /// Output Model
    Out* toModel;

    Data< std::string > object1;
    Data< std::string > object2;
public:
    /// Constructor, taking input and output models as parameters.
    ///
    /// Note that if you do not specify these models here, you must called
    /// setModels with non-NULL value before the intialization (i.e. before
    /// init() is called).
    Mapping(In* from=NULL, Out* to=NULL);

    virtual ~Mapping();

    /// Specify the input and output models.
    virtual void setModels(In* from, Out* to);

    /// Return the pointer to the input model.
    In* getFromModel();
    /// Return the pointer to the output model.
    Out* getToModel();

    /// Return the pointer to the input model.
    objectmodel::BaseObject* getFrom();
    /// Return the pointer to the output model.
    objectmodel::BaseObject* getTo();

    /// Apply the mapping on position vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    ///
    /// This method must be reimplemented by all mappings.
    virtual void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) = 0;

    /// Apply the mapping on derived (velocity, displacement) vectors.
    ///
    /// If the Mapping can be represented as a matrix J, this method computes
    /// $ out = J in $
    ///
    /// This method must be reimplemented by all mappings.
    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) = 0;

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
            std::cerr << "Cannot create "<<className(obj)<<" as object1 is missing.\n";
        if (arg->findObject(arg->getAttribute("object2","..")) == NULL)
            std::cerr << "Cannot create "<<className(obj)<<" as object2 is missing.\n";
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
        obj = new T(
            (arg?dynamic_cast<In*>(arg->findObject(arg->getAttribute("object1","../.."))):NULL),
            (arg?dynamic_cast<Out*>(arg->findObject(arg->getAttribute("object2",".."))):NULL));
        if (context) context->addObject(obj);
        if ((arg) && (arg->getAttribute("object1")))
        {
            obj->object1.setValue( arg->getAttribute("object1") );
            arg->removeAttribute("object1");
        }
        if ((arg) && (arg->getAttribute("object2")))
        {
            obj->object2.setValue( arg->getAttribute("object2") );
            arg->removeAttribute("object2");
        }
        if (arg) obj->parse(arg);
    }

protected:
    bool getShow() const { return this->getContext()->getShowMappings(); }
};

} // namespace core

} // namespace sofa

#endif
