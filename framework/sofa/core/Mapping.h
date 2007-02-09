#ifndef SOFA_CORE_MAPPING_H
#define SOFA_CORE_MAPPING_H

#include <sofa/core/BaseMapping.h>

namespace sofa
{

namespace core
{

template <class TIn, class TOut>
class Mapping : public BaseMapping
{
public:
    typedef TIn In;
    typedef TOut Out;

protected:
    In* fromModel;
    Out* toModel;

public:
    Mapping(In* from, Out* to);
    virtual ~Mapping();

    objectmodel::BaseObject* getFrom();
    objectmodel::BaseObject* getTo();

    virtual void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) = 0;
    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) = 0;

    virtual void init();

    virtual void updateMapping();

    /// Pre-construction check method called by ObjectFactory.
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
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        obj = new T(
            dynamic_cast<In*>(arg->findObject(arg->getAttribute("object1","../.."))),
            dynamic_cast<Out*>(arg->findObject(arg->getAttribute("object2",".."))));
        if (context) context->addObject(obj);
        obj->parse(arg);
    }
};

} // namespace core

} // namespace sofa

#endif
