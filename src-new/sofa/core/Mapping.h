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
};

} // namespace core

} // namespace sofa

#endif
