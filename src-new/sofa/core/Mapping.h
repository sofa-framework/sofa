#ifndef SOFA_CORE_MAPPING_H
#define SOFA_CORE_MAPPING_H

#include "BasicMapping.h"

namespace Sofa
{

namespace Core
{

template <class TIn, class TOut>
class Mapping : public BasicMapping
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

    Abstract::BaseObject* getFrom();
    Abstract::BaseObject* getTo();

    virtual void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) = 0;
    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) = 0;

    virtual void init();

    virtual void updateMapping();
};

} // namespace Core

} // namespace Sofa

#endif
