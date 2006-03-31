#ifndef SOFA_CORE_MAPPING_INL
#define SOFA_CORE_MAPPING_INL

#include "Mapping.h"

namespace Sofa
{

namespace Core
{

template <class In, class Out>
Mapping<In,Out>::Mapping(In* from, Out* to)
    : fromModel(from), toModel(to)
{
}

template <class In, class Out>
Mapping<In,Out>::~Mapping()
{
}

template <class In, class Out>
Abstract::Base* Mapping<In,Out>::getFrom()
{
    return this->fromModel;
}

template <class In, class Out>
Abstract::Base* Mapping<In,Out>::getTo()
{
    return this->toModel;
}

template <class In, class Out>
void Mapping<In,Out>::init()
{
    this->updateMapping();
}

template <class In, class Out>
void Mapping<In,Out>::updateMapping()
{
    if (this->toModel->getX()!=NULL && this->fromModel->getX()!=NULL)
    {
        apply(*this->toModel->getX(), *this->fromModel->getX());
        //this->toModel->propagateX();
    }
    if (this->toModel->getV()!=NULL && this->fromModel->getV()!=NULL)
    {
        applyJ(*this->toModel->getV(), *this->fromModel->getV());
        //this->toModel->propagateV();
    }
}

} // namespace Core

} // namespace Sofa

#endif
