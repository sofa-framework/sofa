#ifndef SOFA_ABSTRACT_BASE_H
#define SOFA_ABSTRACT_BASE_H

#include <string>
#include "FieldContainer.h"

namespace Sofa
{

namespace Abstract
{

/// Base class for everything
class Base:  public FieldContainer
{
public:
    Base();
    virtual ~Base();

    DataField<std::string> name;

    std::string getName() const;
    void setName(const std::string& n);
    virtual const char* getTypeName() const
    {
        return "UNKNOWN";
    }

protected:
    //std::string name;
};

} // namespace Abstract

} // namespace Sofa

#endif

