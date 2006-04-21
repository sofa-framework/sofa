#ifndef SOFA_ABSTRACT_BASE_H
#define SOFA_ABSTRACT_BASE_H

#include <string>

namespace Sofa
{

namespace Abstract
{

/// Base class for everything
class Base
{
public:
    virtual ~Base() {}

    const std::string& getName() const { return name; }
    void setName(const std::string& n) { name = n; }

protected:
    std::string name;
};

} // namespace Abstract

} // namespace Sofa

#endif
