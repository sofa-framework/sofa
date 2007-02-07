#ifndef SOFA_CORE_OBJECTMODEL_BASE_H
#define SOFA_CORE_OBJECTMODEL_BASE_H

#include <string>
#include <sofa/core/objectmodel/FieldContainer.h>

namespace sofa
{

namespace core
{

namespace objectmodel
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
    virtual const char* getTypeName() const;

protected:
    //std::string name;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif

