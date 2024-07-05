#pragma once

#include <sofa/core/objectmodel/Data.h>

namespace sofa
{
namespace core::objectmodel
{


template < class T = void* >
class RenamedData : public Data<T>
{
public:
    RenamedData() {};

    void setParent(Data<T> * data)
    {
        m_originalData = data;
    }

    virtual T* beginEdit()
    {
        return m_originalData->beginEdit();
    }

    virtual T* beginWriteOnly()
    {
        return m_originalData->beginWriteOnly();
    }

    virtual void endEdit() override
    {
        m_originalData->endEdit();
    }

    virtual void setValue(const T& value) override
    {
        m_originalData->setValue(value);
    }

    virtual const T& getValue() const override
    {
        return m_originalData->getValue();
    }
private:
    Data<T>* m_originalData;
};

}
}