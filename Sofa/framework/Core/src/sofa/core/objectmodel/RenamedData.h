#pragma once

#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/accessor/ReadAccessor.h>
#include <sofa/helper/accessor/WriteAccessor.h>

namespace sofa
{
namespace core::objectmodel
{


template < class T = void* >
class RenamedData
{
public:
    RenamedData() {};

    void setOriginalData(Data<T> * data)
    {
        m_originalData = data;
    }

    Data<T>* operator&()
    {
        return m_originalData;
    }

    operator Data<T>() const
    {
        return m_originalData;
    }

    operator sofa::helper::ReadAccessor<Data<T>>() const
    {
        return sofa::helper::ReadAccessor<Data<T>>(m_originalData);
    }

    operator sofa::helper::WriteAccessor<Data<T>>() const
    {
        return sofa::helper::WriteAccessor<Data<T>>(m_originalData);
    }

    operator sofa::helper::WriteOnlyAccessor<Data<T>>() const
    {
        return sofa::helper::WriteOnlyAccessor<Data<T>>(m_originalData);
    }


    //Public methods from BaseData
    bool read(const std::string& value) { return m_originalData->read(value); }

    void printValue(std::ostream& os) const { m_originalData->printValue(os); }

    std::string getValueString() const { return m_originalData->getValueString(); }

    std::string getDefaultValueString() const { return m_originalData->getValueString(); }

    std::string getValueTypeString() const { return m_originalData->getValueTypeString(); }

    const sofa::defaulttype::AbstractTypeInfo* getValueTypeInfo() const { return m_originalData->getValueTypeInfo(); }

    const void* getValueVoidPtr() const { return m_originalData->getValueTypeInfo(); }

    void* beginEditVoidPtr() { return m_originalData->beginEditVoidPtr(); }

    void endEditVoidPtr() { m_originalData->endEditVoidPtr(); }

    const std::string& getHelp() const { return m_originalData->getHelp(); }

    void setHelp(const std::string& val) { m_originalData->setHelp(val); }

    const std::string& getGroup() const { return m_originalData->getGroup(); }

    void setGroup(const std::string& val) { m_originalData->setGroup(val); }

    const std::string& getWidget() const { return m_originalData->getWidget(); }

    void setWidget(const char* val) { m_originalData->setWidget(val); }

    void setFlag(BaseData::DataFlagsEnum flag, bool b)  { m_originalData->setFlag(flag,b); }

    bool getFlag(BaseData::DataFlagsEnum flag) const { return m_originalData->getFlag(flag); }

    bool isDisplayed() const  { return m_originalData->isDisplayed(); }

    bool isReadOnly() const   { return m_originalData->isReadOnly(); }

    bool isPersistent() const { return m_originalData->isPersistent(); }

    bool isAutoLink() const { return m_originalData->isAutoLink(); }

    bool isRequired() const { return m_originalData->isRequired(); }

    void setDisplayed(bool b)  { m_originalData->setDisplayed(b); }

    void setReadOnly(bool b)   { m_originalData->setReadOnly(b); }

    void setPersistent(bool b) { m_originalData->setPersistent(b); }

    void setAutoLink(bool b) { m_originalData->setAutoLink(b); }

    void setRequired(bool b) { m_originalData->setRequired(b); }

    std::string getLinkPath() const { return m_originalData->getLinkPath(); }

    bool canBeLinked() const { return m_originalData->canBeLinked(); }

    Base* getOwner() const { return m_originalData->getOwner(); }

    void setOwner(Base* o) { m_originalData->setOwner(o); }

    BaseData* getData() const { return m_originalData->getData(); }

    const std::string& getName() const { return m_originalData->getName(); }

    void setName(const std::string& name) { m_originalData->setName(name); }

    bool hasDefaultValue() const { return m_originalData->hasDefaultValue(); }

    bool isSet() const { return m_originalData->isSet(); }

    void unset() { m_originalData->unset(); }

    void forceSet() { m_originalData->forceSet(); }

    int getCounter() const { return m_originalData->getCounter(); }

    bool setParent(BaseData* parent, const std::string& path = std::string()) { return m_originalData->setParent(parent,path); }

    bool setParent(const std::string& path) { return m_originalData->setParent(path); }

    bool validParent(const BaseData *parent) { return m_originalData->validParent(parent); }

    BaseData* getParent() const { return m_originalData->getTarget(); }

    void update() { m_originalData->update(); }

    bool copyValueFrom(const BaseData* data) { return m_originalData->copyValueFrom(data);}

    bool updateValueFromLink(const BaseData* data) { return m_originalData->updateValueFromLink(data); }

    //Public methods from Data<T>
    T* beginEdit() { return m_originalData->beginEdit(); }

    T* beginWriteOnly() { return m_originalData->beginWriteOnly(); }

    void endEdit() { m_originalData->endEdit(); }

    void setValue(const T& value) { m_originalData->setValue(value); }

    const T& getValue() const { return m_originalData->getValue(); }

    void operator =( const T& value ) { m_originalData->operator=(value); }

    bool copyValueFrom(const Data<T>* data) { return m_originalData->copyValueFrom(data); }

private:
    Data<T>* m_originalData { nullptr };
};

}
}