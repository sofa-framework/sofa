#pragma once

#include <sofa/helper/system/config.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MutationListener.h>
#include <sofa/gui/qt/GenericWidget.h>
#include <sofa/core/objectmodel/DataCallback.h>
#include <QLineEdit>

namespace sofa {

namespace sofaTypes {

class BaseDataLink : public sofa::core::objectmodel::Data< std::string > {
public:
    typedef sofa::core::objectmodel::Data<std::string > Inherit;
    typedef sofa::simulation::Node Node;

    DataCallback c_callback;

    BaseDataLink(const typename Inherit::InitData& init)
    : Inherit(init)
    , c_callback(*this) {
        setWidget("DataLink");
        c_callback.addCallback(this,&BaseDataLink::computeLink);
    }

    virtual sofa::core::objectmodel::BaseObject * getBaseLink() const = 0;

    virtual void activate(Node * n) = 0;

    virtual void computeLink() = 0;

};

class ActivatorLink : public sofa::core::objectmodel::BaseLink {
public:
    ActivatorLink(BaseDataLink & d)
    : BaseLink(LinkFlagsEnum::FLAG_NONE)
    , m_datalink(d) {
        m_datalink.getOwner()->addLink(this);
    }

    virtual sofa::core::objectmodel::Base* getOwnerBase() const { return NULL; }
    virtual sofa::core::objectmodel::BaseData* getOwnerData() const { return NULL; }
    virtual const sofa::core::objectmodel::BaseClass* getDestClass() const { return NULL; }
    virtual const sofa::core::objectmodel::BaseClass* getOwnerClass() const { return NULL; }
    virtual size_t getSize() const { return 0;}
    virtual sofa::core::objectmodel::Base* getLinkedBase(unsigned int ) const { return NULL; }
    virtual sofa::core::objectmodel::BaseData* getLinkedData(unsigned int ) const { return NULL; }
    virtual std::string getLinkedPath(unsigned int ) const { return ""; }
    virtual bool read( const std::string& ) { return true; }

    virtual bool updateLinks() {
        if (sofa::core::objectmodel::BaseObject* owner = dynamic_cast<sofa::core::objectmodel::BaseObject*>(m_datalink.getOwner())) {
            if (sofa::simulation::Node * node = dynamic_cast<sofa::simulation::Node*>(owner->getContext()->getRootContext())) {
                m_datalink.activate(node);
                return true;
            }
        }
        return false;
    }

    virtual void copyAspect(int , int ) {}

    BaseDataLink & m_datalink;
};

template<class T>
class DataLink : public BaseDataLink, public sofa::simulation::MutationListener {
public:
    typedef BaseDataLink Inherit;

    DataLink(const typename Inherit::InitData& init)
    : Inherit(init)
    , m_actiavtor(*this) {}

    inline T* get() const {
        return reinterpret_cast<T*>(m_link.get());
    }

    inline T* operator->() const {
        return get();
    }

    friend inline bool operator==(const DataLink & lhs, const T * rhs) {
        return lhs.get() == rhs;
    }

    friend inline bool operator!=(const DataLink & lhs, const T * rhs) {
        return lhs.get() != rhs;
    }

    sofa::core::objectmodel::BaseObject * getBaseLink() const {
        return m_link.get();
    }

    void computeLink() {
        m_link = NULL;
        if (sofa::core::objectmodel::BaseObject* owner = dynamic_cast<sofa::core::objectmodel::BaseObject*>(this->getOwner())) {
            std::string path = getValue();
            m_link = owner->getContext()->get<T>(path);
        }
        beginEdit();
        endEdit();
    }

    void activate(Node * n) {
        n->addListener(this);
        computeLink();
    }

protected:
    void addChild(Node* , Node* ) { /*computeLink();*/ }

    void removeChild(Node* , Node* ) { /*computeLink();*/ }

    void moveChild(Node* , Node* , Node* ) { /*computeLink();*/ }

    void addObject(Node* , core::objectmodel::BaseObject* obj) {
        if (m_link == NULL || m_link == obj || obj == getOwner()) computeLink();
    }

    void removeObject(Node* , core::objectmodel::BaseObject* obj) {
        if (m_link == NULL || m_link == obj || obj == getOwner()) computeLink();
    }

    void moveObject(Node* , Node* , core::objectmodel::BaseObject* obj) {
        if (m_link == NULL || m_link == obj || obj == getOwner()) computeLink();
    }

    void addSlave(core::objectmodel::BaseObject* , core::objectmodel::BaseObject* obj) {
        if (m_link == NULL || m_link == obj || obj == getOwner()) computeLink();
    }

    void removeSlave(core::objectmodel::BaseObject* , core::objectmodel::BaseObject* obj) {
        if (m_link == NULL || m_link == obj || obj == getOwner()) computeLink();
    }

    void moveSlave(core::objectmodel::BaseObject* , core::objectmodel::BaseObject* , core::objectmodel::BaseObject* obj) {
        if (m_link == NULL || m_link == obj || obj == getOwner()) computeLink();
    }

    void sleepChanged(Node* ) { /*computeLink();*/ }

protected:
    typename T::SPtr m_link;
    ActivatorLink m_actiavtor;
};

}

template<class T>
using DataLink=sofaTypes::DataLink<T>;


}

