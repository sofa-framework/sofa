#pragma once

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/simulation/Node.h>

namespace sofa {

namespace core {

namespace objectmodel {

class DataCallback : public sofa::core::objectmodel::DDGNode {
public:

    typedef sofa::core::objectmodel::BaseData BaseData;

    //Constructor with multiple data
    DataCallback(BaseData * data) {
        m_data.push_back(data);
        m_updating = false;
        addInput(data);
    }

    DataCallback(std::initializer_list<BaseData*> listdata) {

        for(BaseData* data : listdata)
        {
            m_data.push_back(data);
            m_updating = false;
            addInput(data);
        }
    }

    template<class FwdObject,class FwdFunction>
    void addCallback(FwdObject * obj,FwdFunction f) {
        m_callback.push_back(std::unique_ptr<Callback>(new CallbackImpl<FwdObject,FwdFunction>(obj,f)));
    }

    //Spectific function to avoid passing this as parameter
    template<typename FwdObject>
    void addCallback(void (FwdObject::* f)()) {
        typedef void (FwdObject::* FwdFunction)();
        FwdObject * obj = dynamic_cast<FwdObject*>(getOwner());
        if (obj != NULL) m_callback.push_back(std::unique_ptr<Callback>(new CallbackImpl<FwdObject,FwdFunction>(obj,f)));
        else std::cerr << "Error DataCallback : cannot bind the function with this object type" << std::endl;
    }

    void notifyEndEdit(const core::ExecParams* params) override {
        if (! m_updating) {
            m_updating = true;
            for (unsigned i=0;i<m_callback.size();i++) m_callback[i]->apply();
            sofa::core::objectmodel::DDGNode::notifyEndEdit(params);
            m_updating = false;
        }
    }

    virtual void update() override {}

    const std::string& getName() const override {
        assert(m_data.size() == 0);
        return m_data[0]->getName();
    }

    sofa::core::objectmodel::Base* getOwner() const override {
        assert(m_data.size() == 0);
        return m_data[0]->getOwner();
    }

    sofa::core::objectmodel::BaseData* getData() const override {
        assert(m_data.size() == 0);
        return m_data[0];
    }


private:

    class Callback {
    public:
        virtual ~Callback() {}

        virtual void apply() = 0;
    };

    template<class FwdObject,class FwdFunction>
    class CallbackImpl : public Callback {
    public:

        CallbackImpl(FwdObject * obj,FwdFunction f) : m_object(obj), m_function(f) {}

        void apply() {
            (m_object->*m_function)();
        }

        FwdObject * m_object;
        FwdFunction m_function;
    };

    bool m_updating;
    std::vector<std::unique_ptr<Callback> > m_callback;
    std::vector<sofa::core::objectmodel::BaseData*> m_data;

};

}

}

using DataCallback=core::objectmodel::DataCallback;

}
