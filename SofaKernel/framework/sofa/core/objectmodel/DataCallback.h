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
    DataCallback(BaseData & data) {
        m_data.push_back(&data);
        m_updating = false;
        addInput(&data);
    }

    DataCallback(BaseData & data1,BaseData & data2)
    : DataCallback(data1) {
        m_data.push_back(&data2);
        addInput(&data2);
    }

    DataCallback(BaseData & data1,BaseData & data2,BaseData & data3)
    : DataCallback(data1,data2) {
        m_data.push_back(&data3);
        addInput(&data3);
    }

    DataCallback(BaseData & data1,BaseData & data2,BaseData & data3,BaseData & data4)
    : DataCallback(data1,data2,data3) {
        m_data.push_back(&data4);
        addInput(&data4);
    }

    DataCallback(BaseData & data1,BaseData & data2,BaseData & data3,BaseData & data4,BaseData & data5)
    : DataCallback(data1,data2,data3,data4) {
        m_data.push_back(&data5);
        addInput(&data5);
    }

    DataCallback(BaseData & data1,BaseData & data2,BaseData & data3,BaseData & data4,BaseData & data5,BaseData & data6)
    : DataCallback(data1,data2,data3,data4,data5) {
        m_data.push_back(&data6);
        addInput(&data6);
    }

    DataCallback(BaseData & data1,BaseData & data2,BaseData & data3,BaseData & data4,BaseData & data5,BaseData & data6,BaseData & data7)
    : DataCallback(data1,data2,data3,data4,data5,data6) {
        m_data.push_back(&data7);
        addInput(&data7);
    }

    DataCallback(BaseData & data1,BaseData & data2,BaseData & data3,BaseData & data4,BaseData & data5,BaseData & data6,BaseData & data7,BaseData & data8)
    : DataCallback(data1,data2,data3,data4,data5,data6,data7) {
        m_data.push_back(&data8);
        addInput(&data8);
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
