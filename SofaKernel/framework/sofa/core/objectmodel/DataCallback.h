#pragma once

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/simulation/Node.h>

namespace sofa {

namespace core {

namespace objectmodel {

class DataCallback : public sofa::core::objectmodel::DDGNode {
public:

    DataCallback(sofa::core::objectmodel::BaseData & data) {
        m_data = &data;
        m_updating = false;
        addInput(&data);
    }

    template<class FwdObject,class FwdFunction>
    void addCallback(FwdObject * obj,FwdFunction f) {
        m_callback.push_back(std::unique_ptr<Callback>(new CallbackImpl<FwdObject,FwdFunction>(obj,f)));
    }

    virtual void setDirtyValue(const core::ExecParams* params) {
        update();
        cleanDirtyOutputsOfInputs(params);
    }

    virtual void update() {
        if (! m_updating) {
            m_updating = true;
            for (unsigned i=0;i<m_callback.size();i++) m_callback[i]->apply();
            m_updating = false;
        }
    }

    const std::string& getName() const {
        return m_data->getName();
    }

    sofa::core::objectmodel::Base* getOwner() const {
        return m_data->getOwner();
    }

    sofa::core::objectmodel::BaseData* getData() const {
        return m_data;
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
    sofa::core::objectmodel::BaseData * m_data;
};

}

}

using DataCallback=core::objectmodel::DataCallback;

}
