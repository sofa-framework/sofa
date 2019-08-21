#pragma once

#include <sofa/gui/qt/DataWidget.h>

namespace sofa {

namespace gui {

namespace qt {


template<class DATA,class WIDGET>
class GenericDataWidget : public sofa::gui::qt::DataWidget {
public:
    typedef DATA MyData;
    typedef WIDGET MyWidget;

    GenericDataWidget(QWidget* parent,const char* name, MyData* d)
    : sofa::gui::qt::DataWidget(parent,name,d)
    , m_data(d){}

    virtual bool createWidgets() {
        m_widget = new MyWidget(this, *m_data);
        m_widget->setParent(this);
        setLayout(new QVBoxLayout(this));
        layout()->addWidget(m_widget);
        m_widget->setVisible(true);
        readFromData();
        return true;
    }

    template <class RealObject>
    static RealObject* create( RealObject*, CreatorArgument& arg)
    {
        typename RealObject::MyData* realData = dynamic_cast< typename RealObject::MyData* >(arg.data);
        if (!realData) return NULL;
        else
        {
            RealObject* obj = new RealObject(arg.parent,arg.name.c_str(), realData);
            if( !obj->createWidgets() )
            {
                delete obj;
                obj = NULL;
            }
            if (obj)
            {
                obj->setDataReadOnly(arg.readOnly);
            }
            return obj;
        }

    }


    virtual void setDataReadOnly(bool readOnly) {
        m_widget->setEnabled(!readOnly);
    }

    virtual void readFromData() {
        m_widget->readFromData(*m_data);
    }

    virtual void writeToData() {
        m_widget->writeToData(*m_data);
    }

protected:
    MyData * m_data;
    MyWidget * m_widget;
};

}

}

template<class DATA,class WIDGET>
using GenericDataWidget=sofa::gui::qt::GenericDataWidget<DATA,WIDGET>;

}
