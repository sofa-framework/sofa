/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_QT_SIMPLEDATAWIDGET_H
#define SOFA_GUI_QT_SIMPLEDATAWIDGET_H

#include "DataWidget.h"
#include "ModifyObject.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
//#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/core/topology/Topology.h>
#include "WDoubleLineEdit.h"
#include <limits.h>

#include <sstream>
#include <sofa/helper/Polynomial_LD.inl>
#include <sofa/helper/OptionsGroup.h>

#include <functional>

#include <QLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QComboBox>

#if !defined(INFINITY)
#define INFINITY 9.0e10
#endif
namespace sofa
{

namespace gui
{

namespace qt
{

using sofa::helper::Quater;

/// This class is used to specify how to graphically represent a data type,
/// by default using a simple QLineEdit
template<class T>
class data_widget_trait
{
public:
    typedef T data_type;
    typedef QLineEdit Widget;
    static Widget* create(QWidget* parent, const data_type& /*d*/)
    {
        Widget* w = new Widget(parent);
        return w;
    }
    static void readFromData(Widget* w, const data_type& d)
    {
        std::ostringstream o;
        o << d;
        if (o.str() != w->text().toStdString())
            w->setText(QString(o.str().c_str()));
    }
    static void writeToData(Widget* w, data_type& d)
    {
        std::string s = w->text().toStdString();
        std::istringstream i(s);
        i >> d;
    }
    static void setReadOnly(Widget* w, bool readOnly)
    {
        w->setEnabled(!readOnly);
        w->setReadOnly(readOnly);
    }
    static void connectChanged(Widget* w, DataWidget* datawidget)
    {
        datawidget->connect(w, SIGNAL( textChanged(const QString&) ), datawidget, SLOT(setWidgetDirty()));
    }
};





/// This class is used to create and manage the GUI of a data type,
/// using data_widget_trait to know which widgets to use
template<class T>
class data_widget_container
{
public:
    typedef T data_type;
    typedef data_widget_trait<data_type> helper;
    typedef typename helper::Widget Widget;
    typedef QHBoxLayout Layout;
    Widget* w;
    Layout* container_layout;
    data_widget_container() : w(NULL),container_layout(NULL) {  }

    bool createLayout(DataWidget* parent)
    {
        if(parent->layout() != NULL) return false;
        container_layout = new QHBoxLayout(parent);
        //parent->setLayout(container_layout);
        return true;
    }

    bool createLayout(QLayout* layout)
    {
        if(container_layout) return false;
        container_layout = new QHBoxLayout();
        layout->addItem(container_layout);
        return true;
    }

    bool createWidgets(DataWidget* parent, const data_type& d, bool readOnly)
    {
        w = helper::create(parent,d);
        if (w == NULL) return false;

        helper::readFromData(w, d);
        if (readOnly)
            helper::setReadOnly(w, readOnly);
        else
            helper::connectChanged(w, parent);
        return true;
    }
    void setReadOnly(bool readOnly)
    {
        if(w){
            w->setEnabled(!readOnly);
            helper::setReadOnly(w, readOnly);
        }
    }
    void readFromData(const data_type& d)
    {
        helper::readFromData(w, d);
    }
    void writeToData(data_type& d)
    {
        helper::writeToData(w, d);
    }

    void insertWidgets()
    {
        assert(w);
        container_layout->addWidget(w);
    }
};

/// This class manages the GUI of a BaseData, using the corresponding instance of data_widget_container
template<class T, class Container = data_widget_container<T> >
class SimpleDataWidget : public TDataWidget<T>
{

protected:
    typedef T data_type;
    Container container;
    typedef data_widget_trait<data_type> helper;


public:
    typedef sofa::core::objectmodel::Data<T> MyTData;
    SimpleDataWidget(QWidget* parent,const char* name, MyTData* d):
        TDataWidget<T>(parent,name,d)
    {}
    virtual bool createWidgets()
    {
        const data_type& d = this->getData()->virtualGetValue();
        if (!container.createWidgets(this, d, ! this->isEnabled() ) )
            return false;

        container.createLayout(this);
        container.insertWidgets();

        return true;
    }
    virtual void setDataReadOnly(bool readOnly)
    {
        container.setReadOnly(readOnly);
    }

    virtual void readFromData()
    {
        container.readFromData(this->getData()->getValue());
    }

    virtual void setReadOnly(bool readOnly)
    {
        container.setReadOnly(readOnly);
    }

    virtual void writeToData()
    {
        data_type& d = *this->getData()->beginEdit();
        container.writeToData(d);
        this->getData()->endEdit();
    }
    virtual unsigned int numColumnWidget() { return 5; }
};

////////////////////////////////////////////////////////////////
/// std::string support
////////////////////////////////////////////////////////////////

template<>
class data_widget_trait < std::string >
{
public:
    typedef std::string data_type;
    typedef QLineEdit Widget;
    static Widget* create(QWidget* parent, const data_type& /*d*/)
    {
        Widget* w = new Widget(parent);
        w->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
        return w;
    }
    static void readFromData(Widget* w, const data_type& d)
    {
        if (w->text().toStdString() != d)
            w->setText(QString(d.c_str()));
    }
    static void writeToData(Widget* w, data_type& d)
    {
        d = w->text().toStdString();
    }
    static void setReadOnly(Widget* w, bool readOnly)
    {
        w->setEnabled(!readOnly);
        w->setReadOnly(readOnly);
    }
    static void connectChanged(Widget* w, DataWidget* datawidget)
    {
        datawidget->connect(w, SIGNAL( textChanged(const QString&) ), datawidget, SLOT(setWidgetDirty()) );
    }
};

////////////////////////////////////////////////////////////////
/// bool support
////////////////////////////////////////////////////////////////

template<>
class data_widget_trait < bool >
{
public:
    typedef bool data_type;
    typedef QCheckBox Widget;
    static Widget* create(QWidget* parent, const data_type& /*d*/)
    {
        Widget* w = new Widget(parent);
        return w;
    }
    static void readFromData(Widget* w, const data_type& d)
    {
        if (w->isChecked() != d)
            w->setChecked(d);
    }
    static void writeToData(Widget* w, data_type& d)
    {
        d = (data_type) (w->isChecked());
    }
    static void setReadOnly(Widget* w, bool readOnly)
    {
        w->setEnabled(!readOnly);
    }
    static void connectChanged(Widget* w, DataWidget* datawidget)
    {
        datawidget->connect(w, SIGNAL( toggled(bool) ), datawidget, SLOT(setWidgetDirty()));
    }
};

////////////////////////////////////////////////////////////////
/// float and double support
////////////////////////////////////////////////////////////////

template < typename T >
class real_data_widget_trait
{
public:
    typedef T data_type;
    typedef WDoubleLineEdit Widget;
    static Widget* create(QWidget* parent, const data_type& /*d*/)
    {
        Widget* w = new Widget(parent, "real");

        w->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
        w->setMinValue( (data_type)-INFINITY );
        w->setMaxValue( (data_type)INFINITY );
        w->setMinimumWidth(20);
        return w;
    }
    static void readFromData(Widget* w, const data_type& d)
    {
        if (d != w->getDisplayedValue())
            w->setValue(d);
    }
    static void writeToData(Widget* w, data_type& d)
    {
        d = (data_type) w->getDisplayedValue();
    }
    static void setReadOnly(Widget* w, bool readOnly)
    {
        w->setEnabled(!readOnly);
    }
    static void connectChanged(Widget* w, DataWidget* datawidget)
    {
        datawidget->connect(w, SIGNAL( textChanged(const QString&) ), datawidget, SLOT(setWidgetDirty()));
    }
};

template<>
class data_widget_trait < float > : public real_data_widget_trait< float >
{};

template<>
class data_widget_trait < double > : public real_data_widget_trait< double >
{};

////////////////////////////////////////////////////////////////
/// int, unsigned int, char and unsigned char support
////////////////////////////////////////////////////////////////

template<class T, int vmin, int vmax>
class int_data_widget_trait
{
public:
    typedef T data_type;
    typedef QSpinBox Widget;
    static Widget* create(QWidget* parent, const data_type& /*d*/)
    {
        Widget* w = new Widget(parent);
        w->setMinimum(vmin);
        w->setMaximum(vmax);
        w->setSingleStep(1);

        w->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
        return w;
    }
    static void readFromData(Widget* w, const data_type& d)
    {
        if ((int)d != w->value())
            w->setValue((int)d);
    }
    static void writeToData(Widget* w, data_type& d)
    {
        d = (data_type) w->value();
    }
    static void setReadOnly(Widget* w, bool readOnly)
    {
        w->setEnabled(!readOnly);
    }
    static void connectChanged(Widget* w, DataWidget* datawidget)
    {
        datawidget->connect(w, SIGNAL( valueChanged(int) ), datawidget, SLOT(setWidgetDirty()));
    }
};

template<>
class data_widget_trait < int > : public int_data_widget_trait < int, INT_MIN, INT_MAX >
{};

template<>
class data_widget_trait < unsigned int > : public int_data_widget_trait < unsigned int, 0, INT_MAX >
{};

template<>
class data_widget_trait < char > : public int_data_widget_trait < char, -128, 127 >
{};

template<>
class data_widget_trait < unsigned char > : public int_data_widget_trait < unsigned char, 0, 255 >
{};

////////////////////////////////////////////////////////////////
/// arrays and vectors support
////////////////////////////////////////////////////////////////

/// This class is used to get properties of a data type in order to display it as a table or a list
template<class T>
class vector_data_trait
{
public:

    typedef T data_type;
    /// Type of a row if this data type is viewed in a table or list
    typedef T value_type;
    /// Number of dimensions of this data type
    enum { NDIM = 0 };
    enum { SIZE = 1 };
    /// Get the number of rows
    static int size(const data_type&) { return SIZE; }
    /// Get the name of a row, or NULL if the index should be used instead
    static const char* header(const data_type& /*d*/, int /*i*/ = 0)
    {
        return NULL;
    }
    /// Get a row
    static const value_type* get(const data_type& d, int i = 0)
    {
        return (i == 0) ? &d : NULL;
    }
    /// Set a row
    static void set( const value_type& v, data_type& d, int i = 0)
    {
        if (i == 0)
            d = v;
    }
    /// Resize
    static void resize( int /*s*/, data_type& /*d*/)
    {
    }
};


template<class T, class Container = data_widget_container< typename vector_data_trait<T>::value_type> >
class fixed_vector_data_widget_container
{
public:
    typedef T data_type;
    typedef vector_data_trait<data_type> vhelper;
    typedef typename vhelper::value_type value_type;
    typedef QHBoxLayout Layout;
    enum { N = vhelper::SIZE };
    Container w[N];
    Layout* container_layout;

    fixed_vector_data_widget_container():container_layout(NULL) {}

    bool createLayout(DataWidget* parent)
    {
        if(parent->layout() != NULL) return false;
        container_layout = new QHBoxLayout(parent);
        return true;
    }

    bool createLayout(QLayout* layout)
    {
        if(container_layout) return false;
        container_layout = new QHBoxLayout();
        layout->addItem(container_layout);
        return true;
    }

    bool createWidgets(DataWidget* parent, const data_type& d, bool readOnly)
    {
        for (int i=0; i<N; ++i)
            if (!w[i].createWidgets(parent, *vhelper::get(d,i), readOnly))
                return false;

        return true;
    }
    void setReadOnly(bool readOnly)
    {

        for (int i=0; i<N; ++i)
            w[i].setReadOnly(readOnly);
    }
    void readFromData(const data_type& d)
    {
        for (int i=0; i<N; ++i)
            w[i].readFromData(*vhelper::get(d,i));
    }
    void writeToData(data_type& d)
    {
        for (int i=0; i<N; ++i)
        {
            value_type v = *vhelper::get(d,i);
            w[i].writeToData(v);
            vhelper::set(v,d,i);
        }
    }

    void insertWidgets()
    {
        for (int i=0; i<N; ++i)
        {
            assert(w[i].w != NULL);
            container_layout->addWidget(w[i].w);
        }
    }
};

template<class T, class Container = data_widget_container< typename vector_data_trait< typename vector_data_trait<T>::value_type >::value_type> >
class fixed_grid_data_widget_container
{
public:

    typedef T data_type;
    typedef vector_data_trait<data_type> rhelper;
    typedef typename rhelper::value_type row_type;
    typedef vector_data_trait<row_type> vhelper;
    typedef typename vhelper::value_type value_type;
    enum { L = rhelper::SIZE };
    enum { C = vhelper::SIZE };
    typedef QGridLayout Layout;
    Container w[L][C];
    Layout* container_layout;
    fixed_grid_data_widget_container():container_layout(NULL) {}

    bool createLayout(QWidget* parent)
    {
        if( parent->layout() != NULL ) return false;
        container_layout = new Layout(parent /*,L,C */);
        return true;
    }
    bool createLayout(QLayout* layout)
    {
        if(container_layout != NULL ) return false;
        container_layout = new Layout( /*,L,C */);
        layout->addItem(container_layout);
        return true;
    }

    bool createWidgets(DataWidget* parent, const data_type& d, bool readOnly)
    {
        for (int y=0; y<L; ++y)
            for (int x=0; x<C; ++x)
                if (!w[y][x].createWidgets( parent, *vhelper::get(*rhelper::get(d,y),x), readOnly))
                    return false;
        return true;
    }
    void setReadOnly(bool readOnly)
    {
        for (int y=0; y<L; ++y)
            for (int x=0; x<C; ++x)
                w[y][x].setReadOnly(readOnly);
    }
    void readFromData(const data_type& d)
    {
        for (int y=0; y<L; ++y)
            for (int x=0; x<C; ++x)
                w[y][x].readFromData(*vhelper::get(*rhelper::get(d,y),x));
    }
    void writeToData(data_type& d)
    {
        for (int y=0; y<L; ++y)
        {
            row_type r = *rhelper::get(d,y);
            for (int x=0; x<C; ++x)
            {
                value_type v = *vhelper::get(r,x);
                w[y][x].writeToData(v);
                vhelper::set(v,r,x);
            }
            rhelper::set(r,d,y);
        }
    }

    void insertWidgets()
    {
        assert(container_layout);
        for (int y=0; y<L; ++y)
        {
            for (int x=0; x<C; ++x)
            {
                container_layout->addWidget(w[y][x].w,y,x);
            }
        }
    }
};

////////////////////////////////////////////////////////////////
/// sofa::helper::fixed_array support
////////////////////////////////////////////////////////////////

template<class T, std::size_t N>
class vector_data_trait < sofa::helper::fixed_array<T, N> >
{
public:
    typedef sofa::helper::fixed_array<T, N> data_type;
    typedef T value_type;
    enum { NDIM = 1 };
    enum { SIZE = N };
    static int size(const data_type&) { return SIZE; }
    static const char* header(const data_type& /*d*/, int /*i*/ = 0)
    {
        return NULL;
    }
    static const value_type* get(const data_type& d, int i = 0)
    {
        return ((unsigned)i < (unsigned)size(d)) ? &(d[i]) : NULL;
    }
    static void set( const value_type& v, data_type& d, int i = 0)
    {
        if ((unsigned)i < (unsigned)size(d))
            d[i] = v;
    }
    static void resize( int /*s*/, data_type& /*d*/)
    {
    }

};

template<class T, std::size_t N>
class data_widget_container < sofa::helper::fixed_array<T, N> > : public fixed_vector_data_widget_container < sofa::helper::fixed_array<T, N> >
{};


////////////////////////////////////////////////////////////////
/// Topological edges/triangles/... support
////////////////////////////////////////////////////////////////

template<>
class vector_data_trait < sofa::core::topology::Topology::Edge >
    : public vector_data_trait < sofa::helper::fixed_array < sofa::core::topology::Topology::PointID, 2 > >
{
};

template<>
class data_widget_container < sofa::core::topology::Topology::Edge > : public fixed_vector_data_widget_container < sofa::core::topology::Topology::Edge >
{};

template<>
class vector_data_trait < sofa::core::topology::Topology::Triangle >
    : public vector_data_trait < sofa::helper::fixed_array < sofa::core::topology::Topology::PointID, 3 > >
{
};

template<>
class data_widget_container < sofa::core::topology::Topology::Triangle > : public fixed_vector_data_widget_container < sofa::core::topology::Topology::Triangle >
{};

template<>
class vector_data_trait < sofa::core::topology::Topology::Quad >
    : public vector_data_trait < sofa::helper::fixed_array < sofa::core::topology::Topology::PointID, 4 > >
{
};

template<>
class data_widget_container < sofa::core::topology::Topology::Quad > : public fixed_vector_data_widget_container < sofa::core::topology::Topology::Quad >
{};

template<>
class vector_data_trait < sofa::core::topology::Topology::Tetrahedron >
    : public vector_data_trait < sofa::helper::fixed_array < sofa::core::topology::Topology::PointID, 4 > >
{
};

template<>
class data_widget_container < sofa::core::topology::Topology::Tetrahedron > : public fixed_vector_data_widget_container < sofa::core::topology::Topology::Tetrahedron >
{};

template<>
class vector_data_trait < sofa::core::topology::Topology::Hexahedron >
    : public vector_data_trait < sofa::helper::fixed_array < sofa::core::topology::Topology::PointID, 8 > >
{
};

template<>
class data_widget_container < sofa::core::topology::Topology::Hexahedron > : public fixed_vector_data_widget_container < sofa::core::topology::Topology::Hexahedron >
{};

////////////////////////////////////////////////////////////////
/// sofa::defaulttype::Vec support
////////////////////////////////////////////////////////////////

template<int N, class T>
class vector_data_trait < sofa::defaulttype::Vec<N, T> >
{
public:
    typedef sofa::defaulttype::Vec<N, T> data_type;
    typedef T value_type;
    enum { NDIM = 1 };
    enum { SIZE = N };
    static int size(const data_type&) { return SIZE; }
    static const char* header(const data_type& /*d*/, int /*i*/ = 0)
    {
        return NULL;
    }
    static const value_type* get(const data_type& d, int i = 0)
    {
        return ((unsigned)i < (unsigned)size(d)) ? &(d[i]) : NULL;
    }
    static void set( const value_type& v, data_type& d, int i = 0)
    {
        if ((unsigned)i < (unsigned)size(d))
            d[i] = v;
    }
    static void resize( int /*s*/, data_type& /*d*/)
    {
    }
};

template<>
inline const char* vector_data_trait < sofa::defaulttype::Vec<2, float> >::header(const data_type& /*d*/, int i)
{
    switch(i)
    {
    case 0: return "X";
    case 1: return "Y";
    }
    return NULL;
}

template<>
inline const char* vector_data_trait < sofa::defaulttype::Vec<2, double> >::header(const data_type& /*d*/, int i)
{
    switch(i)
    {
    case 0: return "X";
    case 1: return "Y";
    }
    return NULL;
}

template<>
inline const char* vector_data_trait < sofa::defaulttype::Vec<3, float> >::header(const data_type& /*d*/, int i)
{
    switch(i)
    {
    case 0: return "X";
    case 1: return "Y";
    case 2: return "Z";
    }
    return NULL;
}

template<>
inline const char* vector_data_trait < sofa::defaulttype::Vec<3, double> >::header(const data_type& /*d*/, int i)
{
    switch(i)
    {
    case 0: return "X";
    case 1: return "Y";
    case 2: return "Z";
    }
    return NULL;
}

template<int N, class T>
class data_widget_container < sofa::defaulttype::Vec<N, T> > : public fixed_vector_data_widget_container < sofa::defaulttype::Vec<N, T> >
{};

////////////////////////////////////////////////////////////////
/// std::helper::Quater support
////////////////////////////////////////////////////////////////

template<class T>
class vector_data_trait < Quater<T> >
{
public:
    typedef Quater<T> data_type;
    typedef T value_type;
    enum { NDIM = 1 };
    enum { SIZE = 4 };
    static int size(const data_type&) { return SIZE; }
    static const char* header(const data_type& /*d*/, int i = 0)
    {
        switch(i)
        {
        case 0: return "qX";
        case 1: return "qY";
        case 2: return "qZ";
        case 3: return "qW";
        }
        return NULL;
    }
    static const value_type* get(const data_type& d, int i = 0)
    {
        return ((unsigned)i < (unsigned)size(d)) ? &(d[i]) : NULL;
    }
    static void set( const value_type& v, data_type& d, int i = 0)
    {
        if ((unsigned)i < (unsigned)size(d))
            d[i] = v;
    }
    static void resize( int /*s*/, data_type& /*d*/)
    {
    }
};

template<class T>
class data_widget_container < Quater<T> > : public fixed_vector_data_widget_container < Quater<T> >
{};


////////////////////////////////////////////////////////////////
/// sofa::helper::Polynomial_LD support
////////////////////////////////////////////////////////////////
using sofa::helper::Polynomial_LD;

template<typename Real, unsigned int N>
class data_widget_trait < Polynomial_LD<Real,N> >
{
public:
    typedef Polynomial_LD<Real,N> data_type;
    typedef QLineEdit Widget;
    static Widget* create(QWidget* parent, const data_type& )
    {
        Widget* w = new Widget(parent);
        return w;
    }
    static void readFromData(Widget* w, const data_type& d)
    {
        int length = (int) d.getString().length();
        if (w->text().toStdString() != d.getString())
        {
            w->setMaxLength(length+2); w->setReadOnly(true);
            w->setText(QString(d.getString().c_str()));
        }
    }
    static void writeToData(Widget* , data_type& )
    {
    }
    static void setReadOnly(Widget* w, bool readOnly)
    {
        w->setEnabled(!readOnly);
        w->setReadOnly(readOnly);
    }
    static void connectChanged(Widget* w, DataWidget* datawidget)
    {
        datawidget->connect(w, SIGNAL( textChanged(const QString&) ), datawidget, SLOT(setWidgetDirty()) );
    }
};


#ifdef TODOLINK
////////////////////////////////////////////////////////////////
/// sofa::core::objectmodel::ObjectRef
////////////////////////////////////////////////////////////////

using sofa::core::objectmodel::ObjectRef;

template<>
class data_widget_trait < ObjectRef >
{
public:
    typedef ObjectRef data_type;
    typedef QLineEdit Widget;
    static Widget* create(QWidget* parent, const data_type& d)
    {
        Widget* w = new Widget(parent);
        w->setText(QString(d.getPath().c_str()));
        return w;
    }
    static void readFromData(Widget* w, const data_type& d)
    {
        std::ostringstream _outref; _outref<<d;
        if (w->text().toStdString() != _outref.str())
            w->setText(QString(_outref.str().c_str()));
    }
    static void writeToData(Widget* w, data_type& d)
    {
        bool canwrite = d.setPath ( w->text().toStdString() );
        if(!canwrite)
            msg_info()<<"canot set Path "<<w->text().toStdString()<<std::endl;
    }
    static void setReadOnly(Widget* w, bool readOnly)
    {
        w->setReadOnly(readOnly);
    }
    static void connectChanged(Widget* w, DataWidget* datawidget)
    {
        datawidget->connect(w, SIGNAL( textChanged(const QString&) ), datawidget, SLOT(setWidgetDirty()) );
    }
};

////////////////////////////////////////////////////////////////
/// support sofa::core::objectmodel::VectorObjectRef;
////////////////////////////////////////////////////////////////

using sofa::core::objectmodel::VectorObjectRef;
template<>
class vector_data_trait < sofa::core::objectmodel::VectorObjectRef >
{
public:
    typedef sofa::core::objectmodel::VectorObjectRef data_type;
    typedef sofa::core::objectmodel::ObjectRef       value_type;

    static int size(const data_type& d) { return d.size(); }
    static const char* header(const data_type& , int i = 0)
    {
        std::ostringstream _header; _header<<i;
        return ("Path " + _header.str()).c_str();
    }
    static const value_type* get(const data_type& d, int i = 0)
    {
        return ((unsigned)i < (unsigned)size(d)) ? &(d[i]) : NULL;
    }
    static void set( const value_type& v, data_type& d, int i = 0)
    {
        if ((unsigned)i < (unsigned)size(d))
            d[i] = v;
    }
    static void resize( int /*s*/, data_type& /*d*/)
    {
    }
};
#endif

////////////////////////////////////////////////////////////////
/// sofa::defaulttype::Mat support
////////////////////////////////////////////////////////////////

template<int L, int C, class T>
class vector_data_trait < sofa::defaulttype::Mat<L, C, T> >
{
public:
    typedef sofa::defaulttype::Mat<L, C, T> data_type;
    typedef typename data_type::Line value_type;
    enum { NDIM = 1 };
    enum { SIZE = L };
    static int size(const data_type&) { return SIZE; }
    static const char* header(const data_type& /*d*/, int /*i*/ = 0)
    {
        return NULL;
    }
    static const value_type* get(const data_type& d, int i = 0)
    {
        return ((unsigned)i < (unsigned)size(d)) ? &(d[i]) : NULL;
    }
    static void set( const value_type& v, data_type& d, int i = 0)
    {
        if ((unsigned)i < (unsigned)size(d))
            d[i] = v;
    }
    static void resize( int /*s*/, data_type& /*d*/)
    {
    }
};

template<int L, int C, class T>
class data_widget_container < sofa::defaulttype::Mat<L, C, T> > : public fixed_grid_data_widget_container < sofa::defaulttype::Mat<L, C, T> >
{};

////////////////////////////////////////////////////////////////
/// OptionsGroup support
////////////////////////////////////////////////////////////////


class RadioDataWidget : public TDataWidget<sofa::helper::OptionsGroup >
{
    Q_OBJECT
public :

    ///The class constructor takes a TData<RadioTrick> since it creates
    ///a widget for a that particular data type.
    RadioDataWidget(QWidget* parent, const char* name,
            core::objectmodel::Data<sofa::helper::OptionsGroup >* m_data)
        : TDataWidget<sofa::helper::OptionsGroup >(parent,name,m_data) {};

    ///In this method we  create the widgets and perform the signal / slots connections.
    virtual bool createWidgets();
    virtual void setDataReadOnly(bool readOnly);

protected:
    ///Implements how update the widgets knowing the data value.
    virtual void readFromData();

    ///Implements how to update the data, knowing the widget value.
    virtual void writeToData();

    QButtonGroup *buttonList;
    QComboBox    *comboList;
    bool buttonMode;
};


} // namespace qt

} // namespace gui

} // namespace sofa


#endif
