/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include "SimpleDataWidget.h"
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/component/solidmechanics/spring/SpringForceField.h>
#include <sofa/component/solidmechanics/spring/JointSpring.h>
#include <sofa/component/solidmechanics/spring/JointSpringForceField.h>
#include <sofa/component/solidmechanics/spring/GearSpringForceField.h>

#include <sofa/helper/io/Mesh.h>
#include <sofa/type/RGBAColor.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLayout>

namespace sofa::gui::qt
{

////////////////////////////////////////////////////////////////
/// Generic data structures support
////////////////////////////////////////////////////////////////

template<class T>
class struct_data_trait
{
public:
    typedef T data_type;
    enum { NVAR = 1 };
    static void set( data_type& /*d*/ )
    {
    }
};

template<class T, int I>
class struct_data_trait_var
{
public:
    typedef T data_type;
    typedef T value_type;
    static const char* name() { return nullptr; }
    static const value_type* get(const data_type& d) { return &d; }
    static void set( const value_type& v, data_type& d) { d = v; }
    static bool readOnly() { return false; }
    static bool isCheckable() { return false; }
    static bool isChecked(const data_type& /*d*/) { return true; }
    static void setChecked(bool /*b*/, data_type& /*d*/) {}
};

template<class T, int N = struct_data_trait<T>::NVAR >
class struct_data_widget_container
{
public:
    typedef T data_type;
    typedef struct_data_trait<data_type> shelper;
    typedef struct_data_trait_var<data_type,N-1> vhelper;
    typedef typename vhelper::value_type value_type;
    typedef data_widget_container<value_type> Container;
    typedef struct_data_widget_container<data_type,N-1> PrevContainer;
    typedef QVBoxLayout MasterLayout;
    typedef QHBoxLayout Layout;
    PrevContainer p;
    Container w;
    QCheckBox* check;
    QLabel* label;
    Layout* container_layout;
    MasterLayout* master_layout;
    struct_data_widget_container() : check(nullptr),label(nullptr),container_layout(nullptr),master_layout(nullptr) {}

    void setMasterLayout(MasterLayout* layout)
    {
        p.setMasterLayout(layout);
        master_layout = layout;
    }

    bool createLayout( DataWidget* parent )
    {
        if( parent->layout() != nullptr )
        {
            return false;
        }
        master_layout = new QVBoxLayout(parent);
        setMasterLayout(master_layout);
        return true;
    }

    bool createLayout( QLayout* layout)
    {
        container_layout = new QHBoxLayout();
        layout->addItem(container_layout);
        return true;
    }

    void insertWidgets()
    {
        p.insertWidgets();
        createLayout(master_layout);

        if(check)
        {
            container_layout->addWidget(check);
        }
        if(label)
        {
            container_layout->addWidget(label);
        }
        w.createLayout(container_layout); // create the layout for the rest of the widgets
        w.insertWidgets(); // insert them accordingly

    }

    bool createWidgets(DataWidget * parent, const data_type& d, bool readOnly )
    {

        if (!p.createWidgets( parent, d, readOnly))
            return false;

        const char* name = vhelper::name();
        const bool checkable = vhelper::isCheckable();
        if (checkable)
        {
            check = new QCheckBox(parent);
            if (name && *name)
            {
                check->setText(QString(name));
            }
        }
        else
        {
            if (name && *name && N > 1)
            {
                label = new QLabel(QString(name),parent);
            }
        }
        if (!w.createWidgets(parent, *vhelper::get(d), readOnly || vhelper::readOnly()))
            return false;

        if (checkable)
        {
            const bool isChecked = vhelper::isChecked(d);
            check->setChecked(isChecked);
            if (readOnly || vhelper::readOnly())
                check->setEnabled(false);
            else
            {
                if (!isChecked)
                    w.setReadOnly(true);
                parent->connect(check, SIGNAL( toggled(bool) ),parent, SLOT( setWidgetDirty() ));
                parent->connect(check, SIGNAL( toggled(bool) ),parent, SLOT( setReadOnly(bool) ));

            }
        }
        return true;
    }

    void setReadOnly(bool readOnly)
    {
        p.setReadOnly(readOnly);
        w.setReadOnly(readOnly || vhelper::readOnly() || (check && !(check->checkState() == Qt::Checked)));
    }
    void readFromData(const data_type& d)
    {
        p.readFromData(d);
        if (check)
        {
            const bool wasChecked = (check->checkState() == Qt::Checked);
            const bool isChecked = vhelper::isChecked(d);
            if (isChecked != wasChecked)
            {
                check->setChecked(isChecked);
                if (check->isEnabled())
                    w.setReadOnly(!isChecked);
            }
        }
        w.readFromData(*vhelper::get(d));
    }
    void readConstantsFromData(const data_type& d)
    {
        p.readConstantsFromData(d);
        if (vhelper::readOnly())
        {
            if (check)
            {
                check->setChecked(vhelper::isChecked(d));
            }
            w.readFromData(*vhelper::get(d));
        }
    }
    void writeToData(data_type& d)
    {
        p.writeToData(d);
        if (check)
        {
            bool isChecked = (check->checkState() == Qt::Checked);
            vhelper::setChecked(isChecked, d);
        }
        value_type v = *vhelper::get(d);
        w.writeToData(v);
        vhelper::set(v,d);
        if ( N == struct_data_trait<T>::NVAR )
        {
            shelper::set(d);
            readConstantsFromData(d); // reread constant fields
        }
    }
};

template<class T>
class struct_data_widget_container< T, 0 >
{
public:
    typedef T data_type;
    typedef struct_data_trait<data_type> shelper;
    typedef QVBoxLayout MasterLayout;
    typedef QHBoxLayout Layout ;
    MasterLayout* master_layout;
    Layout* container_layout;
    struct_data_widget_container():master_layout(nullptr),container_layout(nullptr) {}

    void setMasterLayout(MasterLayout* /*layout*/)
    {
    }

    bool createLayout( DataWidget* /*parent*/ )
    {
        return true;
    }

    bool createLayout( QLayout* /*layout*/)
    {
        return true;
    }

    bool createWidgets(DataWidget * /*parent*/, const data_type& /*d*/, bool /*readOnly*/)
    {
        return true;
    }
    void setReadOnly(bool /*readOnly*/)
    {
    }
    void readFromData(const data_type& /*d*/)
    {
    }
    void readConstantsFromData(const data_type& /*d*/)
    {
    }
    void writeToData(data_type& /*d*/)
    {
    }

    void insertWidgets()
    {
    }

};

template<class T, int I>
class default_struct_data_trait_var
{
public:
    typedef T data_type;
    static const char* name() { return nullptr; }
    static const char* shortname() { return nullptr; }
    static bool readOnly() { return false; }
    static bool isCheckable() { return false; }
    static bool isChecked(const data_type& /*d*/) { return true; }
    static void setChecked(bool /*b*/, data_type& /*d*/) {}
};

#define STRUCT_DATA_VAR(parent, vid, vname, sname, vtype, var)	\
      class struct_data_trait_var < parent, vid > : public default_struct_data_trait_var < parent, vid > \
      { \
      public: \
      typedef parent data_type; \
      typedef vtype value_type; \
      static const char* name() { return vname; } \
      static const char* shortname() { return sname; } \
      static const value_type* get(const data_type& d) { return &(d.var); } \
      static void set( const value_type& v, data_type& d) { d.var = v; } \
      }

#define STRUCT_DATA_VAR_READONLY(parent, vid, vname, sname, vtype, var) \
      class struct_data_trait_var < parent, vid > : public default_struct_data_trait_var < parent, vid > \
      { \
      public: \
      typedef parent data_type; \
      typedef vtype value_type; \
      static const char* name() { return vname; } \
      static const char* shortname() { return sname; } \
      static bool readOnly() { return true; }	\
      static const value_type* get(const data_type& d) { return &(d.var); } \
      static void set( const value_type& v, data_type& d) { d.var = v; } \
      }

#define STRUCT_DATA_VAR_CHECK(parent, vid, vname, sname, vtype, var, check) \
      class struct_data_trait_var < parent, vid > : public default_struct_data_trait_var < parent, vid > \
      { \
      public: \
      typedef parent data_type; \
      typedef vtype value_type; \
      static const char* name() { return vname; } \
      static const char* shortname() { return sname; } \
      static const value_type* get(const data_type& d) { return &(d.var); } \
      static void set( const value_type& v, data_type& d) { d.var = v; } \
      static bool isCheckable() { return true; } \
      static bool isChecked(const data_type& d) { return d.check; } \
      static void setChecked(bool b, data_type& d) { d.check = b; } \
      }

// A comma can't appear in a macro argument...
#define COMMA ,


////////////////////////////////////////////////////////////////
/// Rigids (as data-structures) support
////////////////////////////////////////////////////////////////

template<std::size_t N, class T>
class struct_data_trait < sofa::defaulttype::RigidCoord<N, T> >
{
public:
    typedef sofa::defaulttype::RigidCoord<N, T> data_type;
    enum { NVAR = 2 };
    static void set( data_type& /*d*/ )
    {
    }
};
template<class T> STRUCT_DATA_VAR(sofa::defaulttype::RigidCoord<3 COMMA T>, 0, "Center", "", typename data_type::Vec3, getCenter());
template<class T> STRUCT_DATA_VAR(sofa::defaulttype::RigidCoord<3 COMMA T>, 1, "Orientation", "", typename data_type::Quat, getOrientation());

template<class T> STRUCT_DATA_VAR(sofa::defaulttype::RigidCoord<2 COMMA T>, 0, "Center", "", typename data_type::Vec2, getCenter());
template<class T> STRUCT_DATA_VAR(sofa::defaulttype::RigidCoord<2 COMMA T>, 1, "Orientation", "A", typename data_type::Real, getOrientation());

template<std::size_t N, class T>
class data_widget_container < sofa::defaulttype::RigidCoord<N, T> > : public struct_data_widget_container < sofa::defaulttype::RigidCoord<N, T> >
{};

template<std::size_t N, class T>
class struct_data_trait < sofa::defaulttype::RigidMass<N, T> >
{
public:
    typedef sofa::defaulttype::RigidMass<N, T> data_type;
    enum { NVAR = 4 };
    static void set( data_type& d)
    {
        d.recalc();
    }
};

template<std::size_t N, class T> STRUCT_DATA_VAR(sofa::defaulttype::RigidMass<N COMMA T>, 0, "Mass", "Mass", T, mass);
template<std::size_t N, class T> STRUCT_DATA_VAR(sofa::defaulttype::RigidMass<N COMMA T>, 1, "Volume", "Vol", T, volume);
template<class T> STRUCT_DATA_VAR(sofa::defaulttype::RigidMass<2 COMMA T>, 2, "Inertia Matrix", "Inertia", T, inertiaMatrix);
template<class T> STRUCT_DATA_VAR(sofa::defaulttype::RigidMass<3 COMMA T>, 2, "Inertia Matrix", "Inertia", typename data_type::Mat3x3, inertiaMatrix);
template<class T> STRUCT_DATA_VAR_READONLY(sofa::defaulttype::RigidMass<2 COMMA T>, 3, "Inertia Mass Matrix", "InertialMass", T, inertiaMassMatrix);
template<class T> STRUCT_DATA_VAR_READONLY(sofa::defaulttype::RigidMass<3 COMMA T>, 3, "Inertia Mass Matrix", "InertialMass", typename data_type::Mat3x3, inertiaMassMatrix);

template<std::size_t N, class T>
class data_widget_container < sofa::defaulttype::RigidMass<N, T> > : public struct_data_widget_container < sofa::defaulttype::RigidMass<N, T> >
{};


////////////////////////////////////////////////////////////////
/// sofa::component::forcefield::LinearSpring support
////////////////////////////////////////////////////////////////

#define CLASS typename sofa::component::solidmechanics::spring::LinearSpring< T >

template<class T>
class struct_data_trait < CLASS >
{
public:
    typedef CLASS data_type;
    enum { NVAR = 5 };
    static void set( data_type& /*d*/)
    {
    }
};

template<class T> STRUCT_DATA_VAR(CLASS, 0, "Index 1", "Index 1", sofa::Index, m1);
template<class T> STRUCT_DATA_VAR(CLASS, 1, "Index 2", "Index 2", sofa::Index, m2);
template<class T> STRUCT_DATA_VAR(CLASS, 2, "Stiffness", "Ks", T, ks);
template<class T> STRUCT_DATA_VAR(CLASS, 3, "Damping", "Kd", T, kd);
template<class T> STRUCT_DATA_VAR(CLASS, 4, "Rest Length", "L", T, initpos);

template<class T>
class data_widget_container < CLASS > : public struct_data_widget_container < CLASS >
{};

#undef CLASS

////////////////////////////////////////////////////////////////
/// sofa::component::forcefield::JointSpring support
////////////////////////////////////////////////////////////////

#define CLASS typename sofa::component::solidmechanics::spring::JointSpring< T >

template<class T>
class struct_data_trait < CLASS >
{
public:
    typedef CLASS data_type;
    enum { NVAR = 27 };
    static void set( data_type& /*d*/)
    {
    }
};

template<class T> STRUCT_DATA_VAR(CLASS, 0,  "Index 1", "Index 1", sofa::Index, m1);
template<class T> STRUCT_DATA_VAR(CLASS, 1,  "Index 2", "Index 2", sofa::Index, m2);
template<class T> STRUCT_DATA_VAR(CLASS, 2, "Trans X Axis", "Trans X Axis", bool, freeMovements[0]);
template<class T> STRUCT_DATA_VAR(CLASS, 3, "Trans Y Axis", "Trans Y Axis", bool, freeMovements[1]);
template<class T> STRUCT_DATA_VAR(CLASS, 4, "Trans Z Axis", "Trans Z Axis", bool, freeMovements[2]);
template<class T> STRUCT_DATA_VAR(CLASS, 5, "Rot X Axis", "Rot X Axis", bool, freeMovements[3]);
template<class T> STRUCT_DATA_VAR(CLASS, 6, "Rot Y Axis", "Rot Y Axis", bool, freeMovements[4]);
template<class T> STRUCT_DATA_VAR(CLASS, 7, "Rot Z Axis", "Rot Z Axis", bool, freeMovements[5]);
template<class T> STRUCT_DATA_VAR(CLASS, 8,  "Soft Stiffness Translation", "Soft Ks Trans", typename data_type::Real, softStiffnessTrans);
template<class T> STRUCT_DATA_VAR(CLASS, 9,  "Hard Stiffness Translation", "Hard Ks Trans", typename data_type::Real, hardStiffnessTrans);
template<class T> STRUCT_DATA_VAR(CLASS, 10,  "Soft Stiffness Rotation", "Soft Ks Rot", typename data_type::Real, softStiffnessRot);
template<class T> STRUCT_DATA_VAR(CLASS, 11,  "Hard Stiffness Rotation", "Hard Ks Rot", typename data_type::Real, hardStiffnessRot);
template<class T> STRUCT_DATA_VAR(CLASS, 12,  "Bloc Stiffness Rotation", "Bloc Ks Rot", typename data_type::Real, blocStiffnessRot);
template<class T> STRUCT_DATA_VAR(CLASS, 13,  "Damping", "Kd", typename data_type::Real, kd);
template<class T> STRUCT_DATA_VAR(CLASS, 14,  "Min Angle X", "Min Angle X", typename data_type::Real, limitAngles[0]);
template<class T> STRUCT_DATA_VAR(CLASS, 15,  "Max Angle X", "Max Angle X", typename data_type::Real, limitAngles[1]);
template<class T> STRUCT_DATA_VAR(CLASS, 16,  "Min Angle Y", "Min Angle Y", typename data_type::Real, limitAngles[2]);
template<class T> STRUCT_DATA_VAR(CLASS, 17, "Max Angle Y", "Max Angle Y", typename data_type::Real, limitAngles[3]);
template<class T> STRUCT_DATA_VAR(CLASS, 18, "Min Angle Z", "Min Angle Z", typename data_type::Real, limitAngles[4]);
template<class T> STRUCT_DATA_VAR(CLASS, 19, "Max Angle Z", "Max Angle Z", typename data_type::Real, limitAngles[5]);

template<class T> STRUCT_DATA_VAR(CLASS, 20,  "Initial length of the spring X", "L init spring X", typename data_type::Real, initTrans[0]);
template<class T> STRUCT_DATA_VAR(CLASS, 21,  "Initial length of the spring Y", "L init spring Y", typename data_type::Real, initTrans[1]);
template<class T> STRUCT_DATA_VAR(CLASS, 22,  "Initial length of the spring Z", "L init spring Z", typename data_type::Real, initTrans[2]);

template<class T> STRUCT_DATA_VAR(CLASS, 23,  "Initial rotation of the spring X", "Rot init spring X", SReal, initRot[0]);
template<class T> STRUCT_DATA_VAR(CLASS, 24,  "Initial rotation of the spring Y", "Rot init spring Y", SReal, initRot[1]);
template<class T> STRUCT_DATA_VAR(CLASS, 25,  "Initial rotation of the spring Z", "Rot init spring Z", SReal, initRot[2]);
template<class T> STRUCT_DATA_VAR(CLASS, 26,  "Initial rotation of the spring W", "Rot init spring W", SReal, initRot[3]);

template<class T>
class data_widget_container < CLASS > : public struct_data_widget_container < CLASS >
{};

#undef CLASS

////////////////////////////////////////////////////////////////
/// sofa::component::forcefield::GearSpring support
////////////////////////////////////////////////////////////////

#define CLASS typename sofa::component::solidmechanics::spring::GearSpring< T >

template<class T>
class struct_data_trait < CLASS >
{
public:
    typedef CLASS data_type;
    enum { NVAR = 10 };
    static void set( data_type& /*d*/)
    {
    }
};

template<class T> STRUCT_DATA_VAR(CLASS, 0,  "Parent 1", "Parent 1", unsigned int, p1);
template<class T> STRUCT_DATA_VAR(CLASS, 1,  "Index 1", "Index 1", unsigned int, m1);
template<class T> STRUCT_DATA_VAR(CLASS, 2,  "Parent 2", "Parent 2", unsigned int, p2);
template<class T> STRUCT_DATA_VAR(CLASS, 3,  "Index 2", "Index 2", unsigned int, m2);
template<class T> STRUCT_DATA_VAR(CLASS, 4, "Axis 1", "Axis 1", unsigned int, freeAxis[0]);
template<class T> STRUCT_DATA_VAR(CLASS, 5, "Axis 2", "Axis 2", unsigned int, freeAxis[1]);
template<class T> STRUCT_DATA_VAR(CLASS, 6,  "Pivot Stiffness Translation", "Ks Trans", typename data_type::Real, hardStiffnessTrans);
template<class T> STRUCT_DATA_VAR(CLASS, 7,  "Gear Stiffness Rotation", "Gear Ks Rot", typename data_type::Real, softStiffnessRot);
template<class T> STRUCT_DATA_VAR(CLASS, 8,  "Pivot Stiffness Rotation", "Pivot Ks Rot", typename data_type::Real, hardStiffnessRot);
template<class T> STRUCT_DATA_VAR(CLASS, 9,  "Damping", "Kd", typename data_type::Real, kd);

template<class T>
class data_widget_container < CLASS > : public struct_data_widget_container < CLASS >
{};

#undef CLASS


////////////////////////////////////////////////////////////////
/// sofa::type::Material support
////////////////////////////////////////////////////////////////

template<>
class struct_data_trait < sofa::type::Material >
{
public:
    typedef sofa::type::Material data_type;
    enum { NVAR = 6 };
    static void set( data_type& /*d*/)
    {
    }
};

template<> STRUCT_DATA_VAR(sofa::type::Material, 0, "Name", "Name", std::string, name);
template<> STRUCT_DATA_VAR_CHECK(sofa::type::Material, 1, "Ambient", "Amb", sofa::type::RGBAColor, ambient, useAmbient);
template<> STRUCT_DATA_VAR_CHECK(sofa::type::Material, 2, "Diffuse", "Diff", sofa::type::RGBAColor, diffuse, useDiffuse);
template<> STRUCT_DATA_VAR_CHECK(sofa::type::Material, 3, "Specular", "Spec", sofa::type::RGBAColor, specular, useSpecular);
template<> STRUCT_DATA_VAR_CHECK(sofa::type::Material, 4, "Emissive", "Emm", sofa::type::RGBAColor, emissive, useEmissive);
template<> STRUCT_DATA_VAR_CHECK(sofa::type::Material, 5, "Shininess", "Shin", float, shininess, useShininess);

template<>
class data_widget_container < sofa::type::Material > : public struct_data_widget_container < sofa::type::Material >
{};

} //namespace sofa::gui::qt
