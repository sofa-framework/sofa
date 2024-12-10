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
#include <sofa/gui/qt/config.h>
#include "ModifyObject.h"
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/system/FileRepository.h>

#include <QDialog>
#include <QLineEdit>
#include <QTableWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QRadioButton>
#include <QButtonGroup>


//If a table has higher than MAX_NUM_ELEM, its data won't be loaded at the creation of the window
//user has to click on the button update to see the content
#define MAX_NUM_ELEM 100


namespace sofa::gui::qt
{

/**
*\brief Abstract Interface of a qwidget which allows to edit a data.
*/
class SOFA_GUI_QT_API DataWidget : public QWidget
{
    Q_OBJECT
public:
    //
    // Factory related code
    //

    struct CreatorArgument
    {
        std::string name;
        core::objectmodel::BaseData* data;
        QWidget* parent;
        bool readOnly;
    };

    template<class T>
    static T* create(T*, const CreatorArgument& arg)
    {
        typename T::MyData* data = dynamic_cast<typename T::MyData*>(arg.data);
        if(!data) return nullptr;
        T* instance = new T(arg.parent, arg.name.c_str(), data);
        if ( !instance->createWidgets() )
        {
            delete instance;
            instance = nullptr;
        }
        if (instance)
        {
            instance->setDataReadOnly(arg.readOnly);
        }
        return instance;
    }
   
    static DataWidget *CreateDataWidget(const DataWidget::CreatorArgument &dwarg);    


public slots:
    /// Checks that widget has been edited
    /// emit DataOwnerDirty in case the name field has been modified
    void updateDataValue();

    /// First checks that the widget is not currently being edited
    /// checks that the data has changed since the last time the widget
    /// has read the data value.
    /// ultimately read the data value.
    void updateWidgetValue();

    /// You call this slot anytime you want to specify that the widget
    /// value is out of sync with the underlying data value.
    void setWidgetDirty(bool b=true);

    /// slot to be called if DataWidget has not been filled at constructor and need to be filled 
    /// at first call. Will turn toFill to true only if isFilled == false
    void fillFromData();

signals:
    /// Emitted each time setWidgetDirty is called. You can also emit
    /// it if you want to tell the widget value is out of sync with
    /// the underlying data value.
    void WidgetDirty(bool );
    /// Currently this signal is used to reflect the changes of the
    /// component name in the sofaListview.
    void DataOwnerDirty(bool );

    void dataValueChanged(QString dataValueString );
public:
    typedef core::objectmodel::BaseData MyData;

    DataWidget(QWidget* parent,const char* name, MyData* d);

    ~DataWidget() override;

    virtual void setData( MyData* d);

    /// BaseData pointer accessor function.
    inline const core::objectmodel::BaseData* getBaseData() const { return baseData; }
    inline core::objectmodel::BaseData* getBaseData() { return baseData; }

    void updateVisibility();

    inline bool isDirty() { return dirty; }

    /// return if DataWidget as been filled
    bool isFilled() { return m_isFilled; }
    /// method to warn if Data has not been filled at constructor. 
    void setFilled(bool value) { m_isFilled = value; }

    /// The implementation of this method holds the widget creation and the signal / slot
    /// connections.
    virtual bool createWidgets() = 0;
    /// This method is called after createWidgets to configure whether the created widgets should be read-only
    virtual void setDataReadOnly(bool readOnly) = 0;
    /// Helper method to give a size.
    virtual unsigned int sizeWidget() {return 1;}
    /// Helper method for column.
    virtual unsigned int numColumnWidget() {return 3;}

protected:
    /// The implementation of this method tells how the widget reads the value of the data.
    virtual void readFromData() = 0;
    /// The implementation of this methods needs to tell how the widget can write its value
    /// in the data
    virtual void writeToData() = 0;

    core::objectmodel::BaseData* baseData;
    bool dirty;
    int counter;
    bool m_isFilled; ///< tell if DataWidget has been filled from Data true by default
    bool m_toFill; ///< bool to warn action is needed to fill Data, false by default
};



/**
*\brief This class is basically the same as DataWidget, except that it
* takes a template parameter so the actual type of Data can be retrieved
* through the getData() accessor. In most cases you will need to derive
* from this class to implement the edition of your data in the GUI.
**/
template<class T>
class TDataWidget : public DataWidget
{

public:
    typedef sofa::core::objectmodel::Data<T> MyTData;

    template <class RealObject>
    static RealObject* create( RealObject*, CreatorArgument& arg)
    {
        RealObject* obj = nullptr;

        typename RealObject::MyTData* realData = dynamic_cast< typename RealObject::MyTData* >(arg.data);
        if (!realData)
        {
            if constexpr (std::is_constructible_v<RealObject, QWidget*, const char*, core::objectmodel::BaseData*, const T*>)
            {
                if (arg.data)
                {
                    const void* rawPtr = arg.data->getValueVoidPtr();

                    //note that this cast is not type-safe. You need to check
                    //later in createWidget that the baseData is the expected
                    //type. You can use getValueTypeString for example
                    if (const T* castedPtr = static_cast<const T*>(rawPtr))
                    {
                        obj = new RealObject(arg.parent, arg.name.c_str(), arg.data, castedPtr);
                    }
                }
            }
        }
        else
        {
            if constexpr (std::is_constructible_v<RealObject, QWidget*, const char*, MyTData*>)
            {
                obj = new RealObject(arg.parent, arg.name.c_str(), realData);
            }
        }

        if (obj)
        {
            if( !obj->createWidgets() )
            {
                delete obj;
                obj = nullptr;
            }
            if (obj)
            {
                obj->setDataReadOnly(arg.readOnly);
            }
        }
        return obj;
    }

    TDataWidget(QWidget* parent,const char* name, core::objectmodel::BaseData* d, const T* object):
        DataWidget(parent, name, d),
        Tdata(nullptr)
    {
        SOFA_UNUSED(object);
    }

    TDataWidget(QWidget* parent,const char* name, MyTData* d):
        DataWidget(parent,name,d),Tdata(d) {}

    /// Accessor function. Gives you the actual data instead
    /// of a BaseData pointer of it like in getBaseData().
    sofa::core::objectmodel::Data<T>* getData() {return Tdata;}
    const sofa::core::objectmodel::Data<T>* getData() const {return Tdata;}

    using DataWidget::setData;
    inline virtual void setData(MyTData* d)
    {
        Tdata = d;
    }
protected:
    MyTData* Tdata { nullptr };
};





class SOFA_GUI_QT_API QPushButtonUpdater: public QPushButton
{
    Q_OBJECT
public:

    QPushButtonUpdater( const QString & text, QWidget * parent = nullptr ): QPushButton(text,parent) {};

public Q_SLOTS:
    void setDisplayed(bool b);
};

//Widget used to display the name of a Data and if needed the link to another Data
class SOFA_GUI_QT_API QDisplayDataInfoWidget: public QWidget
{
    Q_OBJECT
public:
    QDisplayDataInfoWidget(QWidget* parent, const std::string& helper, core::objectmodel::BaseData* d, bool modifiable, const ModifyObjectFlags& modifyObjectFlags);
public Q_SLOTS:
    void linkModification();
    void linkEdited();
    unsigned int getNumLines() const { return numLines_;}

signals:
	void WidgetDirty();

protected:
	static QIcon& LinkIcon()
	{
		static QIcon icon;
		if(icon.isNull())
		{
			std::string filename = "textures/link.png";
			sofa::helper::system::DataRepository.findFile(filename);
			icon = QIcon(filename.c_str());
		}
		return icon;
	}

protected:
    void formatHelperString(const std::string& helper, std::string& final_text);
    static unsigned int numLines(const std::string& str);
    core::objectmodel::BaseData* data;
    unsigned int numLines_;
    QLineEdit *linkpath_edit;
};

typedef sofa::helper::Factory<std::string, DataWidget, DataWidget::CreatorArgument> DataWidgetFactory;

} //namespace sofa::gui::qt

//MOC_SKIP_BEGIN
#if !defined(SOFA_BUILD_SOFA_GUI_QT)
namespace sofa::helper
{
//delay load of the specialized Factory class. unique definition reside in the cpp file
extern template class SOFA_GUI_QT_API Factory<std::string, gui::qt::DataWidget, gui::qt::DataWidget::CreatorArgument>;
} // namespace sofa::helper

#endif
//MOC_SKIP_END
