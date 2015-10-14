#ifndef SOFA_IMAGE_VECTORVISWIDGET_H
#define SOFA_IMAGE_VECTORVISWIDGET_H

#include <sofa/gui/qt/SimpleDataWidget.h>
#include <sofa/gui/qt/DataWidget.h>
#include <image/image_gui/config.h>
#include "../VectorVis.h"

#include <QCheckBox>

#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Data.h>


namespace sofa
{
namespace gui
{
namespace qt
{


/**
* Abstract, non templated widget that allows us to use and declare signals, while subclass contains template information
*/
class VectorVisSettings: public QObject
{
    Q_OBJECT

public:
    virtual ~VectorVisSettings() {}

    /**
    * @name From Options
    *	Update the settings data when the widget has been manipulated
    */
    /**@{*/
    virtual void shapeFromOptions(bool vis)=0;
    virtual void rgbFromOptions(bool rgb)=0;
    virtual void subsampleXYFromOptions(int subsample)=0;
    virtual void subsampleZFromOptions(int subsample)=0;
    virtual void shapeScaleFromOptions(int scale)=0;
    /**@}*/

    QWidget* getWidget() { return widget;}

signals:
    /**
    * @name Updates
    *	 When the settings data has changed, these functions update the widget to reflect the new values
    */
    /**@{*/
    void settingsModified();
    void updateRgb(bool);
    void updateSubsampleXY(int);
    void updateSubsampleZ(int);
    void updateShapeScale(int);
    void updateShape(bool);
    /**@}*/

protected:
    QWidget* widget;

};

/**
* The subclass is templated, allowing access to the DataType. Currently, VectorVis is the only possible DataType
*/
template <class DataType>
class TVectorVisSettings: public VectorVisSettings
{
protected:
    const DataType* vectorData;

    /**
    * One shape is drawn every subsampleXY values in both the X plane and the Y plane. So, as subsampleXY is increased, the density of the shapes decreases.
    */
    int subsampleXY;

    /**
    * One shape is drawn every subsampleZ values in Z plane. So, as subsampleZ is increased, the density of the shapes decreases.
    */
    int subsampleZ;

    /**
    * The size of the shape is multiplied by this value before it is drawn.
    */
    int shapeScale;

    /**
    * When true, a 3 channel image is displayed as an RGB image. When false, the image is displayed in greyscale, with the value being the norm of the 3 channels.
    */
    bool rgb;

    /**
    * When true, a shape is drawn representing the data. In a 3 channel image, that shape is an arrow, and in a 6 channel image, the shape is an ellipsoid.
    */
    bool shape;


public:

    TVectorVisSettings(QWidget* parent) :vectorData(NULL), subsampleXY(5), subsampleZ(5), shapeScale(10), rgb(false), shape(false)
    {
        //Widget stuff
        QHBoxLayout *layout = new QHBoxLayout(parent);
        widget = new QWidget(parent);
        widget->setLayout(layout);
    }

    virtual ~TVectorVisSettings() {};

    /**
    * When the Data<VectorVis> in the ImageViewer is changed, this gets called.
    */
    void readFromData(const DataType& d0)
    {
        this->vectorData = &d0;

        if(!this->vectorData)
            return;

        this->shape = vectorData->getShape();
        this->rgb = vectorData->getRgb();
        this->subsampleXY = vectorData->getSubsampleXY();
        this->subsampleZ = vectorData->getSubsampleZ();
        this->shapeScale = vectorData->getShapeScale();
        updateGUI();
    }

    /**
    * This changes the data in Data<VectorVis> in the ImageViewer
    */
    void writeToData(DataType& d)
    {
        d.setShape(shape);
        d.setRgb(rgb);
        d.setSubsampleXY(subsampleXY);
        d.setSubsampleZ(subsampleZ);
        d.setShapeScale(shapeScale);
    }

    /**
    * Sends a signal and the appropriate information to the widget so that it can reflect the current settings.
    */
    void updateGUI()
    {
        emit updateRgb(rgb);
        emit updateShape(shape);
        emit updateSubsampleXY(subsampleXY);
        emit updateSubsampleZ(subsampleZ);
        emit updateShapeScale(shapeScale);
    }

    void shapeFromOptions(bool vis)
    {
        this->shape = vis;
        emit settingsModified();

    }

    void rgbFromOptions(bool _rgb)
    {
        this->rgb = _rgb;
        emit settingsModified();
    }

    void subsampleXYFromOptions(int subsample)
    {
        this->subsampleXY = subsample;
        emit settingsModified();
    }

    void subsampleZFromOptions(int subsample)
    {
        this->subsampleZ = subsample;
        emit settingsModified();
    }

    void shapeScaleFromOptions(int scale)
    {
        this->shapeScale = scale;
        emit settingsModified();
    }


};

/**
* Holds a QCheckbox and associated QLabel, arranging them horizontally.
*/
class VectorVisualizationCheckboxWidget : public QWidget
{
    Q_OBJECT
protected:
    QLabel* label;
    QHBoxLayout* layout;

public:
    QCheckBox* checkbox;

    /**
    * @param parent the parent QWidget
    * @param name the name of the checkbox. Will be the text of the QLabel.
    */
    VectorVisualizationCheckboxWidget(QWidget* parent, const QString name) : QWidget(parent)
    {
        layout = new QHBoxLayout(this);
        checkbox = new QCheckBox(this);
        label = new QLabel(this);
        label->setText(name);
        layout->addWidget(checkbox);
        layout->addWidget(label);
    }
};

/**
* Holds a Horizontal QSlider, with a QLabel beside it to display the current value
*/
class VectorVisualizationSliderWidget : public QWidget
{
    Q_OBJECT
protected:
    QLabel* numLabel;
    QHBoxLayout* layout;
public:
    QSlider* slider;

    /**
    * @param parent the parent QWidget
    * @param min the minimum value that can be selected by the slider
    * @ param max the maximum value that can be selected by the slider
    */
    VectorVisualizationSliderWidget(QWidget* parent, int min, int max) : QWidget(parent)
    {
        layout = new QHBoxLayout(this);
        slider = new QSlider(Qt::Horizontal, this);
        slider->setRange(min, max);
        slider->setValue( (int) min + (max-min)/2);
        numLabel = new QLabel(this);
        numLabel->setNum(slider->value());
        layout->addWidget(slider);
        layout->addWidget(numLabel);

        connect(slider, SIGNAL( valueChanged(int) ), this, SLOT( updateNumLabel(int) ) );

    }

    /**
    * Set the value and position of the slider to a specified value.
    */
    void setValue(int value)
    {
        slider->setValue(value);
        numLabel->setNum(value);
    }

public slots:
    /**
    * Slot the keep the value displayed by the label in sync with the value selected by the slider
    */
    void updateNumLabel(int newValue)
    {
        numLabel->setNum(newValue);
    }

};


/**
* Widget containing the controls for viewing vectorized images
*/
class VectorVisOptionsWidget: public QWidget
{
    Q_OBJECT
protected:
    /**
    * When checked, vector data is visualized as shape
    */
    VectorVisualizationCheckboxWidget* shapeCheckbox;
    /**
    * When checked, vector data is visualized as an RGB image
    */
    VectorVisualizationCheckboxWidget* rgbCheckbox;
    /**
    * Selects how many voxels are skipped between each shape in the X and Y planes
    */
    VectorVisualizationSliderWidget* subsampleXYSlider;
    /**
    * Selects how many voxels are skipped between each shape in the Z plane
    */
    VectorVisualizationSliderWidget* subsampleZSlider;
    /**
    * Selects how large the shape will be drawn
    */
    VectorVisualizationSliderWidget* shapeScaleSlider;
    /**
    * Collects and updates the information based on the GUI settings and the Data in the ImageViewer
    */
    VectorVisSettings* settings;

public:
    VectorVisOptionsWidget(VectorVisSettings* _settings, QWidget* parent)
        :QWidget(parent), settings(_settings)
    {
        shapeCheckbox = new VectorVisualizationCheckboxWidget(this, "Visualize Vectors as Shapes");
        rgbCheckbox = new VectorVisualizationCheckboxWidget(this, "Visualize Vectors as RGB Values");

        subsampleXYSlider = new VectorVisualizationSliderWidget(this, 1, 25);
        subsampleZSlider = new VectorVisualizationSliderWidget(this, 1, 25);
        shapeScaleSlider = new VectorVisualizationSliderWidget(this, 1, 100);

        QVBoxLayout * layout = new QVBoxLayout(this);
        layout->addWidget(rgbCheckbox);
        layout->addWidget(shapeCheckbox);
        layout->addWidget(subsampleXYSlider);

        QLabel* xyLabel = new QLabel(this);
        xyLabel->setText("XY Subsampling");
        layout->addWidget(xyLabel);

        layout->addWidget(subsampleZSlider);

        QLabel* zLabel = new QLabel(this);
        zLabel->setText("Z Subsampling");
        layout->addWidget(zLabel);

        layout->addWidget(shapeScaleSlider);

        QLabel* scaleLabel = new QLabel(this);
        scaleLabel->setText("Vector Scale");
        layout->addWidget(scaleLabel);

        //rgbCheckbox->checkbox->setEnabled(false);

        connect(rgbCheckbox->checkbox, SIGNAL(toggled(bool)), this, SLOT(changeRgb(bool)));
        connect(shapeCheckbox->checkbox, SIGNAL(toggled(bool)), this, SLOT(changeShape(bool)));
        connect(subsampleXYSlider->slider, SIGNAL(valueChanged(int)), this, SLOT(changeSubsampleXY(int)));
        connect(subsampleZSlider->slider, SIGNAL(valueChanged(int)), this, SLOT(changeSubsampleZ(int)));
        connect(shapeScaleSlider->slider, SIGNAL(valueChanged(int)), this, SLOT(changeShapeScale(int)));

        connect(settings, SIGNAL(updateRgb(bool)), this, SLOT(updateRgb(bool)));
        connect(settings, SIGNAL(updateShape(bool)), this, SLOT(updateShape(bool)));
        connect(settings, SIGNAL(updateSubsampleXY(int)), this, SLOT(updateSubsampleXY(int)));
        connect(settings, SIGNAL(updateSubsampleZ(int)), this, SLOT(updateSubsampleZ(int)));
        connect(settings, SIGNAL(updateShapeScale(int)), this, SLOT(updateShapeScale(int)));
    }

public slots:
    /**
    *@name Changes
    * When the visualization options are changed in the GUI, the settings are updated
    */
    /**@{*/
    void changeShape(bool shape)
    {
        settings->shapeFromOptions(shape);
        subsampleXYSlider->setEnabled(shape);
        subsampleZSlider->setEnabled(shape);
        shapeScaleSlider->setEnabled(shape);
    }

    void changeRgb(bool rgb)
    {
        settings->rgbFromOptions(rgb);
    }

    void changeSubsampleXY(int value)
    {
        settings->subsampleXYFromOptions(value);
    }

    void changeSubsampleZ(int value)
    {
        settings->subsampleZFromOptions(value);
    }

    void changeShapeScale(int value)
    {
        settings->shapeScaleFromOptions(value);
    }
    /**@}*/

    /**
    *@name Updates
    * When the settings are changed in the ImageViewer (for example, loading the .scn file settings),
    *  the GUI is updated
    */
    /**@{*/
    void updateRgb(bool rgb)
    {
        rgbCheckbox->checkbox->setChecked(rgb);
    }

    void updateSubsampleXY(int subsampleXY)
    {
        subsampleXYSlider->setValue(subsampleXY);
    }

    void updateSubsampleZ(int subsampleZ)
    {
        subsampleZSlider->setValue(subsampleZ);
    }

    void updateShapeScale(int scale)
    {
        shapeScaleSlider->setValue(scale);
    }

    void updateShape(bool shape)
    {
        shapeCheckbox->checkbox->setChecked(shape);
    }
    /**@}*/
};


/**
* Hold the VectorVisOptionsWidget along with template data
*/
template<class T>
class vectorvis_data_widget_container
{
public:
    TVectorVisSettings<T>* settings;
    VectorVisOptionsWidget* options;
    QVBoxLayout* container_layout;

    vectorvis_data_widget_container() : settings(NULL), container_layout(NULL) {}

    bool createLayout(DataWidget* parent)
    {
        if(parent->layout() != NULL || container_layout != NULL) return false;
        container_layout = new QVBoxLayout(parent);
        return true;
    }

    bool createLayout(QLayout* layout)
    {
        if(container_layout != NULL) return false;
        container_layout = new QVBoxLayout();
        layout->addItem(container_layout);
        return true;
    }

    bool createWidgets(DataWidget* parent, const T& d, bool /*readOnly*/)
    {
        settings = new TVectorVisSettings<T>(parent);
        settings->readFromData(d);

        options = new VectorVisOptionsWidget(settings, parent);

        return true;
    }

    void setReadOnly(bool /*readOnly*/) {}
    void readFromData(const T& d0) { settings->readFromData(d0); }
    void writeToData(T& d) {settings->writeToData(d); }

    void insertWidgets()
    {
        assert(container_layout);
        if(settings)
            container_layout->addWidget(settings->getWidget());
        if(options)
            container_layout->addWidget(options);
    }
};

/**
* Data Widget that allows for communication between the GUI and the corresponding Data
*/
template<class T>
class SOFA_IMAGE_GUI_API VectorVisualizationDataWidget : public SimpleDataWidget<T, vectorvis_data_widget_container< T > >
{

public:
    typedef SimpleDataWidget<T, vectorvis_data_widget_container< T > > Inherit;
    typedef sofa::core::objectmodel::Data<T> MyData;

    VectorVisualizationDataWidget(QWidget* parent, const char* name, MyData* d) : Inherit(parent, name, d) {}

    virtual bool createWidgets()
    {
        bool result = Inherit::createWidgets();
        VectorVisSettings* s = dynamic_cast<VectorVisSettings*>(this->container.settings);
        this->connect(s, SIGNAL(settingsModified()), this, SLOT(setWidgetDirty()));
        return result;
    }
};
}
}
}

#endif //SOFA_IMAGE_VECTORVISWIDGET_H
