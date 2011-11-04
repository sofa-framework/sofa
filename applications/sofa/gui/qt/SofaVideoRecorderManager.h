/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_GUI_QT_VIDEORECORDERMANAGER_H
#define SOFA_GUI_QT_VIDEORECORDERMANAGER_H

#include "VideoRecorderManager.h"
#include "SofaGUIQt.h"

#include <vector>

#ifdef SOFA_QT4
#include <QComboBox>
#include <QSpinBox>
#include <QLabel>
#else
#include <qcombobox.h>
#include <qspinbox.h>
#include <qlabel.h>
#endif
namespace sofa
{
namespace gui
{
namespace qt
{

class MovieOptionsWidget : public QWidget
{
    Q_OBJECT
public:
    //Codec = <extension, description>
    typedef std::pair<std::string, std::string> Codec;


    MovieOptionsWidget( QWidget * parent = 0);

    QComboBox* codecComboBox;
    QSpinBox* bitrateSpinBox;
    QSpinBox* framerateSpinBox;

    std::vector< Codec > listCodecs;
};

class SofaVideoRecorderManager: public VideoRecorderManager
{
    Q_OBJECT
public:
    enum RecordingType { SCREENSHOTS, MOVIE };

    SofaVideoRecorderManager();

    static SofaVideoRecorderManager* getInstance()
    {
        static SofaVideoRecorderManager instance;
        return &instance;
    }

    void updateContent();
    std::string getCodecExtension();
    unsigned int getFramerate();
    unsigned int getBitrate();
    RecordingType getRecordingType() { return currentRecordingType; }

    //helper function
    static void internalAddWidget(QWidget* parent, QWidget* widgetToAdd);

public slots:
    virtual void onChangeRecordingType();
    virtual void close();

protected:
    RecordingType currentRecordingType;

    MovieOptionsWidget* movieOptionsWidget;
    QWidget* screenshotsOptionsWidget;
};

}
}
}

#endif //SOFA_GUI_QT_VIDEORECORDERMANAGER_H
