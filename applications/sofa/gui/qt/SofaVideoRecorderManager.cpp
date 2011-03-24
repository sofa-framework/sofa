/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#include "SofaVideoRecorderManager.h"

#include <iostream>
#ifndef SOFA_QT4
#include <qlineedit.h>
#include <qcombobox.h>
#include <qlabel.h>
#include <qgroupbox.h>
#include <qlayout.h>
#include <qradiobutton.h>
#endif

namespace sofa
{
namespace gui
{
namespace qt
{

MovieOptionsWidget::MovieOptionsWidget( QWidget * parent)
    : QWidget(parent)
{
    //Build codec list
    listCodecs.push_back(Codec("mpeg", "Mpeg1 (readable everywhere, not efficient at all)"));
    listCodecs.push_back(Codec("mp4", "Mpeg4 (Best ratio bitrate/visual quality)"));

    QVBoxLayout *layout=new QVBoxLayout(this);

    QHBoxLayout *HLayoutCodec = new QHBoxLayout();
    QLabel *labelCodec=new QLabel(QString("Codec: "), this);
    codecComboBox = new QComboBox(this);
    for(unsigned int i=0; i<listCodecs.size(); i++)
        codecComboBox->insertItem(QString(listCodecs[i].second.c_str()));
    HLayoutCodec->addWidget (labelCodec);
    HLayoutCodec->addWidget (codecComboBox);

    QHBoxLayout *HLayoutBitrate = new QHBoxLayout();
    QLabel *labelBitrate=new QLabel(QString("Bitrate (in KB/s): "), this);
    bitrateSpinBox = new QSpinBox(this);
    bitrateSpinBox->setMinValue(100);
    bitrateSpinBox->setMaxValue(40960);
    bitrateSpinBox->setValue(800);
    HLayoutBitrate->addWidget (labelBitrate);
    HLayoutBitrate->addWidget (bitrateSpinBox);

    QHBoxLayout *HLayoutFramerate = new QHBoxLayout();
    QLabel *labelFramerate=new QLabel(QString("Framerate (in img/s): "), this);
    framerateSpinBox = new QSpinBox(this);
    framerateSpinBox->setMinValue(1);
    framerateSpinBox->setMaxValue(120);
    framerateSpinBox->setValue(60);
    HLayoutFramerate->addWidget (labelFramerate);
    HLayoutFramerate->addWidget (framerateSpinBox);

    layout->addLayout(HLayoutCodec);
    layout->addLayout(HLayoutBitrate);
    layout->addLayout(HLayoutFramerate);

    //this->addLayout(layout);
}

SofaVideoRecorderManager::SofaVideoRecorderManager()
{
    //create option widgets
    //movie option widget
    //movieOptionsWidget = new QWidget(this);

#ifndef SOFA_HAVE_FFMPEG
    MovieRecordingTypeRadioButton->setHidden(true);
#endif

    movieOptionsWidget = new MovieOptionsWidget(this);

    //movieOptionsWidget->setVisible(currentRecordingType == MOVIE);
    movieOptionsWidget->setHidden(!currentRecordingType == MOVIE);

    internalAddWidget(VideoRecorderOptionGroupBox, movieOptionsWidget);
}


std::string SofaVideoRecorderManager::getCodecExtension()
{
    if(movieOptionsWidget)
    {
        unsigned int index = movieOptionsWidget->codecComboBox->currentItem();
        return movieOptionsWidget->listCodecs[index].first;
    }
    return std::string();
}

unsigned int SofaVideoRecorderManager::getFramerate()
{
    if(movieOptionsWidget)
    {
        return movieOptionsWidget->framerateSpinBox->value();
    }

    return 0;
}

unsigned int SofaVideoRecorderManager::getBitrate()
{
    if(movieOptionsWidget)
    {
        return movieOptionsWidget->bitrateSpinBox->value()*1024;
    }

    return 0;
}

void SofaVideoRecorderManager::updateContent()
{
    //movieOptionsWidget->setVisible(currentRecordingType == MOVIE);
    movieOptionsWidget->setHidden(!currentRecordingType == MOVIE);
}

void SofaVideoRecorderManager::onChangeRecordingType()
{
    currentRecordingType = (MovieRecordingTypeRadioButton->isChecked()) ? MOVIE : SCREENSHOTS;

    updateContent();
}

void SofaVideoRecorderManager::internalAddWidget(QWidget* parent, QWidget* widgetToAdd)
{

#ifdef SOFA_QT4
    parent->layout()->addWidget(widgetToAdd);
#else
    parent->layout()->add(widgetToAdd);
#endif
}

void SofaVideoRecorderManager::close()
{
    this->hide();
}

} //namespace qt

} //namespace gui

}//namespace sofa
