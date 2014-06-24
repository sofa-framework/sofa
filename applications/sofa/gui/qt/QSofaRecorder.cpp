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
#include "QSofaRecorder.h"
#include "RealGUI.h"
#include <time.h>

#include <sofa/core/objectmodel/Tag.h>

#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/simulation/common/WriteStateVisitor.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>

#include <SofaLoader/ReadState.h>
#include <SofaExporter/WriteState.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>

#include "RealGUI.h"
#ifdef SOFA_QT4
#include <QToolTip>
#include <QInputDialog>
#include <QHBoxLayout>
#else
#include <qtooltip.h>
#include <qinputdialog.h>
#include <qlayout.h>
#endif

using namespace sofa::simulation;
using namespace sofa::component::misc;
using namespace sofa::helper::system;
using namespace sofa::core::objectmodel;
namespace sofa
{
namespace gui
{
namespace qt
{

QSofaRecorder::QSofaRecorder(QWidget* parent):QWidget(parent)
{
    QHBoxLayout* layout = new QHBoxLayout(this,0,0);
    timerStep = new QTimer(parent);

    fpsLabel = new QLabel ( "9999.9 FPS", this );
    fpsLabel->setMinimumSize ( fpsLabel->sizeHint() );
    fpsLabel->clear();
    layout->addWidget(fpsLabel);

    timeLabel = new QLabel ( "Time: 999.9999 s", this );
    timeLabel->setMinimumSize ( timeLabel->sizeHint() );
    timeLabel->clear();
    layout->addWidget(timeLabel);

    initialTime = new QLabel( "Init:", this);
    initialTime->setMinimumSize( initialTime->sizeHint() );
    layout->addWidget(initialTime);
    record                 = new QPushButton( this, "Record");
    record->setToggleButton(true);
    record->setOn(false);
    layout->addWidget(record);
    backward               = new QPushButton( this, "Backward");
    layout->addWidget(backward);
    stepbackward           = new QPushButton( this, "Step Backward");
    layout->addWidget(stepbackward);
    playforward            = new QPushButton( this, "Play Forward");
    playforward->setToggleButton(true);
    layout->addWidget(playforward);
    stepforward            = new QPushButton( this, "Step Forward");
    layout->addWidget(stepforward);
    forward                = new QPushButton( this, "Forward");
    layout->addWidget(forward);

    timeRecord = new QLabel("T=",this);
    layout->addWidget(timeRecord);
    loadRecordTime = new QLineEdit(this);
    loadRecordTime->setMaximumSize(QSize(75, 100));
    layout->addWidget(loadRecordTime);
    timeSlider = new QSlider( Qt::Horizontal, this, "Time Slider");
    timeSlider->setTickmarks(QSlider::Both);
    timeSlider->setMinValue(0);
    timeSlider->setMaxValue(0);
    layout->addWidget(timeSlider);

    finalTime = new QLabel( "End:", this );
    finalTime->setMinimumSize ( finalTime->sizeHint() );
    layout->addWidget(finalTime);

    QToolTip::add(record               , tr( "Record" ) );
    QToolTip::add(backward      , tr( "Load Initial Time" ) );
    QToolTip::add(stepbackward  , tr( "Make one step backward" ) );
    QToolTip::add(playforward   , tr( "Continuous play forward" ) );
    QToolTip::add(stepforward   , tr( "Make one step forward" ) );
    QToolTip::add(forward       , tr( "Load Final Time" ) );


    RealGUI::SetPixmap("textures/media-record.png", record);
    RealGUI::SetPixmap("textures/media-seek-backward.png", backward);
    RealGUI::SetPixmap("textures/media-skip-backward.png", stepbackward);
    RealGUI::SetPixmap("textures/media-playback-start.png", playforward);
    RealGUI::SetPixmap("textures/media-skip-forward.png", stepforward);
    RealGUI::SetPixmap("textures/media-seek-forward.png", forward);

    record_directory = sofa::helper::system::SetDirectory::GetRelativeFromDir("../../examples/Simulation/",sofa::helper::system::SetDirectory::GetProcessFullPath("").c_str());


    connect ( record,         SIGNAL (toggled (bool) ),  this, SLOT(  slot_recordSimulation( bool) ) );
    connect ( backward,       SIGNAL (clicked () ),      this, SLOT(  slot_backward( ) ) );
    connect ( stepbackward,   SIGNAL (clicked () ),      this, SLOT(  slot_stepbackward( ) ) );
    connect ( playforward,    SIGNAL (clicked () ),      this, SLOT(  slot_playforward( ) ) );
    connect ( stepforward,    SIGNAL (clicked () ),      this, SLOT(  slot_stepforward( ) ) );
    connect ( forward,        SIGNAL (clicked () ),      this, SLOT(  slot_forward( ) ) );
    connect ( loadRecordTime, SIGNAL (returnPressed ()), this, SLOT(  slot_loadrecord_timevalue()));
    connect ( timeSlider, SIGNAL (sliderMoved (int) ),   this, SLOT( slot_sliderValue( int) ) );
    connect ( timeSlider, SIGNAL (valueChanged ( int ) ), this, SLOT( slot_sliderValue(int) ) );
    connect ( timerStep, SIGNAL( timeout() ), this, SLOT(slot_stepforward() ));

    this->setMaximumHeight(timeRecord->height());
    root = NULL;

}

void QSofaRecorder::SetRecordDirectory(const std::string& record_dir)
{
    record_directory = record_dir;
}

void QSofaRecorder::TimerStart(bool value)
{
    if(value)
    {
        timerStep->start(0);
    }
    else
    {
        timerStep->stop();
    }
}


void QSofaRecorder::SetSimulation(simulation::Node* root, const std::string& initT,
        const std::string& endT, const std::string& writeName)
{
    char buf[100];
    this->root = root;
    assert(root);
    double dt = root->getDt();

    sprintf ( buf, "Init: %s s",initT.c_str()  );
    initialTime->setText ( buf );

    sprintf ( buf, "End: %s s",endT.c_str()  );
    finalTime->setText ( buf );

    loadRecordTime->setText( QString(initT.c_str()) );

    timeSlider->setMaxValue( (int)((atof(endT.c_str())-atof(initT.c_str()))/(dt)+0.5));

    addReadState(writeName,true);

}

void QSofaRecorder::UpdateTime(simulation::Node* root)
{
    this->root = root;
    assert(root);
    double time = root->getTime();
    char buf[100];
    sprintf ( buf, "Time: %.3g s", time );
    timeLabel->setText ( buf );
    if (record->isOn())
    {
        setCurrentTime(time);
        double final_time = getFinalTime();

        if ((int)(1000*final_time) < (int)(1000*time))
        {
            setFinalTime(time);
            timeSlider->setMaxValue(timeSlider->maxValue()+1);
            timeSlider->setValue(timeSlider->maxValue());
        }
        else
        {
            timeSlider->setValue(timeSlider->value()+1);
        }
        timeSlider->update();
    }
}
void QSofaRecorder::Clear(simulation::Node* root)
{
    this->root = root;
    assert(root);
    float initial_time = root->getTime();
    timeSlider->setValue(0);
    timeSlider->setMinValue(0);
    timeSlider->setMaxValue(0);
    setInitialTime(initial_time);
    setFinalTime(initial_time);
    setCurrentTime(initial_time);
    timeSlider->update();
}

void QSofaRecorder::setFPS(double fps)
{
    char buf[100];
    sprintf ( buf, "%.1f FPS", fps );
    fpsLabel->setText ( buf );
}

void QSofaRecorder::setInitialTime(double time)
{
    char buf[100];
    sprintf ( buf, "Init: %g s", fabs(time) );
    initialTime->setText ( buf );

}

void QSofaRecorder::setFinalTime(double time)
{
    char buf[100];
    sprintf ( buf, "End: %g s", fabs(time) );
    finalTime->setText( buf );
}
void QSofaRecorder::setCurrentTime(double time)
{
    char buf[100];
    sprintf ( buf, "%g", fabs(time) );
    loadRecordTime->setText( buf );
    setTimeSimulation(time);
}

void QSofaRecorder::setTimeSimulation(double time)
{
    char buf[100];
    sprintf ( buf, "Time: %g s", time );
    timeLabel->setText ( buf );
    if (root)
    {
        root->setTime(time);
    }
}

void QSofaRecorder::slot_recordSimulation(bool value)
{

    assert(root);
    if (value)
    {
        if (querySimulationName())
        {
            Clear(root);
            //Add if needed WriteState
            addWriteState(writeSceneName_);
            addReadState(writeSceneName_,false); //no init done
            emit(RecordSimulation(true));

        }
        else
        {
            record->setOn ( false );
            return;
        }

    }
    else
    {
        //Halt the simulation.
        emit RecordSimulation(false);
        //Save simulation file
        std::string FileName(((RealGUI*)(qApp->mainWidget()))->windowFilePath().ascii());
        std::string simulationFileName = simulationBaseName_ + ".simu";
        std::ofstream out(simulationFileName.c_str());

        if (!out.fail())
        {
            out << sofa::helper::system::DataRepository.getFile ( FileName ) << " " << initialTime->text().ascii() << " " << finalTime->text().ascii() << " " << root->getDt() << " baseName: "<<writeSceneName_;
            out.close();
        }
        std::cout << "Simulation parameters saved in "<<simulationFileName<<std::endl;
    }
    //Change the state of the writers
    WriteStateActivator v_write(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, value);
    v_write.addTag(Tag("AutoRecord"));
    v_write.execute(root);
    ReadStateActivator v_read(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, false);
    v_read.addTag(Tag("AutoRecord"));
    v_read.execute(root);
}
void QSofaRecorder::slot_backward()
{
    if (timeSlider->value() != timeSlider->minValue())
    {
        setCurrentTime(getInitialTime());
        slot_sliderValue(timeSlider->minValue());
        loadSimulation();
    }
}
void QSofaRecorder::slot_stepbackward()
{
    assert(root);
    playforward->setOn(false);
    double init_time  = getInitialTime();
    double time = getCurrentTime() - root->getDt();
    if (time < init_time) time = init_time;

    setCurrentTime(time);
    slot_loadrecord_timevalue(false);
}
void QSofaRecorder::slot_playforward()
{
    if (playforward->isOn() )
    {
        if (timeSlider->value() == timeSlider->maxValue())
        {
            playforward->setOn(false);
        }
        else
        {

            RealGUI::SetPixmap("textures/media-playback-pause.png", playforward);

            timerStep->start ( 0 );
        }
    }
    else
    {
        timerStep->stop();
        RealGUI::SetPixmap("textures/media-playback-start.png", playforward);
    }
}
void QSofaRecorder::slot_stepforward()
{
    assert(root);
    if (timeSlider->value() != timeSlider->maxValue())
    {
        setCurrentTime(getCurrentTime() + root->getDt());
        slot_loadrecord_timevalue(false);
    }
}
void QSofaRecorder::slot_forward()
{
    if (timeSlider->value() != timeSlider->maxValue())
    {
        setCurrentTime(getFinalTime());
        slot_sliderValue(timeSlider->maxValue());
        loadSimulation();
    }


}
void QSofaRecorder::slot_loadrecord_timevalue(bool updateTime)
{
    double init_time = getInitialTime();
    double final_time = getFinalTime();
    double current_time = getCurrentTime();

    int value = (int)((current_time-init_time)/((float)(final_time-init_time))*timeSlider->maxValue());

    if (value > timeSlider->minValue())
        slot_sliderValue(value, updateTime);
    else if ( value < timeSlider->maxValue())
        slot_sliderValue(value, updateTime);
    else if (!updateTime && value == timeSlider->maxValue())
        slot_sliderValue(value, updateTime);
    else if (value <= timeSlider->minValue())
        slot_sliderValue(timeSlider->minValue());

    if (current_time >= final_time)
    {
        playforward->setOn(false);
        slot_sliderValue(timeSlider->maxValue());
        setCurrentTime(final_time);
        slot_playforward();
    }
}
void QSofaRecorder::loadSimulation(bool one_step )
{
    assert(root);
    if (timeSlider->maxValue() == 0)
    {
        playforward->setOn(false);
        return;
    }

    float sleep_time = clock()/(float)CLOCKS_PER_SEC;
    double time=getCurrentTime();

    //update the time in the context
    root->execute< UpdateSimulationContextVisitor >(sofa::core::ExecParams::defaultInstance());
    root->execute< VisualUpdateVisitor >(sofa::core::ExecParams::defaultInstance());
    //read the state for the current time
    ReadStateModifier v(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, time);
    v.addTag(Tag("AutoRecord"));
    v.execute(root);
    if (!one_step)
        sleep(root->getDt(), sleep_time);
    emit NewTime();

    root->execute< sofa::simulation::UpdateMappingVisitor >(sofa::core::ExecParams::defaultInstance());

}
void QSofaRecorder::sleep(float seconds, float init_time)
{
    unsigned int t = 0;
    clock_t goal = (clock_t) (seconds + init_time);
    while (goal > clock()/(float)CLOCKS_PER_SEC) t++;
}

void QSofaRecorder::slot_sliderValue(int value, bool updateTime)
{
    double init_time   = getInitialTime();
    double final_time  = getFinalTime();
    if (updateTime)
    {
        double time = init_time + value/((float)timeSlider->maxValue())*(final_time-init_time);
        setCurrentTime(time);
    }
    if (timeSlider->value() != value)
    {
        timeSlider->setValue(value);
        timeSlider->update();
        if(! this->record->isOn())
            loadSimulation();
    }
}

void QSofaRecorder::addReadState(const std::string& writeSceneName, bool init)
{
    assert(! writeSceneName.empty());
    assert(root);

    ReadStateCreator v(writeSceneName,false, sofa::core::ExecParams::defaultInstance(),init);
    v.addTag(core::objectmodel::Tag("AutoRecord"));
    v.execute(root);
}

void QSofaRecorder::addWriteState(const std::string& writeSceneName )
{
    assert(! writeSceneName.empty());
    assert(root);
    //record X, V, but won't record in the Mapping
    WriteStateCreator v(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, writeSceneName, true, true, true, false);
    v.addTag(Tag("AutoRecord"));
    v.execute(root);
    std::cout << "Recording simulation with base name: " << writeSceneName << "\n";
}

bool QSofaRecorder::querySimulationName()
{


    std::string dir;
    bool ok;
    std::string filename(((RealGUI*)(qApp->mainWidget()))->windowFilePath().ascii());
    const std::string &parentDir=sofa::helper::system::SetDirectory::GetParentDir(filename.c_str());
    if (parentDir.empty()) dir = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + "/";
    else dir = parentDir + "/";

    QString text = QInputDialog::getText("Record Simulation", "Enter the name of your simulation:", QLineEdit::Normal,
            QString::null, &ok, this );
    if (ok && !text.isEmpty() )
    {
        simulationBaseName_ = dir +  text.ascii();
        writeSceneName_ = record_directory + text.ascii();
        return true;
    }
    else
    {
        return false;
    }
}



}//qt
}//gui
}//sofa
