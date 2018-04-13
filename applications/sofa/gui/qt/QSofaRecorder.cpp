/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "QSofaRecorder.h"
#include "RealGUI.h"
#include <time.h>

#include <sofa/core/objectmodel/Tag.h>

#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/simulation/WriteStateVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>

#include <SofaGeneralLoader/ReadState.h>
#include <SofaExporter/WriteState.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>

#include "RealGUI.h"

#include <QToolTip>
#include <QInputDialog>
#include <QHBoxLayout>


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
    QHBoxLayout* layout = new QHBoxLayout(this);
    layout->setMargin(0);
    layout->setSpacing(0);
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
    record                 = new QPushButton( "Record", this);
    record->setCheckable(true);
    record->setChecked(false);
    layout->addWidget(record);
    backward               = new QPushButton( "Backward", this);
    layout->addWidget(backward);
    stepbackward           = new QPushButton( "Step Backward", this );
    layout->addWidget(stepbackward);
    playforward            = new QPushButton( "Play Forward", this);
    playforward->setCheckable(true);
    layout->addWidget(playforward);
    stepforward            = new QPushButton( "Step Forward", this);
    layout->addWidget(stepforward);
    forward                = new QPushButton( "Forward", this);
    layout->addWidget(forward);

    timeRecord = new QLabel("T=",this);
    layout->addWidget(timeRecord);
    loadRecordTime = new QLineEdit(this);
    loadRecordTime->setMaximumSize(QSize(75, 100));
    layout->addWidget(loadRecordTime);
    timeSlider = new QSlider( Qt::Horizontal,this);
    timeSlider->setWindowTitle("Time Slider");
    timeSlider->setTickPosition(QSlider::TicksBothSides);
    timeSlider->setMinimum(0);
    timeSlider->setMaximum(0);
    layout->addWidget(timeSlider);

    finalTime = new QLabel( "End:", this );
    finalTime->setMinimumSize ( finalTime->sizeHint() );
    layout->addWidget(finalTime);

//    QToolTip::add(record        , tr( "Record" ) );
//    QToolTip::add(backward      , tr( "Load Initial Time" ) );
//    QToolTip::add(stepbackward  , tr( "Make one step backward" ) );
//    QToolTip::add(playforward   , tr( "Continuous play forward" ) );
//    QToolTip::add(stepforward   , tr( "Make one step forward" ) );
//    QToolTip::add(forward       , tr( "Load Final Time" ) );
    record->setToolTip(tr( "Record" ) );
    backward->setToolTip(tr( "Load Final Time" ) );
    stepbackward->setToolTip(tr( "Make one step backward" ) );
    playforward->setToolTip(tr( "Continuous play forward" ) );
    stepforward->setToolTip(tr( "Make one step forward" ) );
    forward->setToolTip(tr( "Load Final Time" ) );

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

    timeSlider->setMaximum((int)((atof(endT.c_str())-atof(initT.c_str()))/(dt)+0.5));

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
    if (record->isChecked())
    {
        setCurrentTime(time);
        double final_time = getFinalTime();

        if ((int)(1000*final_time) < (int)(1000*time))
        {
            setFinalTime(time);
            timeSlider->setMaximum(timeSlider->maximum()+1);
            timeSlider->setValue(timeSlider->maximum());
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
    timeSlider->setMinimum(0);
    timeSlider->setMaximum(0);
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
            record->setCheckable( false );
            return;
        }

    }
    else
    {
        //Halt the simulation.
        emit RecordSimulation(false);
        //Save simulation file
        std::string FileName(((RealGUI*)(QApplication::topLevelWidgets()[0]))->windowFilePath().toStdString());
        std::string simulationFileName = simulationBaseName_ + ".simu";
        std::ofstream out(simulationFileName.c_str());

        if (!out.fail())
        {
            out << sofa::helper::system::DataRepository.getFile ( FileName ) << " " << initialTime->text().toStdString() << " " << finalTime->text().toStdString() << " " << root->getDt() << " baseName: "<<writeSceneName_;
            out.close();
        }
        std::cout << "Simulation parameters saved in "<<simulationFileName<<std::endl;
    }
    //Change the state of the writers
    WriteStateActivator v_write(sofa::core::ExecParams::defaultInstance(), value);
    v_write.addTag(Tag("AutoRecord"));
    v_write.execute(root);
    ReadStateActivator v_read(sofa::core::ExecParams::defaultInstance(), false);
    v_read.addTag(Tag("AutoRecord"));
    v_read.execute(root);
}
void QSofaRecorder::slot_backward()
{
    if (timeSlider->value() != timeSlider->minimum())
    {
        setCurrentTime(getInitialTime());
        slot_sliderValue(timeSlider->minimum());
        loadSimulation();
    }
}
void QSofaRecorder::slot_stepbackward()
{
    assert(root);
    playforward->setCheckable(false);
    double init_time  = getInitialTime();
    double time = getCurrentTime() - root->getDt();
    if (time < init_time) time = init_time;

    setCurrentTime(time);
    slot_loadrecord_timevalue(false);
}
void QSofaRecorder::slot_playforward()
{
    if (playforward->isChecked() )
    {
        if (timeSlider->value() == timeSlider->maximum())
        {
            playforward->setChecked(false);
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
    if (timeSlider->value() != timeSlider->maximum())
    {
        setCurrentTime(getCurrentTime() + root->getDt());
        slot_loadrecord_timevalue(false);
    }
}
void QSofaRecorder::slot_forward()
{
    if (timeSlider->value() != timeSlider->maximum())
    {
        setCurrentTime(getFinalTime());
        slot_sliderValue(timeSlider->maximum());
        loadSimulation();
    }


}
void QSofaRecorder::slot_loadrecord_timevalue(bool updateTime)
{
    double init_time = getInitialTime();
    double final_time = getFinalTime();
    double current_time = getCurrentTime();

    int value = (int)((current_time-init_time)/((float)(final_time-init_time))*timeSlider->maximum());

    if (value > timeSlider->minimum())
        slot_sliderValue(value, updateTime);
    else if ( value < timeSlider->maximum())
        slot_sliderValue(value, updateTime);
    else if (!updateTime && value == timeSlider->maximum())
        slot_sliderValue(value, updateTime);
    else if (value <= timeSlider->minimum())
        slot_sliderValue(timeSlider->minimum());

    if (current_time >= final_time)
    {
        playforward->setChecked(false);
        slot_sliderValue(timeSlider->maximum());
        setCurrentTime(final_time);
        slot_playforward();
    }
}
void QSofaRecorder::loadSimulation(bool one_step )
{
    assert(root);
    if (timeSlider->maximum() == 0)
    {
        playforward->setChecked(false);
        return;
    }

    float sleep_time = clock()/(float)CLOCKS_PER_SEC;
    double time=getCurrentTime();

    //update the time in the context
    root->execute< UpdateSimulationContextVisitor >(sofa::core::ExecParams::defaultInstance());
    root->execute< VisualUpdateVisitor >(sofa::core::ExecParams::defaultInstance());
    //read the state for the current time
    ReadStateModifier v(sofa::core::ExecParams::defaultInstance(), time);
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
        double time = init_time + value/((float)timeSlider->maximum())*(final_time-init_time);
        setCurrentTime(time);
    }
    if (timeSlider->value() != value)
    {
        timeSlider->setValue(value);
        timeSlider->update();
        if(! this->record->isChecked())
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
    WriteStateCreator v(sofa::core::ExecParams::defaultInstance(), writeSceneName, true, true, true, false);
    v.addTag(Tag("AutoRecord"));
    v.execute(root);
    std::cout << "Recording simulation with base name: " << writeSceneName << "\n";
}

bool QSofaRecorder::querySimulationName()
{


    std::string dir;
    bool ok;
    std::string filename(((RealGUI*)(QApplication::topLevelWidgets()[0]))->windowFilePath().toStdString());
    const std::string &parentDir=sofa::helper::system::SetDirectory::GetParentDir(filename.c_str());
    if (parentDir.empty()) dir = sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + "/";
    else dir = parentDir + "/";

    QString text = QInputDialog::getText(this, "Record Simulation", "Enter the name of your simulation:", QLineEdit::Normal,
            QString::null, &ok);
    if (ok && !text.isEmpty() )
    {
        simulationBaseName_ = dir +  text.toStdString();
        writeSceneName_ = record_directory + text.toStdString();
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
