#ifndef SOFA_GUI_QT_QSOFARECORDER_H
#define SOFA_GUI_QT_QSOFARECORDER_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef SOFA_QT4
#include <QWidget>
#include <QPushButton>
#include <QSlider>
#include <QLineEdit>
#include <QLabel>
#include <QTimer>
#else
#include <qwidget.h>
#include <qpushbutton.h>
#include <qslider.h>
#include <qlineedit.h>
#include <qlabel.h>
#include <qtimer.h>
#endif

namespace sofa
{
namespace simulation
{
class Node;
}
namespace gui
{
namespace qt
{

class QSofaRecorder : public QWidget
{
    Q_OBJECT
public:
    QSofaRecorder(QWidget* parent);
    void Clear(simulation::Node* root);
    void SetRecordDirectory(const std::string&);
    void SetSimulation(simulation::Node* root, const std::string& initT,
            const std::string& endT, const std::string& writeName);
    void setInitialTime(double);
    void setFinalTime(double);
    void setCurrentTime(double);
    void setTimeSimulation(double);
    void setFPS(double);
    void Start();
    static void sleep(float seconds, float init_time);

    inline double getInitialTime() const
    {
        std::string init_time = initialTime->text().ascii();
        init_time.resize(init_time.size()-2);
        return fabs(atof((init_time.substr(6)).c_str()));
    }
    inline double getFinalTime() const
    {
        std::string final_time = finalTime->text().ascii();
        final_time.resize(final_time.size()-2);
        return fabs(atof((final_time.substr(5)).c_str()));
    }
    inline double getCurrentTime() const
    {
        return fabs(atof(loadRecordTime->text().ascii()));
    }

    QLabel *getTimeLabel() {return timeLabel;};
    QLabel *getFPSLabel() {return fpsLabel;};
public slots:
    void UpdateTime();
    void TimerStart(bool);

signals:
    void RecordSimulation(bool);
    void NewTime();


protected:
    bool querySimulationName();
    void addWriteState(const std::string& writeSceneName);
    void addReadState(const std::string& writeSceneName,bool init);
    QPushButton* record;
    QPushButton* stepbackward;
    QPushButton* playforward;
    QPushButton* stepforward;
    QPushButton* forward;
    QPushButton* backward;
    QSlider*     timeSlider;
    QLabel*      timeRecord ;
    QLineEdit*   loadRecordTime;
    QLabel*      initialTime;
    QLabel*      finalTime;
    QLabel*      fpsLabel;
    QLabel*      timeLabel;
    QTimer*      timerStep;

    std::string  simulationBaseName_;
    std::string  writeSceneName_;
    std::string  record_directory;
    simulation::Node* root;
protected slots:
    void slot_recordSimulation( bool);
    void slot_backward( );
    void slot_stepbackward( );
    void slot_playforward( );
    void slot_stepforward( );
    void slot_forward( );
    void slot_loadrecord_timevalue(bool updateTime = true);
    void slot_sliderValue(int value, bool updateTime = true);
    void loadSimulation(bool one_step = false);
};
}
}
}

#endif //SOFA_GUI_QT_QSOFARECORDER_H
