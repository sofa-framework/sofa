#ifndef QSOFASCENE_H
#define QSOFASCENE_H

#include <QObject>
#include "../SofaGLScene.h"
#include <sofa/simulation/graph/DAGSimulation.h>
typedef sofa::simulation::graph::DAGSimulation SofaSimulation;
//typedef sofa::simulation::tree::TreeSimulation ParentSimulation;

namespace sofa {
namespace newgui {

/**
 * @brief The QSofaScene class is a SofaScene which can be connected to other Qt objects, such as viewers, using signals and slots.
 * It contains the basic simulation functions, but no graphics capabilities.
 *
 * @author Francois Faure, 2014
 */
class QSofaScene : public QObject, public SofaScene
{
    Q_OBJECT
public:
    explicit QSofaScene(QObject *parent = 0);

signals:
    /// Sent after step(), with the value of the time step
    void stepEnd( SReal dt );

public slots:
    /// Apply one simulation time step
    void step();
    /// Set the length of the simulation time step
    void setTimeStep( SReal dt );
    /// Length of the simulation time step
    SReal dt() const;
private:
    SReal _dt;

};


}//newgui
}//sofa

#endif // QSOFASCENE_H
