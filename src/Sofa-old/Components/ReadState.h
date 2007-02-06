#ifndef SOFA_COMPONENTS_READSTATE_H
#define SOFA_COMPONENTS_READSTATE_H

#include "Sofa/Core/ForceField.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/BaseObject.h"
#include "Sofa/Abstract/Event.h"
#include "Sofa/Components/AnimateBeginEvent.h"
#include "Sofa/Components/AnimateEndEvent.h"

#include <fstream>

namespace Sofa
{

namespace Components
{

using namespace Common;
/** Read State vectors to file at each timestep
*/
template<class DataTypes>
class ReadState: public Abstract::BaseObject
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

protected:
    Common::DataField < std::string > f_filename;
    Common::DataField < double > f_interval;

    Core::MechanicalModel<DataTypes>* mmodel;
    std::ifstream* infile;
    double nextTime;

public:
    ReadState(Core::MechanicalModel<DataTypes>* =NULL);

    virtual ~ReadState();

    virtual void init();

    virtual void reset();

    virtual void handleEvent(Sofa::Abstract::Event* event);
};

} // namespace Components

} // namespace Sofa

#endif
