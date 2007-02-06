#ifndef SOFA_COMPONENTS_WRITESTATE_H
#define SOFA_COMPONENTS_WRITESTATE_H

#include "Sofa-old/Core/ForceField.h"
#include "Sofa-old/Core/MechanicalModel.h"
#include "Sofa-old/Abstract/BaseObject.h"
#include "Sofa-old/Abstract/Event.h"
#include "Sofa-old/Components/AnimateBeginEvent.h"
#include "Sofa-old/Components/AnimateEndEvent.h"

#include <fstream>

namespace Sofa
{

namespace Components
{

using namespace Common;
/** Write State vectors to file at each timestep
*/
template<class DataTypes>
class WriteState: public Abstract::BaseObject
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

protected:
    Common::DataField < std::string > f_filename;
    Common::DataField < bool > f_writeX;
    Common::DataField < bool > f_writeV;
    Common::DataField < double > f_interval;

    Core::MechanicalModel<DataTypes>* mmodel;
    std::ofstream* outfile;
    double nextTime;

public:
    WriteState(Core::MechanicalModel<DataTypes>* =NULL);

    virtual ~WriteState();

    virtual void init();

    virtual void reset();

    virtual void handleEvent(Sofa::Abstract::Event* event);
};

} // namespace Components

} // namespace Sofa

#endif
