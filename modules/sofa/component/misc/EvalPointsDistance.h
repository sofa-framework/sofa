/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_COMPONENT_MISC_EVALPOINTSDISTANCE_H
#define SOFA_COMPONENT_MISC_EVALPOINTSDISTANCE_H

#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/ObjectRef.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

using namespace core::behavior;

/** Compute the distance between point/node positions in two objects
*/
template<class TDataTypes>
class EvalPointsDistance: public virtual sofa::core::objectmodel::BaseObject
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(EvalPointsDistance,TDataTypes), sofa::core::objectmodel::BaseObject);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    /// Rendering of lines between associated points (activation)
    Data < bool > f_draw;
    /// Output file name
    sofa::core::objectmodel::DataFileName f_filename;
    /// Period between outputs
    Data < double > f_period;
    /// Computed distances (mean, min, max, standard deviation)
    Data < double > distMean, distMin, distMax, distDev;
    /// Relative computed distances (mean, min, max, standard deviation)
    Data < double > rdistMean, rdistMin, rdistMax, rdistDev;

    /** Default constructor
    */
    EvalPointsDistance(MechanicalState<DataTypes>* , MechanicalState<DataTypes>*);
    virtual ~EvalPointsDistance();

    /// Init the computation of the distances
    virtual void init();
    /// Reset the computation of the distances
    virtual void reset();

    /** Distance computation */

    /// Get the nodes/points coordinates of the two objects and compute the distances
    virtual SReal eval();
    /// Compute the distances between the two objects
    virtual SReal doEval(const VecCoord& x1, const VecCoord& x2, const VecCoord& x0);


    virtual void handleEvent(sofa::core::objectmodel::Event* event);
    virtual void draw(const core::visual::VisualParams* vparams);
    virtual void doDraw(const VecCoord& x1, const VecCoord& x2);

    /// Retrieve the associated MechanicalState (First model)
    core::behavior::MechanicalState<DataTypes>* getMState1() { return mstate1; }
    core::behavior::BaseMechanicalState* getMechModel1() { return mstate1; }

    /// Retrieve the associated MechanicalState (Second model)
    core::behavior::MechanicalState<DataTypes>* getMState2() { return mstate2; }
    core::behavior::BaseMechanicalState* getMechModel2() { return mstate2; }


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::behavior::MechanicalState<DataTypes>* _ms0=NULL;
        core::behavior::MechanicalState<DataTypes>* _ms1=NULL;
        core::behavior::MechanicalState<DataTypes>* _ms2=NULL;


        if(arg->getAttribute("object1",NULL) != NULL)
        {
            _ms1 = sofa::core::objectmodel::ObjectRef::parse< core::behavior::MechanicalState<TDataTypes> >("object1", arg);
        }

        if(arg->getAttribute("object2",NULL) != NULL)
        {
            _ms2 = sofa::core::objectmodel::ObjectRef::parse< core::behavior::MechanicalState<TDataTypes> >("object2", arg);
        }

        _ms0 = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState());

        if( (_ms0==NULL) && (_ms1==NULL) && (_ms2==NULL) )
            return false;
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::behavior::MechanicalState<DataTypes>* _ms0=NULL;
        core::behavior::MechanicalState<DataTypes>* _ms1=NULL;
        core::behavior::MechanicalState<DataTypes>* _ms2=NULL;

        std::string _msPath1;
        std::string _msPath2;

        if(arg)
        {

            if(arg->getAttribute("object1",NULL) != NULL)
            {
                _ms1 = sofa::core::objectmodel::ObjectRef::parse< core::behavior::MechanicalState<TDataTypes> >("object1", arg);
                _msPath1 = arg->getAttribute("object1");
            }

            if(arg->getAttribute("object2",NULL) != NULL)
            {
                _ms2 = sofa::core::objectmodel::ObjectRef::parse< core::behavior::MechanicalState<TDataTypes> >("object2", arg);
                _msPath2 = arg->getAttribute("object2");
            }

            _ms0 = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState());
        }

        if(_ms1 == NULL)
        {
            _ms1 = _ms0;
            _msPath1.clear();
            _msPath1 = "@" + _ms0->getName();
        }
        if(_ms2 == NULL)
        {
            _ms2 = _ms0;
            _msPath2.clear();
            _msPath2 = "@" + _ms0->getName();
        }

        typename T::SPtr obj = sofa::core::objectmodel::New<T>(_ms1,_ms2);
        obj->setPathToMS1(_msPath1);
        obj->setPathToMS2(_msPath2);
        if (context)
        {
            obj->setPeriod(context->getDt());
            context->addObject(obj);
        }
        if (arg) obj->parse(arg);

        return obj;
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const EvalPointsDistance<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    void setPathToMS1(const std::string &o) {m_msPath1.setValue(o);}
    void setPathToMS2(const std::string &o) {m_msPath2.setValue(o);}
    void setPeriod(const double& _dt)      {f_period.setValue(_dt);}

protected:
    /// First model mechanical state
    core::behavior::MechanicalState<DataTypes>* mstate1;
    /// Second model mechanical state
    core::behavior::MechanicalState<DataTypes>* mstate2;
    /// output file
    std::ofstream* outfile;
    /// time value for the distance computations
    double lastTime;
    sofa::core::objectmodel::DataObjectRef m_msPath1;
    sofa::core::objectmodel::DataObjectRef m_msPath2;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
