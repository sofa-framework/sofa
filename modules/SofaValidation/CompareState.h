/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

#ifndef SOFA_COMPONENT_MISC_COMPARESTATE_H
#define SOFA_COMPONENT_MISC_COMPARESTATE_H

#include <sofa/SofaGeneral.h>
#include <SofaLoader/ReadState.h>
#include <sofa/simulation/common/Visitor.h>

#include <fstream>


namespace sofa
{

namespace component
{

namespace misc
{

/** Compare State vectors from file at each timestep
*/
class SOFA_VALIDATION_API CompareState: public ReadState
{
public:
    SOFA_CLASS(CompareState,ReadState);
protected:
    /** Default constructor
    */
    CompareState();
public:
    void handleEvent(sofa::core::objectmodel::Event* event);

    /// Compute the total errors (positions and velocities)
    void processCompareState();

    /** Pre-construction check method called by ObjectFactory.
    Check that DataTypes matches the MechanicalState.*/
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::BaseMechanicalState*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    /// Return the total errors (position and velocity)
    double getTotalError() {return totalError_X + totalError_V;}
    /// Return the total errors (position and velocity)
    double getErrorByDof() {return dofError_X + dofError_V;}

    virtual void draw(const core::visual::VisualParams* vparams);

protected :
    /// total error for positions
    double totalError_X;
    double dofError_X;
    /// total error for velocities
    double totalError_V;
    double dofError_V;
    /// last time, position and velocity (for draw)
    double last_time;
    std::string last_X, last_V;
    std::vector<std::string> nextValidLines;
};

/// Create CompareState component in the graph each time needed
class SOFA_VALIDATION_API CompareStateCreator: public Visitor
{
public:
    CompareStateCreator(const core::ExecParams* params);
    CompareStateCreator(const std::string &n, const core::ExecParams* params, bool i=true, int c=0);
    virtual Result processNodeTopDown( simulation::Node*  );

    void setSceneName(std::string &n) { sceneName = n; }
    void setCounter(int c) { counterCompareState = c; }
    void setCreateInMapping(bool b) { createInMapping=b; }
    virtual const char* getClassName() const { return "CompareStateCreator"; }

protected:

    void addCompareState(sofa::core::behavior::BaseMechanicalState *ms, simulation::Node* gnode);
    std::string sceneName;
    std::string extension;
    bool createInMapping;
    bool init;
    int counterCompareState; //avoid to have two same files if two mechanical objects has the same name
};

class SOFA_VALIDATION_API CompareStateResult: public Visitor
{
public:
    CompareStateResult(const core::ExecParams* params) : Visitor(params)
    { error=errorByDof=0; numCompareState=0;}
    virtual Result processNodeTopDown( simulation::Node*  );

    double getTotalError() { return error; }
    double getErrorByDof() { return errorByDof; }
    unsigned int getNumCompareState() { return numCompareState; }
    virtual const char* getClassName() const { return "CompareStateResult"; }
protected:
    double error;
    double errorByDof;
    unsigned int numCompareState;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
