/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MISC_MONITOR_H
#define SOFA_COMPONENT_MISC_MONITOR_H
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/types/RGBAColor.h>

namespace sofa
{

namespace component
{

namespace misc
{

template <class DataTypes>
class Monitor: public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(Monitor,DataTypes), core::objectmodel::BaseObject);

    typedef sofa::helper::types::RGBAColor RGBAColor;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

protected:
    Monitor ();
    ~Monitor ();
public:
    //init data
    virtual void init ();

    //reset Monitored values
    virtual void reset ();

    /**initialize gnuplot files
    *called when ExportGnuplot box is checked
    */
    virtual void reinit();

    /**function called at every step of simulation;
    *store mechanical state vectors (forces, positions, velocities) into
    *the MonitorData nested class. The filter (which position(s), velocity(ies) or *force(s) are displayed) is made in the gui
    */
    virtual void handleEvent( core::objectmodel::Event* ev );

    virtual void draw (const core::visual::VisualParams* vparams);

    ///create gnuplot files
    virtual void initGnuplot ( const std::string path );

    ///write in gnuplot files the Monitored desired data (velocities,positions,forces)
    virtual void exportGnuplot ( Real time );

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const Monitor<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    /// Editable Data
    Data < helper::vector<unsigned int> > indices;

    Data < bool > saveXToGnuplot;
    Data < bool > saveVToGnuplot;
    Data < bool > saveFToGnuplot;

    Data < bool > showPositions;
    Data <RGBAColor > positionsColor;

    Data < bool > showVelocities;
    Data< RGBAColor > velocitiesColor;

    Data < bool > showForces;
    Data< RGBAColor > forcesColor;

    Data < double > showMinThreshold;

    Data < bool > showTrajectories;
    Data < double > trajectoriesPrecision;
    Data< RGBAColor > trajectoriesColor;

    Data< double > showSizeFactor;


protected:

    std::ofstream* saveGnuplotX;
    std::ofstream* saveGnuplotV;
    std::ofstream* saveGnuplotF;

    //position, velocity and force of the mechanical object monitored;
    const VecCoord * X;
    const VecDeriv * V;
    const VecDeriv * F;


    ///use for trajectoriesPrecision (save value only if trajectoriesPrecision <= internalDt)
    double internalDt;

    //store all the monitored positions, for trajectories display
    sofa::helper::vector < sofa::helper::vector<Coord> > savedPos;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MISC_MONITOR_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_VALIDATION_API Monitor<defaulttype::Vec3dTypes>;
extern template class SOFA_VALIDATION_API Monitor<defaulttype::Vec6dTypes>;
extern template class SOFA_VALIDATION_API Monitor<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_VALIDATION_API Monitor<defaulttype::Vec3fTypes>;
extern template class SOFA_VALIDATION_API Monitor<defaulttype::Vec6fTypes>;
extern template class SOFA_VALIDATION_API Monitor<defaulttype::Rigid3fTypes>;
#endif
#endif


} // namespace misc

} // namespace component

} // namespace sofa

#endif
