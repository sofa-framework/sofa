/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/core/objectmodel/DataFileName.h>

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
    SOFA_CLASS(SOFA_TEMPLATE(Monitor, DataTypes), core::objectmodel::BaseObject);

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
    /// init data
    virtual void init () override;

    /// reset Monitored values
    virtual void reset () override;

    /** initialize gnuplot files
        * called when ExportGnuplot box is checked
    */
    virtual void reinit() override;

    /** function called at every step of simulation;
        * store mechanical state vectors (forces, positions, velocities) into
        * the MonitorData nested class. The filter (which position(s), velocity(ies) or *force(s) are displayed) is made in the gui
    */
    virtual void handleEvent( core::objectmodel::Event* ev ) override;

    virtual void draw (const core::visual::VisualParams* vparams) override;

    /// create gnuplot files
    virtual void initGnuplot ( const std::string path );

    /// write in gnuplot files the Monitored desired data (velocities,positions,forces)
    virtual void exportGnuplot ( Real time );

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const Monitor<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    /// Editable Data
    Data< helper::vector<unsigned int> > d_indices;

    Data< bool > d_saveXToGnuplot;
    Data< bool > d_saveVToGnuplot;
    Data< bool > d_saveFToGnuplot;

    Data< bool > d_showPositions;
    Data<RGBAColor > d_positionsColor;

    Data< bool > d_showVelocities;
    Data< RGBAColor > d_velocitiesColor;

    Data< bool > d_showForces;
    Data< RGBAColor > d_forcesColor;

    Data< double > d_showMinThreshold;

    Data< bool > d_showTrajectories;
    Data< double > d_trajectoriesPrecision;
    Data< RGBAColor > d_trajectoriesColor;

    Data< double > d_showSizeFactor;
    core::objectmodel::DataFileName  d_fileName;

protected:

    std::ofstream* m_saveGnuplotX;
    std::ofstream* m_saveGnuplotV;
    std::ofstream* m_saveGnuplotF;

    const VecCoord * m_X; ///< positions of the mechanical object monitored;
    const VecDeriv * m_V; ///< velocities of the mechanical object monitored;
    const VecDeriv * m_F; ///< forces of the mechanical object monitored;


    double m_internalDt; ///< use for trajectoriesPrecision (save value only if trajectoriesPrecision <= internalDt)

    sofa::helper::vector < sofa::helper::vector<Coord> > m_savedPos; ///< store all the monitored positions, for trajectories display
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
