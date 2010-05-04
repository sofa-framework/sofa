/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#ifndef PLUGINS_PIM_GUI_MOUSEOPERATIONS_H
#define PLUGINS_PIM_GUI_MOUSEOPERATIONS_H

#include <sofa/gui/MouseOperations.h>

namespace plugins
{
namespace pim
{
namespace gui
{

class SOFA_SOFAGUI_API SculptOperation : public sofa::gui::Operation
{
public:
    SculptOperation():force(1), scale(20), checkedFix(false), checkedInflate(false), checkedDeflate(false), animated(false) {}
    virtual ~SculptOperation();
    virtual void start() ;
    virtual void execution() {};
    virtual void end() ;
    virtual void wait() ;

    void setForce(double f) {force = f;}
    virtual double getForce() const { return force;}
    void setScale(double s) {scale = s;}
    virtual double getScale() const {return scale;}
    void setMass(double m) {mass = m;}
    virtual double getMass() const { return mass;}
    void setStiffness(double s) {stiffness = s;}
    virtual double getStiffness() const { return stiffness;}
    void setDamping(double d) {damping = d;}
    virtual double getDamping() const { return damping;}

    virtual bool isCheckedFix() const {return checkedFix;};
    void setCheckedFix(bool b) {checkedFix = b;};
    virtual bool isCheckedInflate() const {return checkedInflate;};
    void setCheckedInflate(bool b) {checkedInflate = b;};
    virtual bool isCheckedDeflate() const {return checkedDeflate;};
    void setCheckedDeflate(bool b) {checkedDeflate = b;};
    virtual bool isAnimated() const {return animated;};

    static std::string getDescription() {return "Sculpt an object using the Mouse";}
protected:
    double force, scale, mass, stiffness, damping;
    bool checkedFix, checkedInflate, checkedDeflate, animated;
};

} // namespace gui
} // namespace pim
} // namespace plugins

#endif
