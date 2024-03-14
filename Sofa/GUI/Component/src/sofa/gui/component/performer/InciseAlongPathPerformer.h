/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/gui/component/config.h>

#include <sofa/gui/component/performer/InteractionPerformer.h>

#include <sofa/gui/component/performer/TopologicalChangeManager.h>
#include <sofa/gui/component/performer/MouseInteractor.h>

namespace sofa::gui::component::performer
{
class InciseAlongPathPerformerConfiguration
{
public:
    void setIncisionMethod (int m) {currentMethod=m;}
    void setSnapingBorderValue (int m) {snapingBorderValue = m;}
    void setSnapingValue (int m) {snapingValue = m;}
    void setCompleteIncision (bool m) {finishIncision = m;}
    void setKeepPoint (bool m) {keepPoint = m;}


protected:
    int currentMethod;
    int snapingBorderValue;
    int snapingValue;
    bool finishIncision;
    bool keepPoint;

};


class SOFA_GUI_COMPONENT_API InciseAlongPathPerformer: public InteractionPerformer, public InciseAlongPathPerformerConfiguration
{
public:
    using Index = sofa::Index;
    InciseAlongPathPerformer(BaseMouseInteractor *i)
        : InteractionPerformer(i)
        , cpt(0)
        , fullcut(0)
        , initialNbTriangles(0)
        , initialNbPoints(0) {};

    ~InciseAlongPathPerformer() override;

    void start() override;

    void execute() override;

    void draw(const core::visual::VisualParams* vparams) override;

    BodyPicked& getFirstIncisionBodyPicked() {return firstIncisionBody;}

    BodyPicked& getLastBodyPicked() {return firstBody;}

    void setPerformerFreeze() override;

protected:
    /// Incision will be perfomed between firstIncisionBody (first point clicked) and firstBody (last point clicked in memory)
    void PerformCompleteIncision( );

    TopologicalChangeManager topologyChangeManager;
    BodyPicked startBody;
    BodyPicked firstBody;
    BodyPicked secondBody;
    BodyPicked firstIncisionBody;

    int cpt;
    bool fullcut;
    unsigned int initialNbTriangles;
    unsigned int initialNbPoints;
};
} // namespace sofa::gui::component::performer
