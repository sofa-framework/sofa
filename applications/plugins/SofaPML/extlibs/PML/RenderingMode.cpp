/***************************************************************************
                              RenderingMode.cpp
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2004/05/11 10:04:20 $
    Version           : $Revision: 1.7 $
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "PhysicalModelIO.h"
#include "RenderingMode.h"

// --------------- constructors -------------------
RenderingMode::RenderingMode(const RenderingMode::Mode mode) {
    // set the visibility flags
    setMode(mode);
}

RenderingMode::RenderingMode(const bool surface, const bool wireframe, const bool points) {
    setVisible(SURFACE, surface);
    setVisible(WIREFRAME, wireframe);
    setVisible(POINTS,points);
}


// --------------- setVisible -------------------
/**
 * Set a rendering mode visible or not.
 * Set the rendering mode \param mode (Mode) visible if \param value is TRUE, unvisible otherwise.
 */
void RenderingMode::setVisible(const RenderingMode::Mode mode, const bool value) {
    switch (mode) {
    case SURFACE:
        surfaceVisibility = value;
        break;
    case WIREFRAME:
        wireframeVisibility = value;
        break;
    case POINTS:
        pointsVisibility = value;
        break;
    case POINTS_AND_SURFACE:
        wireframeVisibility = !value;
        surfaceVisibility = pointsVisibility = value;
        break;
    case WIREFRAME_AND_SURFACE_AND_POINTS:
        surfaceVisibility = wireframeVisibility = pointsVisibility = value;
        break;
    case WIREFRAME_AND_SURFACE:
        surfaceVisibility = wireframeVisibility = value;
        pointsVisibility = !value;
        break;
    case WIREFRAME_AND_POINTS:
        pointsVisibility = wireframeVisibility = value;
        surfaceVisibility = !value;
        break;
    default:
        break;
    }
}

// --------------- setMode -------------------
void RenderingMode::setMode(const RenderingMode::Mode mode) {
    switch (mode) 	{
    case NONE:
        surfaceVisibility = wireframeVisibility = pointsVisibility = false;
        break;
    case POINTS:
        surfaceVisibility = wireframeVisibility = false;
        pointsVisibility = true;
        break;
    case POINTS_AND_SURFACE:
        wireframeVisibility = false;
        surfaceVisibility = pointsVisibility = true;
        break;
    case SURFACE:
        surfaceVisibility = true;
        wireframeVisibility = pointsVisibility = false;
        break;
    case WIREFRAME_AND_SURFACE_AND_POINTS:
        surfaceVisibility = wireframeVisibility = pointsVisibility = true;
        break;
    case WIREFRAME_AND_SURFACE:
        surfaceVisibility = wireframeVisibility = true;
        pointsVisibility = false;
        break;
    case WIREFRAME_AND_POINTS:
        pointsVisibility = wireframeVisibility = true;
        surfaceVisibility = false;
        break;
    case WIREFRAME:
        surfaceVisibility = pointsVisibility = false;
        wireframeVisibility = true;
        break;
    }
}



// --------------- isVisible -------------------
/**
 * Return if a rendering mode is currently visible or not.
 * Return TRUE if the rendering mode \param mode (Mode) is currently visible, FALSE otherwise..
 */
bool RenderingMode::isVisible(const RenderingMode::Mode mode) const {
    switch (mode) {
    case SURFACE:
        return surfaceVisibility;
        break;
    case WIREFRAME:
        return wireframeVisibility;
        break;
    case POINTS:
        return pointsVisibility;
        break;
    case POINTS_AND_SURFACE:
        return (surfaceVisibility && pointsVisibility);
        break;
    case WIREFRAME_AND_SURFACE:
        return (wireframeVisibility && surfaceVisibility);
        break;
    case WIREFRAME_AND_POINTS:
        return (wireframeVisibility && pointsVisibility);
        break;
    case WIREFRAME_AND_SURFACE_AND_POINTS:
        return (wireframeVisibility && surfaceVisibility && pointsVisibility);
        break;
    default:
        return false;
        break;
    }
}

// --------------- isVisible -------------------
bool RenderingMode::isVisible() const {
    // true if at least a mode is visible
    return (surfaceVisibility || wireframeVisibility || pointsVisibility);
}

// ----------------- getMode-----------------------
RenderingMode::Mode RenderingMode::getMode() const {
    if (pointsVisibility) {
        if (surfaceVisibility) {
            if (wireframeVisibility) {
                return WIREFRAME_AND_SURFACE_AND_POINTS;
            } else {
                return POINTS_AND_SURFACE;
            }
        } else {
            if (wireframeVisibility) {
                return WIREFRAME_AND_POINTS;
            } else {
                return POINTS;
            }
        }
    } else {
        if (surfaceVisibility) {
            if (wireframeVisibility) {
                return WIREFRAME_AND_SURFACE;
            } else {
                return SURFACE;
            }
        } else {
            if (wireframeVisibility) {
                return WIREFRAME;
            } else {
                return NONE;
            }
        }
    }
}

// ----------------- getModeString -----------------------
std::string RenderingMode::getModeString() const {
    std::string n;

    if (pointsVisibility) {
        if (surfaceVisibility) {
            if (wireframeVisibility) {
                n = "WIREFRAME_AND_SURFACE_AND_POINTS";
            } else {
                n="POINTS_AND_SURFACE";
            }
        } else {
            if (wireframeVisibility) {
                n="WIREFRAME_AND_POINTS";
            } else {
                n="POINTS";
            }
        }
    } else {
        if (surfaceVisibility) {
            if (wireframeVisibility) {
                n="WIREFRAME_AND_SURFACE";
            } else {
                n="SURFACE";
            }
        } else {
            if (wireframeVisibility) {
                n="WIREFRAME";
            } else {
                n="NONE";
            }
        }
    }

    return n;
}

