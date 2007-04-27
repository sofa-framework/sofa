/***************************************************************************
                               RenderingMode.h
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:22 $
    Version           : $Revision: 1.10 $
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef RENDERING_MODE_H
#define RENDERING_MODE_H
#include "PhysicalModelIO.h"

/**
* Handle rendering options (surface and wireframe) of an Object3D.
  * $Revision: 1.10 $
*/
class RenderingMode {
public:
    /** This is a duplicate of RenderingMode Mode.... BEURK!!! */
    enum Mode {
        NONE,
        POINTS,
        POINTS_AND_SURFACE,
        SURFACE,
        WIREFRAME_AND_SURFACE,
        WIREFRAME_AND_POINTS,
        WIREFRAME,
        WIREFRAME_AND_SURFACE_AND_POINTS,
    };

    /// default constructor with initialisation
    RenderingMode(const Mode mode = SURFACE);
    /** another constructor provided for conveniance
      * @param surface tells if by default the surface is visible
      * @param wireframe tells if by default the surface is visible
      * @param points tells if by default the surface is visible
      */
    RenderingMode(const bool surface, const bool wireframe, const bool points);

    /** Set a rendering mode visible or not.*/
    void setVisible(const Mode mode, const bool value);
    /** Return if a rendering mode is currently visible or not.*/
    bool isVisible(const Mode mode) const;
    /** Return true if at least a mode is currently visible, false otherwise.*/
    bool isVisible() const;
    /** set a vizualisation mode */
    void setMode(const Mode mode);
    /** get current mode */
    RenderingMode::Mode getMode() const;
    /// get the string equivalent to the enum rendering mode
    std::string getModeString() const;
private:

    // Visibility flags
    /** Flag indicating weither the surface mode is currenly visible or not. */
    bool surfaceVisibility;
    /** Flag indicating weither the wireframe mode is currenly visible or not. */
    bool wireframeVisibility;
    /** Flag indicating weither the points mode is currenly visible or not. */
    bool pointsVisibility;
};

#endif
