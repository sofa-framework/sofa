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
/*
 * (c) Copyright 1993, 1994, Silicon Graphics, Inc.
 * ALL RIGHTS RESERVED
 * Permission to use, copy, modify, and distribute this software for
 * any purpose and without fee is hereby granted, provided that the above
 * copyright notice appear in all copies and that both the copyright notice
 * and this permission notice appear in supporting documentation, and that
 * the name of Silicon Graphics, Inc. not be used in advertising
 * or publicity pertaining to distribution of the software without specific,
 * written prior permission.
 *
 * THE MATERIAL EMBODIED ON THIS SOFTWARE IS PROVIDED TO YOU "AS-IS"
 * AND WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR OTHERWISE,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR
 * FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL SILICON
 * GRAPHICS, INC.  BE LIABLE TO YOU OR ANYONE ELSE FOR ANY DIRECT,
 * SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY
 * KIND, OR ANY DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION,
 * LOSS OF PROFIT, LOSS OF USE, SAVINGS OR REVENUE, OR THE CLAIMS OF
 * THIRD PARTIES, WHETHER OR NOT SILICON GRAPHICS, INC.  HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH LOSS, HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE
 * POSSESSION, USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * US Government Users Restricted Rights
 * Use, duplication, or disclosure by the Government is subject to
 * restrictions set forth in FAR 52.227.19(c)(2) or subparagraph
 * (c)(1)(ii) of the Rights in Technical Data and Computer Software
 * clause at DFARS 252.227-7013 and/or in similar or successor
 * clauses in the FAR or the DOD or NASA FAR Supplement.
 * Unpublished-- rights reserved under the copyright laws of the
 * United States.  Contractor/manufacturer is Silicon Graphics,
 * Inc., 2011 N.  Shoreline Blvd., Mountain View, CA 94039-7311.
 *
 * OpenGL(TM) is a trademark of Silicon Graphics, Inc.
 */
/*
 * trackball.h
 * A virtual trackball implementation
 * Written by Gavin Bell for Silicon Graphics, November 1988.
 */

/*
 * Pass the x and y coordinates of the last and current positions of
 * the mouse, scaled so they are from (-1.0 ... 1.0).
 *
 * The resulting rotation is returned as a quaternion rotation in the
 * first paramater.
 */

#ifndef SOFA_HELPER_GL_TRACKBALL_H
#define SOFA_HELPER_GL_TRACKBALL_H

#include <sofa/defaulttype/Quat.h>

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace gl
{

class SOFA_HELPER_API Trackball
{
public:
// 	typedef Quater<double> Quaternion;

    Trackball();
    ~Trackball();

    // Accessors
    void SetQuaternion(sofa::defaulttype::Quaternion Q);
    sofa::defaulttype::Quaternion GetQuaternion(void);

    // Pass the x and y coordinates of the last and current positions of
    // the mouse, scaled so they are from (-1.0 ... 1.0).
    // The resulting rotation is returned as a quaternion rotation
    void ComputeQuaternion(double p1x, double p1y, double p2x,
            double p2y);

    // This function computes a quaternion based on an axis (defined by
    // the given vector) and an angle about which to rotate.  The angle is
    // expressed in radians.  The result is put into the third argument.
    void AxisToQuat(double a[3], double phi, double q[4]);

private:
    sofa::defaulttype::Quaternion	_quat;
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif
