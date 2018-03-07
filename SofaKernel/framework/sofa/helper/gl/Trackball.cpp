/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
 * Trackball code:
 *
 * Implementation of a virtual trackball.
 * Implemented by Gavin Bell, lots of ideas from Thant Tessman and
 *   the August '88 issue of Siggraph's "Computer Graphics," pp. 121-129.
 *
 * Modified by:
 *     Stephane Cotin
 *     Mitsubish Electric America
 *     June 1998
 */

#include <sofa/helper/gl/Trackball.h>
#include <cmath>

namespace sofa
{

namespace helper
{

namespace gl
{

using namespace sofa::defaulttype;

/*
 * This size should really be based on the distance from the center of
 * rotation to the point on the object underneath the mouse.  That
 * point would then track the mouse as closely as possible.  This is a
 * simple example, though, so that is left as an Exercise for the
 * Programmer.
 */
#define TRACKBALLSIZE  (0.8)

static double	tb_project_to_sphere(double, double, double);

// Constructor
Trackball::Trackball()
{
    _quat[0] = 0.0;
    _quat[1] = 0.0;
    _quat[2] = 0.0;
    _quat[3] = 0.0;
}

// Destructor
Trackball::~Trackball()
{
}

Quaternion Trackball::GetQuaternion(void)
{
    Quaternion	Q;

    Q[0] = _quat[0];
    Q[1] = _quat[1];
    Q[2] = _quat[2];
    Q[3] = _quat[3];

    return Q;
}

void Trackball::SetQuaternion(Quaternion Q)
{
    _quat[0] = Q[0];
    _quat[1] = Q[1];
    _quat[2] = Q[2];
    _quat[3] = Q[3];
}

/*
 * Ok, simulate a track-ball.  Project the points onto the virtual
 * trackball, then figure out the axis of rotation, which is the cross
 * product of P1 P2 and O P1 (O is the center of the ball, 0,0,0)
 * Note:  This is a deformed trackball-- is a trackball in the center,
 * but is deformed into a hyperbolic sheet of rotation away from the
 * center.  This particular function was chosen after trying out
 * several variations.
 *
 * It is assumed that the arguments to this routine are in the range
 * (-1.0 ... 1.0)
 */

void Trackball::ComputeQuaternion(double p1x, double p1y, double p2x,
        double p2y)
{
    double	phi;		/* how much to rotate about axis */
    double	t;

    if (p1x == p2x && p1y == p2y)
    {
        /* Zero rotation */
        _quat[0] = _quat[1] = _quat[2] = 0.0;
        _quat[3] = 1.0;

        return;
    }

    // First, figure out z-coordinates for projection of P1 and P2 to deformed sphere
    Vector3	p1	((SReal)p1x, (SReal)p1y, (SReal)tb_project_to_sphere(TRACKBALLSIZE, p1x, p1y));
    Vector3	p2	((SReal)p2x, (SReal)p2y, (SReal)tb_project_to_sphere(TRACKBALLSIZE, p2x, p2y));

    // Now, we want the cross product of P1 and P2
    Vector3	a = cross(p2,p1);

    // Figure out how much to rotate around that axis.
    Vector3	d	= p1 - p2;
    t = d.norm() / (2.0 * TRACKBALLSIZE);

    // Avoid problems with out-of-control values...
    if (t > 1.0)
    {
        t = 1.0;
    }
    if (t < -1.0)
    {
        t = -1.0;
    }
    phi = 2.0 * asin(t);

    _quat.axisToQuat(a, (SReal)phi);
}


// Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
// if we are away from the center of the sphere.
static double tb_project_to_sphere(double r, double x, double y)
{
    double	d, z;

    d = sqrt(x * x + y * y);
    if (d < r * 0.70710678118654752440)
    {
        /* Inside sphere */
        z = sqrt(r * r - d * d);
    }
    else
    {
        /* On hyperbola */
        double t = r / 1.41421356237309504880;
        z = t * t / d;
    }
    return z;
}

} // namespace gl

} // namespace helper

} // namespace sofa

