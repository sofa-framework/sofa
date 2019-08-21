/***************************************************************************
                          Direction.h  -  description
                             -------------------
    begin                : mar f�v 4 2003
    copyright            : (C) 2003 by Emmanuel Promayon
    email                : Emmanuel.Promayon@imag.fr
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#ifndef DIRECTION_H
#define DIRECTION_H

#include "xmlio.h"


/** Class that defines the direction of the Load with x, y and z
 *
 * $Revision: 44 $
 */
class Direction {
  
public:  
    /// default constructor: nothing is specified
    Direction() : x(0.0), y(0.0), z(0.0), xState(NOT_SPECIFIED), yState(NOT_SPECIFIED), zState(NOT_SPECIFIED), towardIndex(-1) {};
    /// constructor with initialization of the toward
    Direction(const unsigned int toward) { setToward(toward); };
    /// constructor with initialization of the 3 directions
    Direction(double x0, double y0, double z0) { setX(x0); setY(y0); setZ(z0); };
    /// copy constructor 
    Direction(const Direction & d) { x=d.x; xState=d.xState; y=d.y; yState=d.yState; z=d.z; zState=d.zState; towardIndex=d.towardIndex;};
    
    /// print to an ostream
    void xmlPrint(std::ostream & o) const {        
        o << "\t<direction ";
        if (isToward()) {
            o << "toward=\"" << towardIndex << "\"";
        }
        else {
            switch (xState) {
                case NOT_SPECIFIED:
                    break;
                case NULL_DIR:
                    o << "x=\"NULL\" ";
                    break;
                case SPECIFIED:
                    o << "x=\"" << x << "\" ";
                    break;
                default:
                    break;
            }
            switch (yState) {
                case NOT_SPECIFIED:
                    break;
                case NULL_DIR:
                    o << "y=\"NULL\" ";
                    break;
                case SPECIFIED:
                    o << "y=\"" << y << "\" ";
                    break;
                default:
                    break;
            }
            switch (zState) {
                case NOT_SPECIFIED:
                    break;
                case NULL_DIR:
                    o << "z=\"NULL\"";
                    break;
                case SPECIFIED:
                    o << "z=\"" << z << "\"";
                    break;
                default:
                    break;
            }
        }
        o << "/>" << std::endl;
    };

    /// set the direction
    void set(const double x, const double y, const double z) {
        setX(x);setY(y);setZ(z);
    };

    /// get the toward index
    int getToward() const {
        return towardIndex;
    };

    /// set the toward index
    void setToward(const unsigned int toward) {
        towardIndex=toward;
        xState = yState = zState = TOWARD;
    };

    /// true only if the direction is set by a toward atom
    bool isToward() const {
        return (towardIndex>=0 && xState==TOWARD && yState==TOWARD && zState==TOWARD);
    };

    ///@name X direction
    //@{
    
    /// get the x coordinate
    double getX() const { return x; };
    
    /// is the x coordinate NULL ?
    bool isXNull() const { return (xState==NULL_DIR); };
    
    /// is the x coordinate specified
    bool isXSpecified() const { return (xState==SPECIFIED); };
    
    /// set the x coordinate as NULL
    void setNullX() { x=0.0; xState = NULL_DIR; };
    
    /// set the x coordinate
    void setX(const double x) { this->x = x; xState = SPECIFIED; };
    //@}
        
    ///@name Y direction
    //@{
    /// get the y coordinate
    double getY() const { return y; };
    
    /// is the y coordinate NULL ?
    bool isYNull() const { return (yState==NULL_DIR); };
    
    /// is the y coordinate specified
    bool isYSpecified() const { return (yState==SPECIFIED); };
    
    /// set the y coordinate as NULL
    void setNullY() { y=0.0; yState = NULL_DIR; };
    
    /// set the y coordinate
    void setY(const double y) { this->y = y; yState = SPECIFIED; };
    
    //@}
        
    ///@name Z direction
    //@{
    /// get the z coordinate
    double getZ() const { return z; };
    
    /// is the z coordinate NULL ?
    bool isZNull() const { return (zState==NULL_DIR); };
    
    /// is the z coordinate specified
    bool isZSpecified() const { return (zState==SPECIFIED); };
    
    /// set the z coordinate as NULL
    void setNullZ() { z=0.0; zState = NULL_DIR; };
    
    /// set the z coordinate
    void setZ(const double z) { this->z = z; zState = SPECIFIED; };
    
    //@}
        
private:
    /// state of the x,y and z
    enum DirState {
        NOT_SPECIFIED,  //!< the direction has never been specified: it is absolutly free
        NULL_DIR,       //!< the direction has been specified to be always null
        SPECIFIED,      //!< the direction has been specified to be something imposed but not null (even 0.0 is possible!)
        TOWARD          //!< the direction is set dynamically depending on the "toward" position
    };

    /// x coordinates
    double x;
    /// y coordinates
    double y;
    /// z coordinates
    double z;
    /// state for the x coordinates
    DirState xState;
    /// state for the y coordinates
    DirState yState;
    /// state for the z coordinates
    DirState zState;
    /// toward atom index
    int towardIndex;
};

#endif //DIRECTION_H
