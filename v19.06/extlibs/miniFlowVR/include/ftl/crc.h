/******* COPYRIGHT ************************************************
*                                                                 *
*                             FlowVR                              *
*                       Template Library                          *
*                                                                 *
*-----------------------------------------------------------------*
* COPYRIGHT (C) 20054 by                                          *
* Laboratoire Informatique et Distribution (UMR5132) and          *
* INRIA Project MOVI. ALL RIGHTS RESERVED.                        *
*                                                                 *
* This source is covered by the GNU LGPL, please refer to the     *
* COPYING file for further information.                           *
*                                                                 *
*-----------------------------------------------------------------*
*                                                                 *
*  Original Contributors:                                         *
*    Jeremie Allard,                                              *
*    Clement Menier.                                              *
*                                                                 * 
*******************************************************************
*                                                                 *
* File: include/ftl/vec.h                                         *
*                                                                 *
* Contacts: 20/09/2005 Clement Menier <clement.menier.fr>         *
*                                                                 *
******************************************************************/
#ifndef FTL_CRC_H
#define FTL_CRC_H

namespace ftl
{

extern unsigned int crc32MsbToLsb(const void *buf, unsigned int size);

} // namespace ftl

#endif
