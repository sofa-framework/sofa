#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa

class BoxROI:

    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax, PointCoords):
    	self.xmin = xmin
    	self.xmax = xmax
    	self.ymin = ymin
    	self.ymax = ymax
	self.zmin = zmin
	self.zmax = zmax
	self.PointCoords = PointCoords

    def isTetraInBoxROI(self, Tetra):

	c1 = self.PointCoords[Tetra[0]]
	c2 = self.PointCoords[Tetra[1]]
	c3 = self.PointCoords[Tetra[2]]
	c4 = self.PointCoords[Tetra[2]]
	CenterX = (c1[0]+c2[0]+c3[0]+c4[0])/4
	CenterY = (c1[1]+c2[1]+c3[1]+c4[1])/4
	CenterZ = (c1[2]+c2[2]+c3[2]+c4[2])/4
	if CenterX > self.xmin and CenterX < self.xmax:
		if CenterY > self.ymin and CenterY < self.ymax:
			if CenterZ > self.zmin and CenterZ < self.zmax:
				return True
	return False

    def calcTetrasInROI(self, Tetras):
	NTetras = len(Tetras)
	self.Inliers = []
	self.Outliers = []
	for i in range(0,NTetras):
		if self.isTetraInBoxROI(Tetras[i]):
			self.Inliers.append(Tetras[i])
		else:
			self.Outliers.append(Tetras[i])	
			
			
    def calcUnion(self,Tetras1,Tetras2):
        Union = []
        while len(Tetras1)!=0:
            Union = Union + Tetras1[0]      
            DelIdxs = []
            for i in range(0,len(Tetras2)):                
                if Tetras1[0] == Tetras2[i]:                   
                    DelIdxs = DelIdxs + [i]                    
            for j in range(0, len(DelIdxs)):
                del(Tetras2[DelIdxs[-j]])
            del(Tetras1[0])
        Union = Union+Tetras2           
                    
        return Union    

	

        



