#!/usr/bin/python2.6
import Sofa
class CollisionPoints(Sofa.PythonScriptController):

	def initGraph(self, node):
		self.here = node;
		self.object2track = node.getObject('./Instrument/VisualModel/InstrumentVisualModel');
		self.omni = node.getObject('./Omni/omniDriver1');
		self.carving = node.getObject('./Opening/ctc');
		#print 'OK initGraph';
		return 0	
		
	def onBeginAnimationStep(self,dt):
		#print 'Recuperation valeur'
		tmp = self.object2track.position[0][0:3];
		#tmp = self.trk.position[0][0:3];
		#print 'valeur tmp';
		#print tmp;
		#print 'valeur trk';
		#print self.carving.findData('trackedPosition').value[0] ;
		#print 'test reassignation'
		#self.object2track.position[0][0:3] = tmp;
		#print 'valeur reassigne'
		#print self.object2track.position[0][0:3];
		self.carving.findData('trackedPosition').value = tmp[0:3];
		#print self.carving.findData('trackedPosition').value;	
		return 0;