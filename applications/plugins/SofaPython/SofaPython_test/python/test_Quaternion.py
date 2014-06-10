import SofaPython.Quaternion as Quaternion

from SofaTest.Macro import *

def test_inv(q):
    q_inv=Quaternion.inv(q)
    print "inv_ok"
    print q, q_inv
    return EXPECT_VEC_EQ(Quaternion.id(), Quaternion.prod(q,q_inv), "test_inv")

def run():
    ok=True
    qi= Quaternion.id()
    ok &= EXPECT_VEC_EQ([0,0,0,1],qi)
    q1=[0,0.5,0,1]
    ok &= test_inv(q1)
    return ok

