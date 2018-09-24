import Sofa
import array
import timeit

def time(root):
        pr = cProfile.Profile()
        pr.enable()
        
        for i in range(0, 10000):
                pass        
        
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

def createScene(root):
        root.createObject("MechanicalObject", 
                          name="mstate", position=[[1.0,1.1,1.2,],[2.0,2.1,2.2]]*10)
        
        print(str(root.mstate.position))
        print(repr(root.mstate.position))
        
        print(str(root.mstate.showColor))
        print(repr(root.mstate.showColor))
        
        print(str(root.mstate.name))
        print(repr(root.mstate.name))

        print("========= VIEW on BUFFER =============")
        view = memoryview(root.mstate.position) 
        print("itemsize: "+str(view.itemsize))
        print("shape: "+str(view.shape))
        print("ndim: "+str(view.ndim))
        print("strides: "+str(view.strides))
        print("nbytes: "+str(view.nbytes))
        print(view[1,0])
        print(view.tolist())

        print("========= VIEW on COLOR =============")
        view = memoryview(root.mstate.showColor) 
        print("itemsize: "+str(view.itemsize))
        print("shape: "+str(view.shape))
        print("ndim: "+str(view.ndim))
        print("strides: "+str(view.strides))
        print("nbytes: "+str(view.nbytes))
        print(view.tolist())

        p = memoryview(root.mstate.position)

        print("get_object: " + str( timeit.timeit("str(root.mstate.position)", globals={"root":root}) ))
        print("get_object_data: " + str( timeit.timeit("str(memoryview(root.mstate.position).tolist())", globals={"root":root}) ))
        
        l = p.tolist()
        
        print("get_object: " + str( timeit.timeit("root.mstate", globals={"root":root}) ))
        print("get_object_data: " + str( timeit.timeit("root.mstate.position", globals={"root":root}) ))
        print("memoryview(get_object_data): " + str( timeit.timeit("memoryview(root.mstate.position)", globals={"root":root}) ))
        print("memoryview::tolist: " + str( timeit.timeit("p.tolist()", globals={"p":p}) ))
        print("listacess: " + str( timeit.timeit("l[5][0]", globals={"l":l}) ))
        print("memoryview::__getitem__: " + str( timeit.timeit("p[5,0]", globals={"p":p}) ))
        
        print("FIN")
        
