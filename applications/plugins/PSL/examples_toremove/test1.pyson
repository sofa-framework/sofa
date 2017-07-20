Node : {
        name : "myScene"

        Node : {
            name : "ItsNamed"
            dt : "2.0"

            Node : {
                name : "ItsAChild"


                MechanicalObject : {
                    name : "test"
                }

                Python : '''
                print("================"+str(ItsAChild.name)+"==============")
                for i in range(0,10):
                                myScene.createChild("NAME "+str(i))
                 '''
            }
        }


        //Node : {
        //        name : "childNode1"
        //        MechanicalObject : {
        //            name : "mecha"
        //            position : "1 2 3 4 5 6"
        //        }
        //}

        ///////////////////////////// DYNAMIC ///////////////////////////////////////////////////
        //Python : '''print("Hello pyhon")'''


        ///////////////////////////// STACK BASED ///////////////////////////////////////////////////
        Node : {
            name : "childNode2"
            Python : '''
                    print("The name of this Node is: {"+name+"}")
                    print("The name of this Node is: {"+self.name+"}")
                    print("The name of this Node is: {"+root.name+"}")
                    Sofa.msg_warning(childNode2, "Salut Root... c'est "+self.name+" qui te parle")
            '''

            OglModel : {}
        }

        ////////////////////////// THIS IS A COMMENT ///////////////////////////////////////////
        //Node : {
        //        name : "childNode3"
        //
        //        Python : '''
        //                print("The name of this Node is: {"+self.name+"}")
        //                '''
        //        VisualModel : {
        //                name : "visual1"
        //                filename : p"os.getcwd()+'/yolo.txt'"
        //        }
        //}

        //Node : {
        //    name : "child2"
        //    Python : '''
        //            for i in range(0, 2):
        //                self.createChild("NAME_"+str(i))
        //            '''
        //}

        //OneMore : {}
}

