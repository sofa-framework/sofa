import QtQuick 2.0
import QtQuick.Controls 1.0
import QtQuick.Controls.Styles 1.3
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.1
import Qt.labs.settings 1.0
import SofaBasics 1.0
import SofaInteractor 1.0
import "qrc:/SofaCommon/SofaCommonScript.js" as SofaCommonScript

QtObject {
    id: root

    // private
    QtObject {
        id: d

        property Scene scene: Scene {}
        //property Component interactor: UserInteractor {}

        property SceneListModel sceneListModel: null
    }

    readonly property Scene scene: d.scene
    //readonly property Component interactor: defaultInteractor
    //readonly property Component defaultInteractor: UserInteractor_Selection

    // create a sceneListModel only if needed
    function sceneListModel() {
        if(null === d.sceneListModel)
            d.sceneListModel = SofaCommonScript.InstanciateComponent(SceneListModel, root);

        return d.sceneListModel;
    }

    //property UISettings uiSettings: UISettings {}
    //property RecentSettings uiSettings: RecentSettings {}
}
