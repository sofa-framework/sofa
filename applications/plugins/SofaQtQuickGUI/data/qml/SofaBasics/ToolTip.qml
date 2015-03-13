import QtQuick 2.0
import QtQuick.Controls 1.0
import "qrc:/SofaCommon/SofaCommonScript.js" as SofaCommonScript
import "qrc:/SofaCommon/SofaToolsScript.js" as SofaToolsScript

MouseArea {
    id: root
    hoverEnabled:true

    property string title: ""
    property string description: ""
    property int delay: 500
    property int xOffset: 5
    property int yOffset: 0

    onPressed: {d.release(); mouse.accepted = false;}

    onEntered: d.init();
    onExited: d.release();

    QtObject {
        id: d

        property Item toolTip

        function init() {
            if(0 === root.title.length && 0 === root.description.length)
                return;

            timer.start();
        }

        function release() {
            timer.stop();

            if(toolTip)
                toolTip.destroy();
        }
    }

    Timer {
        id: timer
        running: false
        repeat: false
        interval: root.delay
        onTriggered: {
            var contentItem = root;
            var pos = Qt.point(mouseX + root.xOffset, mouseY + root.yOffset);

            var window = SofaToolsScript.Tools.window(root);
            if(window && window.contentItem) {
                contentItem = window.contentItem;
                pos = root.mapToItem(contentItem, pos.x, pos.y);
            }

            d.toolTip = new SofaCommonScript.InstanciateComponent(toolTipComponent, contentItem, {x: pos.x, y: pos.y});

            // ensure the tooltip stays in the item window
            if(contentItem === window.contentItem) {
                if(d.toolTip.x + d.toolTip.width > contentItem.width)
                    d.toolTip.x -= d.toolTip.width + 2 * root.xOffset;

                if(d.toolTip.y + d.toolTip.height > contentItem.height)
                    d.toolTip.y -= d.toolTip.height + 2 * root.yOffset;
            }
        }
    }

    Component {
        id: toolTipComponent

        Rectangle {
            implicitWidth: column.implicitWidth
            implicitHeight: column.implicitHeight

            color: "lightgoldenrodyellow"
            radius: 5

            Column {
                id: column

                Text {
                    text: root.title
                    font.bold: true
                }

                Text {
                    text: root.description
                    //font.italic: true
                }
            }
        }
    }
}
