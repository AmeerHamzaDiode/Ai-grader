import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    visible: true
    id: appWindow
    width: 1920
    height: 1080

    Rectangle {
        id: colorButton
        width: appWindow.width / 2
        height: appWindow.height / 2
        anchors.centerIn: parent
        radius: appWindow.height / 16
        color: "steelblue"

        Text {
            id: messageText
            anchors.centerIn: parent
            text: "Press Me"
            color: "white"
            font.pixelSize: 30
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                console.log("calling slot")
                graderApp.toggle_state()
            }
        }
    }

    Connections {
        target: graderApp
        function onMessageChanged(msg) {
            messageText.text = msg
            colorButton.color = (msg === "Button Pressed!") ? "orange" : "steelblue"
        }
    }
}