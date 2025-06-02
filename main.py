import os
import sys
import signal
from pathlib import Path
from PySide6.QtGui import QGuiApplication, Qt
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QCoreApplication, QTimer
from app.grader_app import GraderApp

# Force Fusion style
QCoreApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"

def handle_sigint(signum, frame):
    print("Exiting via Ctrl+C")
    QCoreApplication.quit()

if __name__ == "__main__":
    # Register SIGINT handler
    signal.signal(signal.SIGINT, handle_sigint)

    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    graderApp = GraderApp()

    engine.rootContext().setContextProperty("graderApp", graderApp)

    qml_file = os.path.join(Path(__file__).resolve().parent, "UI", "main.qml")
    engine.load(qml_file)

    if not engine.rootObjects():
        sys.exit(-1)

    # Trick to keep signal handling alive
    timer = QTimer()
    timer.start(100)
    timer.timeout.connect(lambda: None)  # Dummy no-op to keep the event loop alive

    sys.exit(app.exec())
