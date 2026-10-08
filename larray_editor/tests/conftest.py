import os

# Qt tests must neither need a display (CI runs headless) nor show windows and
# dialogs while running, so we use the offscreen platform unless told otherwise
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
