from . import zoomcontrols

NODE_CLASS_MAPPINGS = {
    "TKVideoZoom": zoomcontrols.TKVideoZoom,
    "TKVideoSpeedZones": zoomcontrols.TKVideoSpeedZones,
    "TKVideoSmoothLooper": zoomcontrols.TKVideoSmoothLooper,
    "TKVideoStitcher": zoomcontrols.TKVideoStitcher,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
     "TKVideoZoom": "Various Video Effects, Zoom,Slide,Spin,etc.",
     "TKVideoSpeedZones": "Slow down/speed up video in parts of video.",
     "TKVideoSmoothLooper": "Loop a video .",
     "TKVideoStitcher": "Stitch video clips together with transition.",

}