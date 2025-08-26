from . import zoomcontrols

NODE_CLASS_MAPPINGS = {
    "TKVideoZoom": zoomcontrols.TKVideoZoom,
    "TKVideoSpeedZones": zoomcontrols.TKVideoSpeedZones,
    "TKVideoSmoothLooper": zoomcontrols.TKVideoSmoothLooper,
    "TKVideoStitcher": zoomcontrols.TKVideoStitcher,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
     "TKVideoZoom": "Video Effects",
     "TKVideoSpeedZones": "Video Speed Adjuster",
     "TKVideoSmoothLooper": "Loop a video .",
     "TKVideoStitcher": "Video Stitcher.",

}