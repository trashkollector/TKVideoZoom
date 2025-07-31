import cv2
import nodes
import torch
from PIL import Image
import torchvision.transforms as Transforms
import torch.nn.functional as F   
import numpy as np
import random

#  TK Collector - Custom Node for Video Zooming
#  July 31, 2025
#  https://github.com/trashkollector/TKVideoZoom
    
class TKVideoZoom:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "frame_count":  ("INT",),
                "movement_speed": (["very slow", "slow ","medium", "fast", "very fast"],),
                "movement_type": (["zoom in", "zoom out","wobble","slide"],),
                "up_dn": (["top", "bottom","center"],),
                "left_right": (["left", "right","center"],),

            },
        }

    RETURN_TYPES = ("IMAGE","AUDIO")
    RETURN_NAMES = ("image","audio")

    FUNCTION = "tkvideozoom"

    #OUTPUT_NODE = False

    CATEGORY = "TKVideoZoom"

    # Assume 'video_tensor' is your video data as a PyTorch tensor
    # Format: [T, H, W, C] (Time, Height, Width, Channels)
    # Data type should be uint8, with values in the range [0, 255]
    #  number of channels (e.g., 3 for RGB)
    # To represent a video, NumPy extends this concept by adding an extra dimension for the frames.
    #    This results in a 4D array with the shape (frames, height, width, channels).
    
    def tkvideozoom(self, image, audio, frame_count, up_dn, left_right, movement_speed, movement_type):
        tensor_shape = image.shape
        height = tensor_shape[1]
        width = tensor_shape[2]
        
        video_numpy = (image.numpy() * 255).astype(np.uint8)  #convert video
   
        speed=1.0
        if (movement_speed=="very slow") :
            speed=1.0
        elif (movement_speed=="slow") :
            speed=2.0
        elif (movement_speed=="medium") :
            speed=5.0        
        elif (movement_speed=="fast") :
            speed=10.0      
        elif (movement_speed=="very fast") :
            speed=20.0   


        if (movement_type == "zoom in") :
            factor = 1.0
        elif  (movement_type == "zoom out") :
            factor = 2.0
        elif (movement_type == "wobble") :
            factor = 1.1
            increment = True
        elif (movement_type == "slide") :
            factor = 2.0
            
        i=0
        nW = width
        nH = height
        frames=[]
        slideX=0
        slideY=0
        xOff=0
     
        for frame in video_numpy: 
            if (i >= frame_count-1) :
                break
            print("frame "+str(i))
            
            nW = int(width * factor)
            nH = int(height * factor)

            # width, height
            zoomFrame = cv2.resize(frame, (nW, nH), interpolation=cv2.INTER_AREA)
            
          
            diffH = int((nH - height)/2)
            diffW = int((nW - width)/2)
            
            if up_dn == "top" :
                diffH =0
            elif up_dn == "bottom" :
                diffH = nH - height
                
                
            if left_right == "left" :
                diffW =0   
            elif left_right == "right" :
                diffW = nW - width  
            
            # only side to side
            if (movement_type == "slide") :
                if (left_right == "left") or (left_right=="center")  :
                    slideX = -1
                    slideY = 0
                    diffW = nW - width                    
                elif left_right == "right" :
                    slideX = 1
                    slideY =0
                    diffW = 0 


                              
            xTemp =  diffW + int((i * slideX) * speed) 
            if (xTemp>0) and (xTemp + width < nW) :
               xOff = xTemp
            
            # start y, end y, start x , end x --- CROP
            if (movement_type == "slide") :
                cropImg = zoomFrame[ diffH:height+diffH,    xOff:width+xOff :]
            # all other movments    
            else :
                cropImg = zoomFrame[ diffH:height+diffH,   diffW:width+diffW, :]
                
            frame = cropImg
            frames.append(frame)
            
            
            if (movement_type == "zoom in") :
                factor = factor + (0.001 * speed)
                if factor >= 2.0 :
                   factor =2.0
            elif (movement_type == "zoom out") :
                factor = factor - (0.001 * speed)
                if (factor <= 1.0) :
                    factor = 1.0
            elif  (movement_type == "wobble") :
                if increment:
                    factor = factor + (0.001)
                else :
                    factor = factor - (0.001)
                if (increment and factor > 1.109) :
                    increment = False
                elif (not increment and factor <= 1.001) :
                    increment = True
            elif (movement_type =="slide") :
                factor = 2.0   #  static always
                    
            i = i+1
            
            
            
        # Convert to tensor
        numpy_array = np.array(frames)
        theTensor = torch.from_numpy(numpy_array)
        theTensor = theTensor.float()  / 255.0
        print(theTensor.shape)
            
        return (theTensor,audio)
        
        
           

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# Add custom API routes, using router
from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get("/tkvideozoom")
async def get_tkzoomcontrols(request):
    return web.json_response("tkvideozoom")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TKVideoZoom": TKVideoZoom
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TKVideoZoom": "TKVideoZoom"
}
