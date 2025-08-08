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
                "visual_effect": (["normal", "grayscale","sepia"],),

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
    
    def tkvideozoom(self, image, audio, frame_count, up_dn, left_right, movement_speed, movement_type, visual_effect):
        tensor_shape = image.shape
        height = tensor_shape[1]
        width = tensor_shape[2]
        
        # Generate a random waveform function along the y-axis - used for wobble
        y_trans = self.generate_translation(frame_count, 16, tuple([0.5, 2.5]), tuple([5.0,10.0]))
        x_trans = self.generate_translation(frame_count, 16, tuple([0.0, 0.8]), tuple([1.0,5.0]))
              
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
        zoomW = width
        zoomH = height
        frames=[]
        slideX=0
        slideY=0
        xOff=0
        yOff=0
     
        print("Applying TK Effect")
     
        for frame in video_numpy: 
            if (i >= frame_count-1) :
                break
            
            zoomW = int(width * factor)
            zoomH = int(height * factor)

            # width, height
            zoomFrame = cv2.resize(frame, (zoomW, zoomH), interpolation=cv2.INTER_AREA)
            
          
            diffH = int((zoomH - height)/2)
            diffW = int((zoomW - width)/2)
            
            if up_dn == "top" :
                diffH =0
            elif up_dn == "bottom" :
                diffH = zoomH - height
                
                
            if left_right == "left" :
                diffW =0   
            elif left_right == "right" :
                diffW = zoomW - width  
            
            # only side to side
            if (movement_type == "slide") :
                if (left_right == "left") or (left_right=="center")  :
                    slideX = -1
                    slideY = 0
                    diffW = zoomW - width                    
                elif left_right == "right" :
                    slideX = 1
                    slideY =0
                    diffW = 0 
                          
                xTemp =  diffW + int((i * slideX) * speed) 
                if (xTemp>0) and (xTemp + width < zoomW) :
                   xOff = xTemp
                   
            elif (movement_type == "wobble") :
                xTemp =  diffW + int( x_trans[i] * speed)
                if (xTemp>0) and (xTemp + width < zoomW ) :
                   xOff = xTemp
                yTemp =  diffH + int( y_trans[i] * speed) 
                if (yTemp>0) and (yTemp + height < zoomH) :
                   yOff = yTemp
               

            # start y, end y, start x , end x --- CROP
            if (movement_type == "slide") :
                cropImg = zoomFrame[ diffH:height+diffH,    xOff:width+xOff :]
            # all other movments    
            elif (movement_type=="wobble") :
                cropImg = zoomFrame[ yOff:height+yOff,    xOff:width+xOff :]
            else :
                cropImg = zoomFrame[ diffH:height+diffH,   diffW:width+diffW, :]
                
                
            if (visual_effect == "grayscale") :
                cropImg = np.array( self.grayscale(cropImg))
            elif (visual_effect =="sepia") :
                cropImg = np.array( self.sepia(cropImg))

                
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
                factor = 1.3
            elif (movement_type =="slide") :
                if (speed=="slow") or (speed =="very slow") :
                    factor = 1.3
                else :
                    factor = 1.7   #  static always
                    
            i = i+1
            
            
            
        # Convert to tensor
        numpy_array = np.array(frames)
        theTensor = torch.from_numpy(numpy_array)
        theTensor = theTensor.float()  / 255.0
        
        print(theTensor.shape)
            
        return (theTensor,audio)
        
        
    def generate_translation(
        self,
        n_frames: int, 
        fps: float, 
        amplitudes: tuple[float, float],
        frequencies: tuple[float, float]
    ) -> list:
        """
        Generate a list of translation values to create a hand shaking effect along one 
        axis in an 80-second video.

        Parameters:
            n_frames (int): The total number of frames in the video.
            fps (float): Frames per second of the video.
            amplitudes (tuple[float, float]): The amplitudes of the sinusoidal waves 
                                              along y-axis.
            frequencies (tuple[float, float]): The frequencies of the sinusoidal waves 
                                               along the x-axis.

        Returns:
            list: A list of translation values for each frame to create the hand shaking
                  effect.
        """
        

        num_points = 2000
        num_waves = np.random.randint(30, 40)
        amplitude_min, amplitude_max = amplitudes
        frequence_min, frequence_max = frequencies
        
        x = np.linspace(0, 10, num_points)
        y = np.zeros_like(x)
        
        for _ in range(num_waves):
            frequency = np.random.uniform(frequence_min, frequence_max)
            amplitude = np.random.uniform(amplitude_min, amplitude_max)
            phase_shift = np.random.uniform(0, 2*np.pi)
            y += amplitude * np.sin(2*np.pi*frequency*x + phase_shift)

        duration = n_frames / fps
        fixed = int(num_points * duration / 80)

        x_fixed, y_fixed = x[:fixed], y[:fixed]

        x_interpolated = np.linspace(0, max(x_fixed), n_frames)
        y_interpolated = np.interp(x_interpolated, x_fixed, y_fixed)

              
        return y_interpolated    

    
    def sepia(self, npImg):

        filt = cv2.transform( npImg, np.matrix([[ 0.393, 0.769, 0.189],
                                              [ 0.349, 0.686, 0.168],
                                              [ 0.272, 0.534, 0.131]
        ]) )

        # Check wich entries have a value greather than 255 and set it to 255
        filt[np.where(filt>255)] = 255

        # Create an image from the array
        return filt
            
        
    def grayscale(self, src_image):
        gray = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
        normalized_gray = np.array(gray, np.float32)/255
        #solid color
        sepia = np.ones(src_image.shape)
        sepia[:,:,0] *= 200 #B
        sepia[:,:,1] *= 200 #G
        sepia[:,:,2] *= 200 #R
        #hadamard
        sepia[:,:,0] *= normalized_gray #B
        sepia[:,:,1] *= normalized_gray #G
        sepia[:,:,2] *= normalized_gray #R
        return np.array(sepia, np.uint8)        
        
        
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
