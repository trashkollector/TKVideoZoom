import cv2
import nodes
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as Transforms
import torch.nn.functional as F   
import numpy as np
import random
import comfy.utils

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
                "movement_type": (["zoom in", "zoom out","wobble","slide","spin","materialize","seesaw"],),
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
        
        pbar = comfy.utils.ProgressBar(frame_count) 
         
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
        elif (movement_type == "seesaw") :
            factor = 1.0
        elif (movement_type == "spin") :
            factor = 1.3
        else :
            factor = 1.0
           
        zoomW = width
        zoomH = height
        
        i=0
        frames=[]
        slideX=0
        slideY=0
        xOff=0
        yOff=0
        angle=0
        stopSpin=False
        alpha=0
        incSway=True
      
        print("Applying TK Effect "+movement_type+" "+movement_speed)
     
            
        for frame in video_numpy: 
            print(".",end="")
            if (i > frame_count-1) :
                break
                

            if pbar:
                pbar.update(i)
                    
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
            elif (movement_type == "spin") :
                cropImg = self.continuous_rotate(zoomFrame, angle)
            elif (movement_type == "seesaw") :
                cropImg = self.continuous_rotate(zoomFrame, angle)
            elif (movement_type =="materialize") :
                 cropImg = self.materialize(zoomFrame, alpha)
            elif (movement_type =="curl") :
                 cropImg = self.page_curl(zoomFrame, curl_radius, angle)
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
                if (movement_speed=="very slow")  :
                    factor = 1.1
                if (movement_speed=="slow")  :
                    factor = 1.2               
                if (movement_speed=="medium")  :
                    factor = 1.3               
                if (movement_speed=="fast")  :
                    factor = 1.4               
                if (movement_speed=="very fast")  :
                    factor = 1.5
                    
            i = i+1
            
            if movement_type=="spin" :
                # Increment the angle for the next frame
                if (angle >= 340) :
                    angle =0
                    stopSpin=True
                elif (not stopSpin) :
                    angle = (angle + (1 * speed)) % 360  
                    
            elif movement_type=="seesaw":
                if incSway :
                   angle +=  (1 * speed)
                else :  
                   angle -=  (1 * speed)
                if angle > 30 :
                    incSway=False
                elif angle < -30 :
                    incSway=True
                
            
            # materialize
            alpha += (0.01 * speed) # Adjust this value for faster/slower transition
            if (alpha >1.0) :
               alpha=1.0
            
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
        
    
    def continuous_rotate(self, img, angle):
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        
        # Create an alpha channel, fully opaque (255)
        alpha_channel = np.full((height, width), 255, dtype=np.uint8)

        # Stack the RGB and alpha channels to create RGBA
        imgalpha = np.dstack((img, alpha_channel))
            
        # Get the 2D rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0) # Angle, scale

        # Apply the affine transformation
        rotated_img = cv2.warpAffine(imgalpha, rotation_matrix, (width, height), borderMode=cv2.BORDER_TRANSPARENT)
  

        rimg = Image.fromarray(rotated_img)
        if rimg.mode != 'RGBA': 
            rimg = rimg.convert('RGBA')


        rimg = Image.fromarray(rotated_img)
        simg = Image.fromarray(img)
        if simg.mode != 'RGBA': 
            simg = simg.convert('RGBA')
        
        result_img = Image.alpha_composite(simg, rimg)  # bg, fg
        
        return result_img  







    # alpha 1.0 = solid, 0 = full transparent.
    def materialize(self, img, alpha) :
        height, width = img.shape[:2]
        bg = self.checkerboard(width, height)
        
         
        mat_img = cv2.addWeighted(img, alpha, bg, 1-alpha, 0)
        return mat_img


    def checkerboard(self, width, height) :
        checkerboard_img = np.zeros((height, width, 3), dtype=np.uint8)

        square_size = 10
        # Define colors (BGR format for OpenCV)
        color1 = (30, 30, 30) 
        color2 = (80, 80, 80)  

        # Loop through rows and columns to draw squares
        for row in range(height // square_size):
            for col in range(width // square_size):
                # Determine which color to use based on row and column parity
                if (row + col) % 2 == 0:
                    color = color1
                else:
                    color = color2

                # Calculate coordinates for the current square
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size

                # Draw the rectangle (square) on the image
                checkerboard_img = cv2.rectangle(checkerboard_img, (x1, y1), (x2, y2), color, -1) # -1 fills the rectangle
        
        
        return checkerboard_img
        
   


            

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "TKVideoZoom": TKVideoZoom
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TKVideoZoom": "TKVideoZoom"
}
