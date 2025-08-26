import cv2
import nodes
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as Transforms
import torch.nn.functional as F   
import numpy as np
import random
import comfy.utils
import math


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
                "movement_speed": (["very slow", "slow","medium", "fast", "very fast"],),
                "movement_type": (["zoom in", "zoom out","zoom both","camera shake","slide","spin","materialize","seesaw","wobble"],),
                "up_dn": (["top", "bottom","center"],),
                "left_right": (["left", "right","center"],),
                "visual_effect": (["normal", "grayscale","sepia","noise"],),

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
   
        # SPEED
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

        # ZOOM
        zfactor = self.getZoomFactor(movement_type, movement_speed)
           
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
      
        print("Applying TK Effect "+movement_type+" "+movement_speed+" "+str(zfactor))
     
            
        for frame in video_numpy: 
            print(".",end="")

            if pbar:
                pbar.update(i)
                    
            zoomW = int(width * zfactor)
            zoomH = int(height * zfactor)

            # width, height -  ZOOM IMAGE
            zoomFrame = cv2.resize(frame, (zoomW, zoomH), interpolation=cv2.INTER_AREA)
            
          
            # CENTER
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
                    diffW = zoomW - width                    
                elif left_right == "right" :
                    slideX = 1
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

                
            # color change  
            if (visual_effect == "grayscale") :
                zoomFrame = np.array( self.grayscale(zoomFrame))
            elif (visual_effect =="sepia") :
                zoomFrame = np.array( self.sepia(zoomFrame))
            elif (visual_effect == "noise") :
                ran = i % (int(100 / (1+(speed/3))))
                if (ran >0 ) and (ran < 10) :
                   zoomFrame = np.array(self.noise(zoomFrame))

            # start y, end y, start x , end x --- CROP
            if (movement_type == "slide") :
                cropImg = zoomFrame[ diffH:height+diffH,    xOff:width+xOff :]
  
            elif (movement_type=="wobble") :
                cropImg = zoomFrame[ yOff:height+yOff,    xOff:width+xOff :]
                
            elif (movement_type == "spin") :
                cropImg = self.continuous_rotate(zoomFrame, angle)
                
            elif (movement_type == "seesaw") :
                zoomFrame = self.continuous_rotate(zoomFrame, angle)
                zoomnp = np.array(zoomFrame)
                cropImg = zoomnp[ diffH:height+diffH,    diffW:width+diffW :]
                
            elif (movement_type =="materialize") :
                 cropImg = self.materialize(zoomFrame, alpha)
            elif (movement_type=="camera shake") :
                freq=0.5
                if (movement_speed=="slow" ) :
                    freq=0.7
                elif (movement_speed=="medium" ) :
                    freq=1.0                   
                elif (movement_speed=="fast" ) :
                    freq=1.3
                elif ( movement_speed=="very fast") :
                    freq=2.0                 
                cropImg = self.smooth_camera_shake(zoomFrame, 30, i, 15, freq)
            else :
                cropImg = zoomFrame[ diffH:height+diffH,   diffW:width+diffW, :]
                
   
                
            frame = cropImg
            frames.append(frame)
            
            
            if (movement_type == "zoom in") or (movement_type == "zoom both" and incSway==True) :
                zfactor += (0.001 * speed)
                if zfactor >= 2.0 :
                   zfactor =2.0
            elif (movement_type == "zoom out") or (movement_type=="zoom both" and incSway==False)            :
                zfactor -= (0.001 * speed)
                if (zfactor <= 1.0) :
                    zfactor = 1.0
            
                    
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
                   angle +=  (1 * (speed / 10))
                else :  
                   angle -=  (1 * (speed /10))
                if angle > 30 :
                    incSway=False
                elif angle < -30 :
                    incSway=True
            elif movement_type=="zoom both" :
                if incSway :
                   if zfactor > 1.2:
                      incSway = False
                else :
                   if (zfactor <= 1.01) :
                      incSway = True
                       
            
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
    
    def getZoomFactor(self, movement_type, movement_speed) :
        
        zfactor=1.0
        
        if (movement_type == "zoom in") :
            zfactor = 1.0
        elif  (movement_type == "zoom out") :
            zfactor = 2.0
        elif  (movement_type == "zoom both") :
            zfactor = 1.1
        elif (movement_type == "wobble") :
            zfactor = 1.2
        elif (movement_type == "camera shake") :
            zfactor = 1.0
        elif (movement_type == "slide") :
            if (movement_speed=="very slow")  :
                zfactor = 1.1
            elif (movement_speed=="slow")  :
                zfactor = 1.2               
            elif (movement_speed=="medium")  :
                zfactor = 1.3               
            elif (movement_speed=="fast")  :
                zfactor = 1.4               
            elif (movement_speed=="very fast")  :
                zfactor = 1.5            
        elif (movement_type == "seesaw") :
            zfactor = 1.4
        elif (movement_type == "spin") :
            zfactor = 1.3
        else :
            zfactor = 1.0
            
        return zfactor
        
    def smooth_camera_shake(self, img, fps,frame_idx, max_shift=15, frequency=3):
        """
        Adds a camera shake effect by randomly shifting frames.
        
        Args:
            input_path (str): path to input video
            output_path (str): path to save shaken video
            max_shift (int): maximum pixel shift for shake
        """  
        height, width = img.shape[:2]


  
        t = frame_idx / fps  # time in seconds

        # Smooth sine-wave shake (x and y with phase offset)
        dx = int(max_shift * math.sin(2 * math.pi * frequency * t))
        dy = int(max_shift * math.sin(2 * math.pi * frequency * t + math.pi/2))

        # Apply transformation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shaken_frame = cv2.warpAffine(img, M, (width, height))

        # Optional: zoom slightly to hide black edges
        zoom = 1.05
        zoom_w, zoom_h = int(width * zoom), int(height * zoom)
        resized = cv2.resize(shaken_frame, (zoom_w, zoom_h))

        # Crop back to original size
        x_start = (zoom_w - width) // 2
        y_start = (zoom_h - height) // 2
        cropped = resized[y_start:y_start+height, x_start:x_start+width]

        return cropped    
        
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
            
        
    def noise (self, src_image) :
        mean = 0
        variance = 2000 # Adjust for desired noise intensity
        sigma = variance**0.5 # Standard deviation

        # Generate Gaussian noise with the same shape as the image
        # For grayscale: noise = np.random.normal(mean, sigma, img_gray.shape)
        # For color:
        noise = np.random.normal(mean, sigma, src_image.shape)

        # Add noise to the image
        noisy_img = src_image + noise

        # Clip pixel values to the valid range (0-255) and convert to uint8
        src_image = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return src_image
        
        
        
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
        
   







class TKVideoSpeedZones:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "image": ("IMAGE",),
                "frame_count":  ("INT",{"default" : "81"}),
                "fps":  ("FLOAT",{"default" : "16"}),
                "new_fps": (["30","60"],),
               
                "speed_zone1": (["very slow", "slow","normal", "fast","very fast"],),
                "speed_zone2": (["very slow", "slow","normal", "fast","very fast"],),
                "speed_zone3": (["very slow", "slow","normal", "fast","very fast"],),
                "speed_zone4": (["very slow", "slow","normal", "fast","very fast"],),
          
                

            },
        }



    RETURN_TYPES = ("IMAGE","FLOAT","INT")
    RETURN_NAMES = ("image","new_fps","new_frame_count")


    FUNCTION = "tkvideospeedzones"

    CATEGORY = "TKVideoZoom"

    # Assume 'video_tensor' is your video data as a PyTorch tensor
    # Format: [T, H, W, C] (Time, Height, Width, Channels)
    # Data type should be uint8, with values in the range [0, 255]
    #  number of channels (e.g., 3 for RGB)
    # To represent a video, NumPy extends this concept by adding an extra dimension for the frames.
    #    This results in a 4D array with the shape (frames, height, width, channels).
    
    def tkvideospeedzones(self, image , fps, new_fps, frame_count, speed_zone1, speed_zone2, speed_zone3, speed_zone4):
        
        #image = tensor
 
       
        pbar = comfy.utils.ProgressBar(frame_count) 
        totalSeconds = float(frame_count) /fps
        
        
        tensor_shape = image.shape
        height = tensor_shape[1]
        width = tensor_shape[2]
                    
        video_numpy = (image.numpy() * 255).astype(np.uint8)  #convert video
      
        i=0
        frames=[]
        numFrame=0
      
        print("Applying TK Speed ")
     
        prevframe=None    
        
        for frame in video_numpy: 
            print(".",end="")
            if (i > frame_count-1) :
                break
 
            if pbar:
                pbar.update(i)
                
            if (i == 0) :
                frames.append(frame)
                numFrame +=1
            
            if (i > 0) :
               ninserts=1.0
               
               # first section
               if (i < int(float(frame_count) * 0.25)) :
                   ninserts = self.calcInterpolatesForWarp(speed_zone1,fps, float(new_fps))
                   
               # second section
               elif (i >= int(float(frame_count) * 0.25)) and  (i < int(float(frame_count) * 0.5)) : 
                   ninserts = self.calcInterpolatesForWarp(speed_zone2,fps,float(new_fps))
                   
               # 3rd section
               elif (i >= int(float(frame_count) * 0.5)) and  (i < int(float(frame_count) * 0.75)) : 
                   ninserts = self.calcInterpolatesForWarp(speed_zone3,fps, float(new_fps))
                   
               # 4th section
               elif (i >= int(float(frame_count) * 0.75)) : 
                   ninserts = self.calcInterpolatesForWarp(speed_zone4,fps, float(new_fps))

                 
               if (ninserts > 0 and prevframe is not None) : 
                   interopArr = self.interpolate_images_linear(prevframe, frame, int(ninserts))
                   for interop in interopArr :           
                       frames.append( interop)
                       numFrame += 1
               
               if (ninserts < 0) :  
                  ninserts = int(ninserts * -1.0)
                  if i % ninserts == 0 :
                      frames.append(frame)
                      numFrame += 1
                     
               else :                   
                  frames.append(frame)
                  numFrame += 1
            
            prevframe = frames[-1]
            i +=1
            
        # Convert to tensor
        numpy_array = np.array(frames)
        theTensor = torch.from_numpy(numpy_array)
        theTensor = theTensor.float()  / 255.0
        
        print(theTensor.shape)
        
        return (theTensor , new_fps, numFrame)
        
        


    def interpolate_images_linear(self, frame1, frame2, numInserts):
        """
        Performs linear interpolation between two frames.

        Args:
            frame1 (np.ndarray): The first frame (e.g., image data).
            frame2 (np.ndarray): The second frame (e.g., image data).
            alpha (float): The interpolation factor, a value between 0.0 and 1.0.
                           0.0 corresponds to frame1, 1.0 corresponds to frame2.

        Returns:
            np.ndarray: The interpolated frame.

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")
        """    
            
        interpolated_frames = []
        for i in range(int(numInserts)):
            alpha = i / (numInserts+1) 

            
            interpolated_frame = (1 - alpha) * frame1 + alpha * frame2
            interpolated_frames.append(interpolated_frame)

        return interpolated_frames
        
        
        
    def calcInterpolatesForWarp(self, speedStr, fps, new_fps) :
        
        ratio = new_fps / fps
        
        if (speedStr=="slow") :
            return 2.0   * ratio
        elif (speedStr=="very slow") :
            return 4.0   * ratio
        elif (speedStr =="normal") :
            return ratio
        elif (speedStr =="fast") : 
            return ratio / 2.0
        elif (speedStr=="very fast") :
            return ratio / 4.0
 
      
class TKVideoSmoothLooper:
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "image": ("IMAGE",),
                "audio":  ("AUDIO",),
                "fps":  ("FLOAT",{"default" : "16", "minimum" :"8", "maximum":"60"}),   
                "loop_type": (["ping pong", "cross fade", "fade"],),

            },
        }



    RETURN_TYPES = ("IMAGE","AUDIO","INT")
    RETURN_NAMES = ("image","audio","new_frame_count")


    FUNCTION = "tkvideosmoothlooper"

    CATEGORY = "TKVideoZoom"
    
    
    def tkvideosmoothlooper(self, image, audio, loop_type, fps):
        
        #image = tensor
        
        tensor_shape = image.shape
        frame_count = tensor_shape[0]
        height = tensor_shape[1]
        width = tensor_shape[2]
        
        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        pbar = comfy.utils.ProgressBar(frame_count) 
                    
        video_numpy = (image.numpy() * 255).astype(np.uint8)  #convert video
        
      
        frames=[]
        ncross = int(fps/2.0)
        numFrame=0
   
     
        print("Applying TK Looper ")
 
        i=0
        for frame in video_numpy: 
            print(".",end="")

            if pbar:
                pbar.update(i)
      
          
            f=None
            alpha=1
            if ( i <= ncross)  :
                alpha = i / ncross
                    
                if loop_type=="cross fade" :
                   f = self.crossfade(frame, video_numpy[frame_count-1-i],1- alpha)
                elif loop_type=="fade" :
                   f = self.crossfade(frame, black_image, 1-alpha)
                elif loop_type=="ping pong" :
                   f = frame
                   frames.append(f)
                   numFrame +=1 
                   
            elif ( i > frame_count - ncross) :
                alpha = (frame_count - i) / ncross
                
                
                if loop_type=="cross fade" :
                   f = self.crossfade(frame, video_numpy[i], 1-alpha)
                elif loop_type =="fade" :
                   f = self.crossfade(frame, black_image, 1-alpha) 
                elif loop_type=="ping pong":
                   f = frame    
                   frames.append(f)
                   numFrame +=1                   
                   
            else :
                f= frame

            frames.append(f)
            numFrame +=1  
          
                
          
            i += 1
            
        revarr =[]
        if (loop_type=="ping pong"):
            
            for frame in reversed(frames) :
                 revarr.append(frame)
            
            frames = frames+ revarr
            numFrame *= 2
             
            
        # Convert to tensor
        numpy_array = np.array(frames)
        theTensor = torch.from_numpy(numpy_array)
        theTensor = theTensor.float()  / 255.0
        
        print(theTensor.shape)
        
        return (theTensor , audio, numFrame)
    
 
        
    def crossfade(self, frame1, frame2, alpha):
    
        interpolated_frame = (1 - alpha) * frame1 + alpha * frame2
          
        return interpolated_frame
        
    
    
    
    
class TKVideoStitcher :
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "targetWidth": ("INT", {"default": "500"}),
                "targetHeight": ("INT", {"default": "500"}),

                "image1":  ("IMAGE", ),     
                "image2":  ("IMAGE", ), 
                
                },
            "optional": {
                "audio": ("AUDIO", ),   
                
                "image3":  ("IMAGE", ),   
                "image4":  ("IMAGE", ),        
                }                
            }
        

    RETURN_TYPES = ("IMAGE",  "AUDIO",)
    RETURN_NAMES = ("image","audio",)


    FUNCTION = "tkvideostitcher"

    #OUTPUT_NODE = False

    CATEGORY = "TKVideoZoom"

    
    def tkvideostitcher(self, targetWidth, targetHeight,  image1, image2,image3=None,image4=None, audio=None):

        (bigVid ,sentinels) = self.appendImages(targetWidth, targetHeight, image1, image2, image3, image4, )
        
        pbar = comfy.utils.ProgressBar( len(bigVid) ) 
        numFrame=0

        frames=[]
      
        print("Applying Stitching ")
     
        i=0    
        while True : 
            print(".",end="")

            if (i >= len(bigVid)-1) :
                break
                
            if (i < 16) :
                i += 1
                continue;
               
            if pbar:
                pbar.update(i)

            f=None
            
            foundSentinel=False
            for sentinel in sentinels :
           
                if ( i == sentinel -16)  :
                
                  foundSentinel=True
                  # slide
                  for x in range(16) :
                      chunk = targetWidth /16
                      offset = int(targetWidth - (chunk * float(x)))
                      
                      endIdx = i+ 16
                      
                      if (endIdx >= len(bigVid)) :
                          endIdx = x
                          
                      f = self.slide(bigVid[i+x], bigVid[endIdx], offset, targetWidth, targetHeight)
                      if (f is  None) :
                          return (None, None)  #  something went horribly wrong
                          
                      frames.append(f)
                      numFrame += 1  
                      
                  i+=16
                      
            if not foundSentinel :
                frames.append(bigVid[i])
                numFrame += 1       
                
            i += 1
 
            
        # Convert to tensor
        numpy_array = np.array(frames)
        theTensor = torch.from_numpy(numpy_array)
        theTensor = theTensor.float()  / 255.0
        
        print(theTensor.shape)
                        
        return (theTensor , audio, )


    def appendImages(self,  targetWidth, targetHeight, image1,image2,image3,image4  ) :
       
       frames=[]
       sentinels=[]
       
       # IMAGE 1
       tensor_shape = image1.shape
       frame_count = tensor_shape[0]
       height = tensor_shape[1]
       width = tensor_shape[2]
        
       video_numpy = (image1.numpy() * 255).astype(np.uint8)  #convert video
       for frame in video_numpy :
          f = self.resizeImage(frame, targetWidth, targetHeight, width , height)
          frames.append(f)
       sentinels.append( len(frames))
        
       # IMAGE 2
       tensor_shape = image2.shape
       frame_count = tensor_shape[0]
       height = tensor_shape[1]
       width = tensor_shape[2]
        
       video_numpy = (image2.numpy() * 255).astype(np.uint8)  #convert video
       for frame in video_numpy :
          f = self.resizeImage(frame, targetWidth, targetHeight, width, height)
          frames.append(f)
       sentinels.append( len(frames))
        

       # IMAGE 3
       if (image3 is not None) :
           tensor_shape = image3.shape
           frame_count = tensor_shape[0]
           height = tensor_shape[1]
           width = tensor_shape[2]
            
           video_numpy = (image3.numpy() * 255).astype(np.uint8)  #convert video
           for frame in video_numpy :
              f = self.resizeImage(frame, targetWidth, targetHeight, width, height)
              frames.append(f)
           sentinels.append( len(frames))
  
  
       if (image4 is not None) :
           tensor_shape = image4.shape
           frame_count = tensor_shape[0]
           height = tensor_shape[1]
           width = tensor_shape[2]
            
           video_numpy = (image4.numpy() * 255).astype(np.uint8)  #convert video
           for frame in video_numpy :
              f = self.resizeImage(frame, targetWidth, targetHeight, width, height)
              frames.append(f)
           sentinels.append( len(frames))
           
       return (frames,sentinels)
  
    
    def resizeImage(self, image, targetWidth,targetHeight, width, height) :
        if (image is None) :
           return None
           
          
        if (  float(width) / float(targetWidth) < float(height) / float(targetHeight) ) :  # reduce by height factor
            factor = float(width)/float(targetWidth)
            zWidth=int(width / factor)
            zHeight=int(height / factor)
            zoomFrame = cv2.resize(image, (zWidth,zHeight), interpolation=cv2.INTER_AREA)
            diffH = int((zHeight - targetHeight)/2)
            cropImg = zoomFrame[ diffH:targetHeight+diffH,    0:targetWidth :]
        else :  # reduce by width factor
            factor = float(height)/float(targetHeight)
            zWidth=int(width / factor)
            zHeight=int(height / factor)
            zoomFrame = cv2.resize(image, (zWidth, zHeight), interpolation=cv2.INTER_AREA)
            diffW = int((zWidth - targetWidth)/2)
            cropImg = zoomFrame[ 0:targetHeight,    diffW:targetWidth+diffW :]
          
            
            
        return cropImg
        
        
    def slide(self, frame1, frame2, offset, targetWidth, targetHeight, ):

    
        slideframe = frame1.copy()
        try :
            slideframe[0:targetHeight,  0:offset,:]           = frame1[0:targetHeight,   targetWidth-offset:targetWidth,:]
            slideframe[0:targetHeight,  offset:targetWidth,:] = frame2[0:targetHeight,   0:targetWidth-offset,:]
        except ValueError :
          print("Value error - Cancelling.. try again with different resolution")  
          return None

          
        return slideframe
                    