# Locating a Defined Area in the Star Map

In this python script, I tried to locate a defined area in a given image using Brute-Force Matching with ORB Descriptors. The output images give related feature matches and corresponding corner points of the defined area using homography techniques. 

The python script can be run from terminal with the command:<br/>
``` Ruby
python cv_task_handenur.py --input1 'Small_image.png' --input2 'Small_image_rotated.png' --input3 'StarMap.png'
```
You can see the original images and the output images below:<br/>

<p align="center">
<img src="https://github.com/handenurc/cv_task_hande/blob/master/StarMap.png" height="400" width="600"> 
<p>
  
<p align="center">
Image: Star Map
<p>
  
<p align="center">
<img src="https://github.com/handenurc/cv_task_hande/blob/master/Small_area.png" height="100" width="100"/> <img src="https://github.com/handenurc/cv_task_hande/blob/master/Small_area_rotated.png" height="100" width="100"/>
<p>
  
<p align="center">
Images of cropped (and possibly rotated) areas
<p>

<p align="center">
<img src="https://github.com/handenurc/cv_task_hande/blob/master/Matches1.png" height="600" width="800">
<p>
<p align="center">
Output image 1: Feature matching lines and corner points (blue) <br/> between cropped image (left) and Star Map (right)
<p>
  
<p align="center">
<img src="https://github.com/handenurc/cv_task_hande/blob/master/Matches2.png" height="600" width="800">
<p>

<p align="center">
Output image 2: Feature matches and corner points (blue) <br/> between cropped and rotated image (left) and Star Map (right)
<p>
