# AR-Simcity
Description: 
Name: AR Simcity
Description: Use AR to build virtual buildings at given ‘floor’ in the real ‘world’ and edit them based on the options given. There will be population, revenue, and costs change during the whole process. 

How to run:
Firstly, you need two iPads(or an iPhone and an iPad, but two iPads will be preferable). Then, there are two photos in the zip file whose name is iPad and iPad(Hand)/iPhone(Hand). Show the iPad photo on your iPad and use the computer camera to take photos from different angles, which will help the code to get the camera parameters of your camera(parameters help to correct the distortion of the camera; photos’ name should be train* and test*). Or you can use the photos already uploaded to run this game, but since the distortion of the different camera is not the same, it may affect the stability and performance of the code.

Secondly, run the code using the photos taken or photos given. Place the iPad in a still place(it’s better to turn up the brightness of two iPads) and show the ‘iPad’ image on the screen. Meanwhile, use another iPad to show the ‘Hand’ image, and then you can use the iPad(iPhone) in your hand to point at options given to take actions(build, upgrade, demolish, move)(If you try to use iPhone to play this game, try to make sure the QR code’s in the iPad and iPhone are the same. There are two ways to do that: 1. Use the iPad(Hand) picture and try to zoom a little bit; 2. Use the iPhone picture and don’t let the picture take up the entire screen - thumbnail in your album will be good enough - there is a iPhoneHandDemo picture in the zip file showing). 

You can use your ‘hand’ to select any random location in the ‘ground,’ and when the block is selected, you can take any actions you want based on the options given. Or you can use the ‘Move’ button to move your location and do the same actions as the selected part. There is the secondary menu when you point at 'build' and 'move' options, and you can go back to the main menu by pointing at the 'back' option or point at the same button again, and the 'cancel' option will undo your actions. Different types of building in the upper interface will be a 3D model, which means you can lift the iPad placed in the table and rotate your iPad to see from different angles. Meanwhile, when you build the house using the buttons in the ‘real’ world(buildings and actions will show in the iPad ground), the bottom interface will also show the location and the parameters the city will have. 

Libraries:
OpenCV
Tkinter	

Shortcuts:
1: build ’House’
2: build ‘Apartment’
3: build ‘Official’
u: upgrade
d : demolish
i: up
k: down
l: right
j: left
