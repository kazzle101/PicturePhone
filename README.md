# Picture Phone Decode 
This is an attempt to decode the data sample given in Techmoan's YouTube video about the Sony PCT-15 Face to Face - a 80's telephone that can transmit and recieve photos. https://www.youtube.com/watch?v=8_Yz0TT439Q I took the data sample from the video.

The method I have used is a bit basic, rather than listening for the sync tone the width of the image is adjusted until it looks right. The picturephone.py creates two files, ppVideo.mp4 and ppVideoShift.mp4. The first is a video of the width being adjusted, the second shifts the image to the left so it is properly framed.

Here is an edited together video of the output: https://www.youtube.com/watch?v=tmt7xz1QkUM

I won't be doing anymore on this particular method. I need to work out how to do this properly, by having the program listen to the tones and decode accoringly.
