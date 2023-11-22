<font face="times new roman">


<table width="800">
<tbody><tr>
<td>
## Fall 2023 CS543/ECE549 
## Assignment 1: Registering Prokudin-Gorskii color separations of the Russian Empire
### Due date: Monday, September 11, 11:59:59 PM

![](./mp4_part1_files/prokudin_gorskii.jpg)  

<small>This assignment was adapted from [A. Efros](http://graphics.cs.cmu.edu/courses/15-463/2010_fall/) by Prof. S. Lazebnik; updated by Daniel McKee; and gently modified by D.A. Forsyth.</small>

### Background

&#10;[Sergei Mikhailovich Prokudin-Gorskii](http://en.wikipedia.org/wiki/Prokudin-Gorskii) (1863-1944)
was a photographer who, between the years 1909-1915,
traveled the Russian empire and took thousands of photos of everything he saw. He used an early color technology that
involved recording three exposures of every scene onto a glass plate using a red, green, and blue filter. These are known as **color separations**.  Back then,
there was no way to print such photos, and they had to be displayed using a special projector. Prokudin-Gorskii
left Russia in 1918. His glass plate negatives survived and were purchased by the Library of Congress in 1948.
Today, a digitized version of the Prokudin-Gorskii collection is 
[available online](http://www.loc.gov/exhibits/empire/gorskii.html).
&#10;### Overview

 The goal of this assignment is to learn to work with images by taking
the digitized Prokudin-Gorskii glass plate images and automatically producing a color image 
with as few visual artifacts as possible. In order to do this, you will 
need to extract the three color channel images, place them on top of each other, and align them so that they form 
	a single RGB color image. 
	
	 Some details are quite important.  You should notice that it matters how you crop the images
		when you align them -- the separations may not overlap exactly.  We have provided an RGB image to check your code on at [ this location](http://luthuli.cs.uiuc.edu/~daf/courses/CV23/Assignments/checkimage.png).  You should separate this image into three layers (R, G and B), then place each of those layers inside a slightly bigger, all white, layer, at different locations.  Now register these three.  You can tell whether you have the right answer in two ways: first, you shifted the layers with respect to one another, so you know the right shift to register them; second, if you look at the registered picture, the colors should be pure.
	
	
	You will need to implement this assignment in Python, and you should familiarize yourself with libraries 
for scientific computing and image processing including [NumPy](https://numpy.org/) and [PIL](https://pillow.readthedocs.io/en/stable/).

### Data

A zip archive with six input images for the basic alignment experiments is available [here](https://slazebni.cs.illinois.edu/fall22/assignment1/data.zip). 
The high-resolution images for multiscale alignment experiments are available in [this archive](https://slazebni.cs.illinois.edu/fall22/assignment1/data_hires.zip) (the file is over 150MB).
	
	<b> Separations: </b>  We have made no effort to determine what order the separations are in, and we believe that this changes from file to file.  You should try each of the 6 available options and
	see which gives the most plausible image.  This is relatively easy to do (if you get the order wrong for the example image at the top of the page, you'll find that the doors are an implausible color).

### Detailed instructions

Your program should divide the image into three equal parts (channels) and align two of the channels to the 
third (you should try different orders of aligning the channels and figure out which one works the best).
For each input image, you will need to include in your report the colorized output and the (x,y) displacement vectors 
that were used to align the channels.

&#10;<b>Basic alignment.</b> The easiest way to align the parts is to exhaustively search over a window of possible displacements 
(say [-15,15] pixels independently for the x and y axis), score each one using some image matching metric, 
and take the displacement with the best score.
There is a number of possible metrics that one could use to score how well the images match.
The most basic one is the *L2 norm* of the pixel differences of the two channels, also known as 
the *sum of squared differences* (SSD), which in Python is simply <tt>sum((image1-image2)**2)</tt> for images loaded as NumPy arrays.
Note that in our case, the images to be matched do not actually have the same brightness values (they are
different color channels), so a cleverer metric might work better. One such possibility is *normalized 
cross-correlation (NCC)*, which is simply the dot product between the two images normalized to have
zero mean and unit norm. Test your basic alignment solution on the first set of six lower resolution images.



&#10;<b>Multiscale alignment.</b>
For the high-resolution glass plate scans provided above, exhaustive search over all possible displacements will become prohibitively expensive.
To deal with this case, implement a faster search procedure using an *image pyramid*. An image pyramid 
represents the image at multiple scales (usually scaled by a factor of 2) and the processing is done 
sequentially starting from the coarsest scale (smallest image) and going down the pyramid, updating your 
estimate as you go. It is very easy to implement by adding recursive calls to your original single-scale 
implementation.


### For Bonus Points

Implement and test any additional ideas you may have for improving the
quality of the colorized images. 
For example, the borders of the photograph will have strange colors since the three channels won't exactly 
align. See if you can devise an automatic way of cropping the border to get rid of the bad stuff. 
One possible idea is that the information in the good parts of the image generally agrees across 
the color channels, whereas at borders it does not. 


If you have other ideas for further speeding up alignment of high-resolution images, you may also implement and test those.


### What to turn in

You should turn in both your <b><font color="red">code</font></b> and a <b><font color="red">report</font></b> discussing your solution and results.
The report should contain the following:


-  A brief description of your implemented solution, focusing especially on the more
    "non-trivial" or interesting parts of the solution. What implementation choices did you
    make, and how did they affect the quality of the result and the speed of computation?
    What are some artifacts and/or limitations of your implementation, and what are 
    possible reasons for them?  
    <br>
    
- For the multiscale solution, report on its improvement in terms of running time (feel free to use an estimate if the single-scale solution takes too long to run). For timing, you can use the python <tt>time</tt> module. For example:   
    <br>
    
    <tt>
    import time  
    
    start_time = time.time()   
    
    # your code   
    
    end_time = time.time()    
    
    total_time = end_time - start_time  
    <br>
    </tt>
    
-  The output color image for every single input glass plate and the displacement vectors
    that were used to align the channels. 
    Include outputs and displacements for both the six lower resolution images processed with your basic solution 
    and the three high-resolution images aligned with the multiscale solution.
    When inserting results images into your report, you
    should resize/compress them appropriately to keep the file size manageable (under 20MB ideally) -- but make sure
    that the correctness and quality of your output can be clearly and easily judged. 
    <b>You will not receive credit for any results you have obtained, but failed to include
    directly in the report PDF file.</b>
      
    <br>
    
-  Any bonus improvements you attempted, with output. <b>Any parts of the report you are submitting for extra credit should be clearly marked as such.</b>


### Submission Instructions

&#10;To submit this assignment, you must upload the following files on <b>[Canvas](https://canvas.illinois.edu/)</b>:

1. Your code. The filename should be <b>lastname_firstname_a1.py</b> (or another Python extension).
1. A brief report <b>in a single PDF file</b> with all your results and discussion following this <b>[template](https://docs.google.com/document/d/1HM3RkxhWviSILRLVOccSZrIOxiasjvL0L1Nf2uQUxoE/edit?usp=sharing)</b>. The filename should be <b>lastname_firstname_a1.pdf</b>.
1. All your output images <b>in a single zip file</b>. The filename should be <b>lastname_firstname_a1.zip</b>. Note that this zip file is for backup documentation only, in case we cannot see the images in your PDF report clearly enough. As stated above, <b><font color="red">you will not receive credit for any output images that are part of the zip file but are not shown (in some form) in the report PDF.</font></b>


Please refer to [course policies](http://slazebni.cs.illinois.edu/fall22/policies.html) on academic honesty, collaboration, late days, etc.


</td></tr></tbody></table>


</font><div id="extwaiokist" style="display: none;" v="dckih" q="dfaaaefa" c="736.4" i="765" u="3.595" s="11202301" sg="svr_09102316-ga_11202301-bai_11202311" d="1" w="false" e="" a="2" m="BMe=" vn="3adgd"><div id="extwaigglbit" style="display: none;" v="dckih" q="dfaaaefa" c="736.4" i="765" u="3.595" s="11202301" sg="svr_09102316-ga_11202301-bai_11202311" d="1" w="false" e="" a="2" m="BMe="></div></div>