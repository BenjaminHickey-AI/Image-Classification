# AI Image Processing and Classification Project
##Heat Map
The Classifier focused almost entirely on the cat's face, looking for key features such as the shape of its eyes, nose, mouth, and ears.
This makes sense as it is looking of visual patterns and similariities in those patterns to recognize specific objects.
It very much ignored the background and just focused on what really popped out in the image.
##Occlusions
The classifier was unable to identify the image with the blackbox occlusion with a sharp drop in confidence.
The noise occluder, which covered the region in random monochrome noise also caused the classsifier to struggle to understand the image.
The blur occluder did not remove features such as shape, so the classifier still had the ability to decifer the cat.
##Custom Filter
The custom filter I chose was a pop art halftone filter, this filter works by first posterizing the image, reducing its color depth.
Next, the filter adds halftone dots of varying sizes based on brightness.
Finally, it blends the resulting dots with the posterized image.
This gives it the art style of old comic books with their iconic dots and reduced saturation.
##AI Collaboration
I used AI for practically all parts of the programming in the project, giving it prompts to both conceptualize, write, and explain the code.
The biggest challenge I had was troubleshooting the packages required for the project, which I used AI to attempt to solve.
