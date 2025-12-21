from moviepy import VideoFileClip

INPUT_VIDEO = "373-demo-final.mp4"     # path to your MP4
OUTPUT_GIF = "output.gif"

START_TIME = 53.0
END_TIME = 70.0

FPS = 10
SCALE = 0.4

clip = VideoFileClip(INPUT_VIDEO)

# MoviePy 2.x uses subclipped()
subclip = clip.subclipped(START_TIME, END_TIME)

# Resize
subclip = subclip.resized(SCALE)

# Write GIF
subclip.write_gif(OUTPUT_GIF, fps=FPS)

print("GIF created successfully!")
