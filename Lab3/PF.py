import cv2
import numpy as np
from PF_utils import *


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
VFILENAME = "../material/Video3.mp4"

NUM_PARTICLES = 5000
VEL_RANGE = 0.5
frame_height = 720
frame_width = 1280

TARGET_COLOR = np.array((66,63, 105))

POS_SIGMA = 0.75
VEL_SIGMA = 0.1


# location =[]
# particles = initialize_particles(NUM_PARTICLES,frame_width,frame_height,VEL_RANGE)


# for frame in get_frames(VFILENAME):
#     if frame is None: break
#     terminate = display(frame, particles, location,NUM_PARTICLES)
#     if terminate:
#         break

# cv2.destroyAllWindows() 

# location = []
# particles = initialize_particles(NUM_PARTICLES,frame_width,frame_height,VEL_RANGE)


# for frame in get_frames(VFILENAME):
#     if frame is None: break
#     particles = apply_velocity(particles)
#     particles = enforce_edges(particles,NUM_PARTICLES,frame_width,frame_height)
#     terminate = display(frame, particles, location,NUM_PARTICLES)
#     if terminate:
#         break

# cv2.destroyAllWindows()

# particles = initialize_particles(NUM_PARTICLES,frame_width,frame_height,VEL_RANGE)


# for frame in get_frames(VFILENAME):
#     if frame is None: break
#     particles = apply_velocity(particles)
#     particles = enforce_edges(particles,NUM_PARTICLES,frame_width,frame_height)
#     errors = compute_errors(particles, frame,NUM_PARTICLES,TARGET_COLOR)
#     weights = compute_weights(errors,particles,frame_width,frame_height)
#     particles, location = resample(particles, weights,NUM_PARTICLES)

#     terminate = display(frame, particles, location,NUM_PARTICLES)
#     if terminate:
#         break

# cv2.destroyAllWindows()

particles = initialize_particles(NUM_PARTICLES,frame_width,frame_height,VEL_RANGE)

for frame in get_frames(VFILENAME):
    if frame is None: break

    particles = apply_velocity(particles)
    particles = enforce_edges(particles,NUM_PARTICLES,frame_width,frame_height)
    errors = compute_errors(particles, frame,NUM_PARTICLES,TARGET_COLOR)
    weights = compute_weights(errors,particles,frame_width,frame_height)
    particles, location = resample(particles, weights,NUM_PARTICLES)
    particles = apply_noise(particles,POS_SIGMA,VEL_SIGMA,NUM_PARTICLES)
    terminate = display(frame, particles, location,NUM_PARTICLES)
    if terminate:
        break

cv2.destroyAllWindows()

