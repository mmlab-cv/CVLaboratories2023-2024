import cv2
import numpy as np

def get_frames(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame =video.read()
        if ret:
            yield frame 
        else:
            break
    video.release()        
    yield None


def display(frame, particles, location,NUM_PARTICLES):
    if len(particles)> 0:
        for i in range(NUM_PARTICLES):
            x = int(particles[i,0])
            y = int(particles[i,1])
            cv2.circle(frame,(x,y),1,(0,255,0),1)
    if len(location) > 0:
        cv2.circle(frame,location,15,(0,0,255),5)
    cv2.imshow('frame', frame)
    #stop the video if pressing the escape button
    if cv2.waitKey(30)==27:
        if cv2.waitKey(0)==27:
            return True 

    return False

def initialize_particles(NUM_PARTICLES,frame_width,frame_height,VEL_RANGE):
    particles = np.random.rand(NUM_PARTICLES,4)
    particles = particles * np.array((frame_width,frame_height, VEL_RANGE,VEL_RANGE))
    particles[:,2:4] -= VEL_RANGE/2.0
    return particles

def apply_velocity(particles):
    particles[:,0] += particles[:,2]
    particles[:,1] += particles[:,3]

    return particles

def enforce_edges(particles,NUM_PARTICLES,frame_width,frame_height):
    for i in range(NUM_PARTICLES):
        particles[i,0] = max(0,min(frame_width-1, particles[i,0]))
        particles[i,1] = max(0,min(frame_height-1, particles[i,1]))
    return particles

def compute_errors(particles, frame,NUM_PARTICLES,TARGET_COLOR):
    
    errors = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        x = int(particles[i,0])
        y= int(particles[i,1])
        pixel_color = frame[y, x, :]
        errors[i] = np.sum((TARGET_COLOR - pixel_color)**2)
    
    return errors


def compute_weights(errors,particles,frame_width,frame_height):
    weights = np.max(errors) - errors
    
    weights[
        (particles[:,0]==0) |
        (particles[:,0]==frame_width-1) |
        (particles[:,1]==0) |
        (particles[:,1]==frame_height-1) ] = 0

    weights = weights**8
        
    return weights

def resample(particles, weights,NUM_PARTICLES):
    probabilities = weights / np.sum(weights)
    index_numbers = np.random.choice(
        NUM_PARTICLES,
        size=NUM_PARTICLES,
        p=probabilities)
    particles = particles[index_numbers, :]
    
    x = np.mean(particles[:,0])
    y = np.mean(particles[:,1])
    
    return particles, [int(x), int(y)]

def apply_noise(particles,POS_SIGMA,VEL_SIGMA,NUM_PARTICLES):
    noise= np.concatenate(
    (
        np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
        np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES,1)),
        np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1)),
        np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES,1))
    
    ),
    axis=1)
    
    particles += noise
    return particles
