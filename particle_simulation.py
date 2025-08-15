import multiprocessing
import time
import numpy as np
import pygame

N = 1000  # Number of particles; increase for more CPU stress
RADIUS = 5.0
DT = 1.0
SIZE = np.array([800, 600])

def simulate_frame(pos, vel):
    pos += vel * DT

    # Bounce off walls
    over_left = pos[:, 0] < RADIUS
    over_right = pos[:, 0] > SIZE[0] - RADIUS
    over_top = pos[:, 1] < RADIUS
    over_bottom = pos[:, 1] > SIZE[1] - RADIUS

    vel[over_left, 0] *= -1
    vel[over_right, 0] *= -1
    vel[over_top, 1] *= -1
    vel[over_bottom, 1] *= -1

    # Collisions
    ii, jj = np.triu_indices(N, k=1)
    diff = pos[jj] - pos[ii]
    dist_sq = np.sum(diff**2, axis=1)
    collide = dist_sq < (2 * RADIUS)**2
    colliding_pairs = np.where(collide)[0]

    for k in colliding_pairs:
        i = ii[k]
        j = jj[k]
        dvec = pos[j] - pos[i]
        d = np.sqrt(dist_sq[k])
        if d < 1e-5:
            continue
        n = dvec / d
        dv = vel[j] - vel[i]
        dot = np.dot(dv, n)
        if dot > 0:
            continue  # separating
        impulse = dot * n
        vel[i] += impulse
        vel[j] -= impulse

def worker(duration):
    np.random.seed()  # Different seed per process
    pos = np.random.uniform(RADIUS, SIZE - RADIUS, (N, 2))
    vel = np.random.uniform(-50, 50, (N, 2))
    start = time.time()
    frames = 0
    while time.time() - start < duration:
        simulate_frame(pos, vel)
        frames += 1
    return frames

if __name__ == '__main__':
    duration = 45 * 60  # 45 minutes in seconds
    num_cores = multiprocessing.cpu_count()
    print(f"Starting CPU load test with automated particle simulation on {num_cores} cores for 45 minutes...")

    # Start worker processes if more than 1 core
    if num_cores > 1:
        pool = multiprocessing.Pool(num_cores - 1)
        futures = [pool.apply_async(worker, (duration,)) for _ in range(num_cores - 1)]
    else:
        pool = None
        futures = []

    # Main process runs the graphic simulation
    pygame.init()
    screen = pygame.display.set_mode((int(SIZE[0]), int(SIZE[1])))
    pygame.display.set_caption("Automated Particle Simulation CPU Load Test")
    clock = pygame.time.Clock()

    pos = np.random.uniform(RADIUS, SIZE - RADIUS, (N, 2))
    vel = np.random.uniform(-50, 50, (N, 2))
    start = time.time()
    frames_main = 0
    running = True
    while running and time.time() - start < duration:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        simulate_frame(pos, vel)
        frames_main += 1

        screen.fill((0, 0, 0))
        for p in pos:
            pygame.draw.circle(screen, (255, 255, 255), (int(p[0]), int(p[1])), int(RADIUS))
        pygame.display.flip()

        # No FPS limit to maximize CPU usage

    pygame.quit()

    # Collect results from workers
    if pool:
        pool.close()
        pool.join()
        results = [f.get() for f in futures]
    else:
        results = []

    print("Frames processed in main (with graphics):", frames_main)
    if results:
        print("Frames processed per worker:", results)
    print("Load test completed.")