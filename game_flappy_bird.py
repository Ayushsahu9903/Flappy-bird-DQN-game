import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Creating our environment
env = gym.make("FlappyBird-v0", render_mode="human")

state, info = env.reset()
done = False

# Initialize Pygame keyboard
pygame.init()

while not done:

    action = 0  # 0 = no flap, 1 = flap

    # Check keyboard events
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            done = True

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:
                action = 1  # flap

    # Take action
    state, reward, terminated, truncated, info = env.step(action)

    # Episode ends if terminated OR truncated
    done = terminated or truncated

# Close environment
env.close()
pygame.quit()   