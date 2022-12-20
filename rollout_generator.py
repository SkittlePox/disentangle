import os
import matplotlib.pyplot as plt
from visgrid.envs import GridworldEnv, TaxiEnv
from visgrid.agents.expert import TaxiExpert
from visgrid.wrappers.transforms import NoiseWrapper, ClipWrapper


def save_rollouts(images, marker):
    os.mkdir(f"./images/agent-{marker}")
    for i in range(len(images)):
        plt.imsave(f"./images/agent-{marker}/{i}.png", images[i])


def generate_rollouts(env, agent, max_steps=1000, min_rollouts=100, save=True):
    imsum = 0
    all_images = []

    def rollout():
        ob, _ = env.reset()
        images = [ob]

        n_steps = 0
        while n_steps < max_steps:
            action = agent.act()
            ob, reward, terminal, _, _ = env.step(action)
            images.append(ob)
            n_steps += 1
            if terminal:
                break

        return images

    rollout_num = 0

    while imsum < min_rollouts:
        images = rollout()
        imsum += len(images)
        all_images.append(images)
        if save:
            save_rollouts(images, rollout_num)
            rollout_num += 1

    return all_images


env = TaxiEnv(exploring_starts=False,
              terminate_on_goal=True,
              depot_dropoff_only=False,
              should_render=True)
# env = NoiseWrapper(env, 0.05)
# env = ClipWrapper(env)

expert = TaxiExpert(env)

generate_rollouts(env, expert, min_rollouts=1000)
