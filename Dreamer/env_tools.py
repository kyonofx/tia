import os
import json
import dmc2gym
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

import tools
import wrappers

def preprocess(obs, config):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    with tf.device('cpu:0'):
        obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
        clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[
            config.clip_rewards]
        obs['reward'] = clip_rewards(obs['reward'])
    return obs

def count_steps(datadir, config):
    return tools.count_episodes(datadir)[1] * config.action_repeat

def summarize_episode(episode, config, datadir, writer, prefix):
    episodes, steps = tools.count_episodes(datadir)
    length = (len(episode['reward']) - 1) * config.action_repeat
    ret = episode['reward'].sum()
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [
        (f'{prefix}/return', float(episode['reward'].sum())),
        (f'{prefix}/length', len(episode['reward']) - 1),
        ('episodes', episodes)]

    step = count_steps(datadir, config)
    with (config.logdir / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
    with writer.as_default():  # Env might run in a different thread.
        tf.summary.experimental.set_step(step)
        [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
        if prefix == 'test':
            tools.video_summary(f'sim/{prefix}/video', episode['image'][None])

def make_env(config, writer, prefix, datadir, video_dir, store):
    suite, domain_task_distractor = config.task.split('_', 1)
    domain, task_distractor = domain_task_distractor.split('_', 1)
    task, distractor = task_distractor.split('_', 1)

    if distractor == 'driving':
        img_source = 'video'
        total_frames = 1000
        resource_files = os.path.join(video_dir, '*.mp4')
    elif distractor == 'noise':
        img_source = 'noise'
        total_frames = None
        resource_files = None
    elif distractor == 'none':
        img_source = None
        total_frames = None
        resource_files = None
    else:
        raise NotImplementedError

    env = dmc2gym.make(
        domain_name=domain,
        task_name=task,
        resource_files=resource_files,
        img_source=img_source,
        total_frames=total_frames,
        seed=config.seed,
        visualize_reward=False,
        from_pixels=True,
        height=config.image_size,
        width=config.image_size,
        frame_skip=config.action_repeat
    )
    env = wrappers.DMC2GYMWrapper(env)
    env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
    callbacks = []
    if store:
        callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    callbacks.append(
        lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
    env = wrappers.Collect(env, callbacks, config.precision)
    env = wrappers.RewardObs(env)
    return env
