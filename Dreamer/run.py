import wrappers
import tools
import tensorflow as tf
import argparse
import functools
import yaml
import os
import pathlib
import sys

from tensorflow.keras.mixed_precision import experimental as prec
from dreamers import Dreamer, SeparationDreamer, InverseDreamer
from env_tools import count_steps, make_env

METHOD2DREAMER = {
    'dreamer': Dreamer,
    'tia': SeparationDreamer,
    'inverse': InverseDreamer
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'
tf.get_logger().setLevel('ERROR')
sys.path.append(str(pathlib.Path(__file__).parent))


def main(method, config):

    if method == 'separation':
        config.logdir = os.path.join(
            config.logdir, config.task,
            'separation' + '_' + str(config.disen_neg_rew_scale) +
            '_' + str(config.disen_rec_scale),
            str(config.seed))
    else:
        config.logdir = os.path.join(
            config.logdir, config.task,
            method,
            str(config.seed))
    
    logdir = pathlib.Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = os.path.join(config.logdir, 'snapshots')
    snapshot_dir = pathlib.Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.logdir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(config), f, sort_keys=False)

    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_policy(prec.Policy('mixed_float16'))
    config.steps = int(config.steps)
    config.logdir = logdir
    print('Logdir', config.logdir)

    # Create environments.
    datadir = config.logdir / 'episodes'
    writer = tf.summary.create_file_writer(
        str(config.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    train_envs = [wrappers.Async(lambda: make_env(
        config, writer, 'train', datadir, config.video_dir, store=True), config.parallel)
        for _ in range(config.envs)]
    test_envs = [wrappers.Async(lambda: make_env(
        config, writer, 'test', datadir, config.video_dir, store=False), config.parallel)
        for _ in range(config.envs)]
    actspace = train_envs[0].action_space

    # Prefill dataset with random episodes.
    step = count_steps(datadir, config)
    prefill = max(0, config.prefill - step)
    print(f'Prefill dataset with {prefill} steps.')
    def random_agent(o, d, _): return ([actspace.sample() for _ in d], None)
    tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
    writer.flush()

    # Train and regularly evaluate the agent.
    step = count_steps(datadir, config)
    print(f'Simulating agent for {config.steps-step} steps.')
    DreamerModel = METHOD2DREAMER[method]
    agent = DreamerModel(config, datadir, actspace, writer)
    if (config.logdir / 'variables.pkl').exists():
        print('Load checkpoint.')
        agent.load(config.logdir / 'variables.pkl')
    state = None
    should_snapshot = tools.Every(config.snapshot_every)
    while step < config.steps:
        print('Start evaluation.')
        tools.simulate(
            functools.partial(agent, training=False), test_envs, episodes=1)
        writer.flush()
        print('Start collection.')
        steps = config.eval_every // config.action_repeat
        state = tools.simulate(agent, train_envs, steps, state=state)
        step = count_steps(datadir, config)
        agent.save(config.logdir / 'variables.pkl')
        if should_snapshot(step):
            agent.save(snapshot_dir / ('variables_' + str(step) + '.pkl'))
    for env in train_envs + test_envs:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method', type=str, choices=['dreamer', 'inverse', 'tia'], required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    config_path = 'train_configs/' + args.method + '.yaml'
    configs = yaml.safe_load(
        (pathlib.Path(__file__).parent / config_path).read_text())
    config_ = {}
    for name in args.configs:
        config_.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in config_.items():
        parser.add_argument(
            f'--{key}', type=tools.args_type(value), default=value)
    main(args.method, parser.parse_args(remaining))
