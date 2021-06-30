import tools
import models
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec
import tensorflow as tf
import numpy as np
import collections
import functools
import json
import time

from env_tools import preprocess, count_steps

def load_dataset(directory, config):
    episode = next(tools.load_episodes(directory, 1))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}

    def generator(): return tools.load_episodes(
        directory, config.train_steps, config.batch_length,
        config.dataset_balance)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(preprocess, config=config))
    dataset = dataset.prefetch(10)
    return dataset

class Dreamer(tools.Module):

    def __init__(self, config, datadir, actspace, writer):
        self._c = config
        self._actspace = actspace
        self._actdim = actspace.n if hasattr(
            actspace, 'n') else actspace.shape[0]
        self._writer = writer
        self._random = np.random.RandomState(config.seed)
        self._should_pretrain = tools.Once()
        self._should_train = tools.Every(config.train_every)
        self._should_log = tools.Every(config.log_every)
        self._last_log = None
        self._last_time = time.time()
        self._metrics = collections.defaultdict(tf.metrics.Mean)
        self._metrics['expl_amount']  # Create variable for checkpoint.
        self._float = prec.global_policy().compute_dtype
        self._strategy = tf.distribute.MirroredStrategy()
        with tf.device('cpu:0'):
            self._step = tf.Variable(count_steps(
                datadir, config), dtype=tf.int64)
        with self._strategy.scope():
            self._dataset = iter(self._strategy.experimental_distribute_dataset(
                load_dataset(datadir, self._c)))
        self._build_model()

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step.numpy().item()
        tf.summary.experimental.set_step(step)
        if state is not None and reset.any():
            mask = tf.cast(1 - reset, self._float)[:, None]
            state = tf.nest.map_structure(lambda x: x * mask, state)
        if self._should_train(step):
            log = self._should_log(step)
            n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
            print(f'Training for {n} steps.')
            with self._strategy.scope():
                for train_step in range(n):
                    log_images = self._c.log_images and log and train_step == 0
                    self.train(next(self._dataset), log_images)
            if log:
                self._write_summaries()
        action, state = self.policy(obs, state, training)
        if training:
            self._step.assign_add(len(reset) * self._c.action_repeat)
        return action, state

    @tf.function
    def policy(self, obs, state, training):
        if state is None:
            latent = self._dynamics.initial(len(obs['image']))
            action = tf.zeros((len(obs['image']), self._actdim), self._float)
        else:
            latent, action = state
        embed = self._encode(preprocess(obs, self._c))
        latent, _ = self._dynamics.obs_step(latent, action, embed)
        feat = self._dynamics.get_feat(latent)
        if training:
            action = self._actor(feat).sample()
        else:
            action = self._actor(feat).mode()
        action = self._exploration(action, training)
        state = (latent, action)
        return action, state

    def load(self, filename):
        super().load(filename)
        self._should_pretrain()

    @tf.function()
    def train(self, data, log_images=False):
        self._strategy.experimental_run_v2(
            self._train, args=(data, log_images))

    def _train(self, data, log_images):
        with tf.GradientTape() as model_tape:
            embed = self._encode(data)
            post, prior = self._dynamics.observe(embed, data['action'])
            feat = self._dynamics.get_feat(post)

            image_pred = self._decode(feat)
            reward_pred = self._reward(feat)

            likes = tools.AttrDict()
            likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
            likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
            if self._c.pcont:
                pcont_pred = self._pcont(feat)
                pcont_target = self._c.discount * data['discount']
                likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
                likes.pcont *= self._c.pcont_scale

            prior_dist = self._dynamics.get_dist(prior)
            post_dist = self._dynamics.get_dist(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self._c.free_nats)

            model_loss = self._c.kl_scale * div - sum(likes.values())
            model_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as actor_tape:
            imag_feat = self._imagine_ahead(post)
            reward = self._reward(imag_feat).mode()
            if self._c.pcont:
                pcont = self._pcont(imag_feat).mean()
            else:
                pcont = self._c.discount * tf.ones_like(reward)
            value = self._value(imag_feat).mode()
            returns = tools.lambda_return(
                reward[:-1], value[:-1], pcont[:-1],
                bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            actor_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as value_tape:
            value_pred = self._value(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = - \
                tf.reduce_mean(discount * value_pred.log_prob(target))
            value_loss /= float(self._strategy.num_replicas_in_sync)

        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)

        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            if self._c.log_scalars:
                self._scalar_summaries(
                    data, feat, prior_dist, post_dist, likes, div,
                    model_loss, value_loss, actor_loss, model_norm, value_norm,
                    actor_norm)
            if tf.equal(log_images, True):
                self._image_summaries(data, embed, image_pred)

    def _build_model(self):
        acts = dict(
            elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
            leaky_relu=tf.nn.leaky_relu)
        cnn_act = acts[self._c.cnn_act]
        act = acts[self._c.dense_act]
        self._encode = models.ConvEncoder(
            self._c.cnn_depth, cnn_act, self._c.image_size)
        self._dynamics = models.RSSM(
            self._c.stoch_size, self._c.deter_size, self._c.deter_size)
        self._decode = models.ConvDecoder(
            self._c.cnn_depth, cnn_act, (self._c.image_size, self._c.image_size, 3))
        self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
        if self._c.pcont:
            self._pcont = models.DenseDecoder(
                (), 3, self._c.num_units, 'binary', act=act)
        self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
        self._actor = models.ActionDecoder(
            self._actdim, 4, self._c.num_units, self._c.action_dist,
            init_std=self._c.action_init_std, act=act)
        model_modules = [self._encode, self._dynamics,
                         self._decode, self._reward]
        if self._c.pcont:
            model_modules.append(self._pcont)
        Optimizer = functools.partial(
            tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
            wdpattern=self._c.weight_decay_pattern)
        self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
        self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
        self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
        self.train(next(self._dataset))

    def _exploration(self, action, training):
        if training:
            amount = self._c.expl_amount
            if self._c.expl_decay:
                amount *= 0.5 ** (tf.cast(self._step,
                                          tf.float32) / self._c.expl_decay)
            if self._c.expl_min:
                amount = tf.maximum(self._c.expl_min, amount)
            self._metrics['expl_amount'].update_state(amount)
        elif self._c.eval_noise:
            amount = self._c.eval_noise
        else:
            return action
        if self._c.expl == 'additive_gaussian':
            return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        if self._c.expl == 'completely_random':
            return tf.random.uniform(action.shape, -1, 1)
        if self._c.expl == 'epsilon_greedy':
            indices = tfd.Categorical(0 * action).sample()
            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            return tf.where(
                tf.random.uniform(action.shape[:1], 0, 1) < amount,
                tf.one_hot(indices, action.shape[-1], dtype=self._float),
                action)
        raise NotImplementedError(self._c.expl)

    def _imagine_ahead(self, post):
        if self._c.pcont:  # Last step could be terminal.
            post = {k: v[:, :-1] for k, v in post.items()}

        def flatten(x): return tf.reshape(x, [-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in post.items()}

        def policy(state): return self._actor(
            tf.stop_gradient(self._dynamics.get_feat(state))).sample()
        states = tools.static_scan(
            lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
            tf.range(self._c.horizon), start)
        imag_feat = self._dynamics.get_feat(states)
        return imag_feat

    def _scalar_summaries(
            self, data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, model_norm, value_norm,
            actor_norm):
        self._metrics['model_grad_norm'].update_state(model_norm)
        self._metrics['value_grad_norm'].update_state(value_norm)
        self._metrics['actor_grad_norm'].update_state(actor_norm)
        self._metrics['prior_ent'].update_state(prior_dist.entropy())
        self._metrics['post_ent'].update_state(post_dist.entropy())
        for name, logprob in likes.items():
            self._metrics[name + '_loss'].update_state(-logprob)
        self._metrics['div'].update_state(div)
        self._metrics['model_loss'].update_state(model_loss)
        self._metrics['value_loss'].update_state(value_loss)
        self._metrics['actor_loss'].update_state(actor_loss)
        self._metrics['action_ent'].update_state(self._actor(feat).entropy())

    def _image_summaries(self, data, embed, image_pred):
        truth = data['image'][:6] + 0.5
        recon = image_pred.mode()[:6]
        init, _ = self._dynamics.observe(embed[:6, :5], data['action'][:6, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self._dynamics.imagine(data['action'][:6, 5:], init)
        openl = self._decode(self._dynamics.get_feat(prior)).mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        tools.graph_summary(
            self._writer, tools.video_summary, self._step, 'agent/openl', openl)

    def image_summary_from_data(self, data):
        truth = data['image'][:6] + 0.5
        embed = self._encode(data)
        post, _ = self._dynamics.observe(
            embed[:6, :5], data['action'][:6, :5])
        feat = self._dynamics.get_feat(post)
        init = {k: v[:, -1] for k, v in post.items()}
        recon = self._decode(feat).mode()[:6]
        prior = self._dynamics.imagine(data['action'][:6, 5:], init)
        openl = self._decode(self._dynamics.get_feat(prior)).mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        tools.graph_summary(
            self._writer, tools.video_summary, self._step, 'agent/eval_openl', openl)

    def _write_summaries(self):
        step = int(self._step.numpy())
        metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
        if self._last_log is not None:
            duration = time.time() - self._last_time
            self._last_time += duration
            metrics.append(('fps', (step - self._last_log) / duration))
        self._last_log = step
        [m.reset_states() for m in self._metrics.values()]
        with (self._c.logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
        [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        self._writer.flush()


class SeparationDreamer(Dreamer):

    def __init__(self, config, datadir, actspace, writer):
        self._metrics_disen = collections.defaultdict(tf.metrics.Mean)
        self._metrics_disen['expl_amount']
        super().__init__(config, datadir, actspace, writer)   

    def _train(self, data, log_images):
        with tf.GradientTape(persistent=True) as model_tape:

            # main
            embed = self._encode(data)
            post, prior = self._dynamics.observe(embed, data['action'])
            feat = self._dynamics.get_feat(post)

            # disen
            embed_disen = self._disen_encode(data)
            post_disen, prior_disen = self._disen_dynamics.observe(
                embed_disen, data['action'])
            feat_disen = self._disen_dynamics.get_feat(post_disen)

            # disen image pred
            image_pred_disen = self._disen_only_decode(feat_disen)

            # joint image pred
            image_pred_joint, image_pred_joint_main, image_pred_joint_disen, mask_pred = self._joint_decode(
                feat, feat_disen)

            # reward pred
            reward_pred = self._reward(feat)

            # optimize disen reward predictor till optimal
            for _ in range(self._c.num_reward_opt_iters):
                with tf.GradientTape() as disen_reward_tape:
                    reward_pred_disen = self._disen_reward(
                        tf.stop_gradient(feat_disen))
                    reward_like_disen = reward_pred_disen.log_prob(
                        data['reward'])
                    reward_loss_disen = -tf.reduce_mean(reward_like_disen)
                    reward_loss_disen /= float(
                        self._strategy.num_replicas_in_sync)
                reward_disen_norm = self._disen_reward_opt(
                    disen_reward_tape, reward_loss_disen)

            # disen reward pred with optimal reward predictor
            reward_pred_disen = self._disen_reward(feat_disen)
            reward_like_disen = tf.reduce_mean(
                reward_pred_disen.log_prob(data['reward']))

            # main model loss
            likes = tools.AttrDict()
            likes.image = tf.reduce_mean(
                image_pred_joint.log_prob(data['image']))
            likes.reward = tf.reduce_mean(reward_pred.log_prob(
                data['reward'])) * self._c.reward_scale
            if self._c.pcont:
                pcont_pred = self._pcont(feat)
                pcont_target = self._c.discount * data['discount']
                likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
                likes.pcont *= self._c.pcont_scale

            prior_dist = self._dynamics.get_dist(prior)
            post_dist = self._dynamics.get_dist(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self._c.free_nats)

            model_loss = self._c.kl_scale * div - sum(likes.values())
            model_loss /= float(self._strategy.num_replicas_in_sync)

            # disen model loss with reward negative gradient
            likes_disen = tools.AttrDict()
            likes_disen.image = tf.reduce_mean(
                image_pred_joint.log_prob(data['image']))
            likes_disen.disen_only = tf.reduce_mean(
                image_pred_disen.log_prob(data['image']))

            reward_like_disen = reward_pred_disen.log_prob(data['reward'])
            reward_like_disen = tf.reduce_mean(reward_like_disen)
            reward_loss_disen = -reward_like_disen

            prior_dist_disen = self._disen_dynamics.get_dist(prior_disen)
            post_dist_disen = self._disen_dynamics.get_dist(post_disen)
            div_disen = tf.reduce_mean(tfd.kl_divergence(
                post_dist_disen, prior_dist_disen))
            div_disen = tf.maximum(div_disen, self._c.free_nats)

            model_loss_disen = div_disen * self._c.disen_kl_scale + \
                reward_like_disen * self._c.disen_neg_rew_scale - \
                likes_disen.image - likes_disen.disen_only * self._c.disen_rec_scale
            model_loss_disen /= float(self._strategy.num_replicas_in_sync)

            decode_loss = model_loss_disen + model_loss

        with tf.GradientTape() as actor_tape:
            imag_feat = self._imagine_ahead(post)
            reward = self._reward(imag_feat).mode()
            if self._c.pcont:
                pcont = self._pcont(imag_feat).mean()
            else:
                pcont = self._c.discount * tf.ones_like(reward)
            value = self._value(imag_feat).mode()
            returns = tools.lambda_return(
                reward[:-1], value[:-1], pcont[:-1],
                bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            actor_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as value_tape:
            value_pred = self._value(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = - \
                tf.reduce_mean(discount * value_pred.log_prob(target))
            value_loss /= float(self._strategy.num_replicas_in_sync)

        model_norm = self._model_opt(model_tape, model_loss)
        model_disen_norm = self._disen_opt(model_tape, model_loss_disen)
        decode_norm = self._decode_opt(model_tape, decode_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)

        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            if self._c.log_scalars:
                self._scalar_summaries(
                    data, feat, prior_dist, post_dist, likes, div,
                    model_loss, value_loss, actor_loss, model_norm, value_norm, actor_norm)
                self._scalar_summaries_disen(
                    prior_dist_disen, post_dist_disen, likes_disen, div_disen,
                    model_loss_disen, reward_loss_disen,
                    model_disen_norm, reward_disen_norm)
            if tf.equal(log_images, True):
                self._image_summaries_joint(
                    data, embed, embed_disen, image_pred_joint, mask_pred)
                self._image_summaries(
                    self._disen_dynamics, self._disen_decode, data, embed_disen, image_pred_joint_disen, tag='disen/openl_joint_disen')
                self._image_summaries(
                    self._disen_dynamics, self._disen_only_decode, data, embed_disen, image_pred_disen, tag='disen_only/openl_disen_only')
                self._image_summaries(
                    self._dynamics, self._main_decode, data, embed, image_pred_joint_main, tag='main/openl_joint_main')

    def _build_model(self):
        acts = dict(
            elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
            leaky_relu=tf.nn.leaky_relu)
        cnn_act = acts[self._c.cnn_act]
        act = acts[self._c.dense_act]

        # Distractor dynamic model
        self._disen_encode = models.ConvEncoder(
            self._c.disen_cnn_depth, cnn_act, self._c.image_size)
        self._disen_dynamics = models.RSSM(
            self._c.disen_stoch_size, self._c.disen_deter_size, self._c.disen_deter_size)
        self._disen_only_decode = models.ConvDecoder(
            self._c.disen_cnn_depth, cnn_act, (self._c.image_size, self._c.image_size, 3))
        self._disen_reward = models.DenseDecoder(
            (), 2, self._c.num_units, act=act)

        # Task dynamic model
        self._encode = models.ConvEncoder(
            self._c.cnn_depth, cnn_act, self._c.image_size)
        self._dynamics = models.RSSM(
            self._c.stoch_size, self._c.deter_size, self._c.deter_size)
        self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
        if self._c.pcont:
            self._pcont = models.DenseDecoder(
                (), 3, self._c.num_units, 'binary', act=act)
        self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
        self._actor = models.ActionDecoder(
            self._actdim, 4, self._c.num_units, self._c.action_dist,
            init_std=self._c.action_init_std, act=act)

        # Joint decode
        self._main_decode = models.ConvDecoderMask(
            self._c.cnn_depth, cnn_act, (self._c.image_size, self._c.image_size, 3))
        self._disen_decode = models.ConvDecoderMask(
            self._c.disen_cnn_depth, cnn_act, (self._c.image_size, self._c.image_size, 3))
        self._joint_decode = models.ConvDecoderMaskEnsemble(
            self._main_decode, self._disen_decode, self._c.precision
        )

        disen_modules = [self._disen_encode,
                         self._disen_dynamics, self._disen_only_decode]
        model_modules = [self._encode, self._dynamics, self._reward]
        if self._c.pcont:
            model_modules.append(self._pcont)

        Optimizer = functools.partial(
            tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
            wdpattern=self._c.weight_decay_pattern)
        self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
        self._disen_opt = Optimizer('disen', disen_modules, self._c.model_lr)
        self._decode_opt = Optimizer(
            'decode', [self._joint_decode], self._c.model_lr)
        self._disen_reward_opt = Optimizer(
            'disen_reward', [self._disen_reward], self._c.disen_reward_lr)
        self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
        self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
        self.train(next(self._dataset))

    def _scalar_summaries_disen(
            self, prior_dist_disen, post_dist_disen, likes_disen, div_disen,
            model_loss_disen, reward_loss_disen,
            model_disen_norm, reward_disen_norm):
        self._metrics_disen['model_grad_norm'].update_state(model_disen_norm)
        self._metrics_disen['reward_grad_norm'].update_state(reward_disen_norm)
        self._metrics_disen['prior_ent'].update_state(
            prior_dist_disen.entropy())
        self._metrics_disen['post_ent'].update_state(post_dist_disen.entropy())
        for name, logprob in likes_disen.items():
            self._metrics_disen[name + '_loss'].update_state(-logprob)
        self._metrics_disen['div'].update_state(div_disen)
        self._metrics_disen['model_loss'].update_state(model_loss_disen)
        self._metrics_disen['reward_loss'].update_state(
            reward_loss_disen)

    def _image_summaries(self, dynamics, decoder, data, embed, image_pred, tag='agent/openl'):
        truth = data['image'][:6] + 0.5
        recon = image_pred.mode()[:6]
        init, _ = dynamics.observe(embed[:6, :5], data['action'][:6, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = dynamics.imagine(data['action'][:6, 5:], init)
        if isinstance(decoder, models.ConvDecoderMask):
            openl, _ = decoder(dynamics.get_feat(prior))
            openl = openl.mode()
        else:
            openl = decoder(dynamics.get_feat(prior)).mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        tools.graph_summary(
            self._writer, tools.video_summary, self._step, tag, openl)

    def _image_summaries_joint(self, data, embed, embed_disen, image_pred_joint, mask_pred):
        truth = data['image'][:6] + 0.5
        recon_joint = image_pred_joint.mode()[:6]
        mask_pred = mask_pred[:6]

        init, _ = self._dynamics.observe(
            embed[:6, :5], data['action'][:6, :5])
        init_disen, _ = self._disen_dynamics.observe(
            embed_disen[:6, :5], data['action'][:6, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        init_disen = {k: v[:, -1] for k, v in init_disen.items()}
        prior = self._dynamics.imagine(
            data['action'][:6, 5:], init)
        prior_disen = self._disen_dynamics.imagine(
            data['action'][:6, 5:], init_disen)

        feat = self._dynamics.get_feat(prior)
        feat_disen = self._disen_dynamics.get_feat(prior_disen)
        openl, _, _, openl_mask = self._joint_decode(feat, feat_disen)

        openl = openl.mode()
        model = tf.concat([recon_joint[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = tf.concat([truth, model, error], 2)
        openl_mask = tf.concat([mask_pred[:, :5] + 0.5, openl_mask + 0.5], 1)

        tools.graph_summary(
            self._writer, tools.video_summary, self._step, 'joint/openl_joint', openl)
        tools.graph_summary(
            self._writer, tools.video_summary, self._step, 'mask/openl_mask', openl_mask)

    def image_summary_from_data(self, data):
        truth = data['image'][:6] + 0.5

        # main
        embed = self._encode(data)
        post, _ = self._dynamics.observe(
            embed[:6, :5], data['action'][:6, :5])
        feat = self._dynamics.get_feat(post)
        init = {k: v[:, -1] for k, v in post.items()}

        # disen
        embed_disen = self._disen_encode(data)
        post_disen, _ = self._disen_dynamics.observe(
            embed_disen[:6, :5], data['action'][:6, :5])
        feat_disen = self._disen_dynamics.get_feat(post_disen)
        init_disen = {k: v[:, -1] for k, v in post_disen.items()}

        # joint image pred
        recon_joint, recon_main, recon_disen, recon_mask = self._joint_decode(
            feat, feat_disen)
        recon_joint = recon_joint.mode()[:6]
        recon_main = recon_main.mode()[:6]
        recon_disen = recon_disen.mode()[:6]
        recon_mask = recon_mask[:6]

        prior = self._dynamics.imagine(
            data['action'][:6, 5:], init)
        prior_disen = self._disen_dynamics.imagine(
            data['action'][:6, 5:], init_disen)
        feat = self._dynamics.get_feat(prior)
        feat_disen = self._disen_dynamics.get_feat(prior_disen)

        openl_joint, openl_main, openl_disen, openl_mask = self._joint_decode(
            feat, feat_disen)
        openl_joint = openl_joint.mode()
        openl_main = openl_main.mode()
        openl_disen = openl_disen.mode()

        model_joint = tf.concat(
            [recon_joint[:, :5] + 0.5, openl_joint + 0.5], 1)
        error_joint = (model_joint - truth + 1) / 2
        model_main = tf.concat(
            [recon_main[:, :5] + 0.5, openl_main + 0.5], 1)
        model_disen = tf.concat(
            [recon_disen[:, :5] + 0.5, openl_disen + 0.5], 1)
        model_mask = tf.concat(
            [recon_mask[:, :5] + 0.5, openl_mask + 0.5], 1)

        output_joint = tf.concat(
            [truth, model_main, model_disen, model_joint, error_joint], 2)
        output_mask = model_mask

        tools.graph_summary(
            self._writer, tools.video_summary, self._step, 'summary/openl', output_joint)
        tools.graph_summary(
            self._writer, tools.video_summary, self._step, 'summary/openl_mask', output_mask)

    def _write_summaries(self):
        step = int(self._step.numpy())
        metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
        metrics_disen = [(k, float(v.result()))
                         for k, v in self._metrics_disen.items()]
        if self._last_log is not None:
            duration = time.time() - self._last_time
            self._last_time += duration
            metrics.append(('fps', (step - self._last_log) / duration))
        self._last_log = step
        [m.reset_states() for m in self._metrics.values()]
        [m.reset_states() for m in self._metrics_disen.values()]
        with (self._c.logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
        [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
        [tf.summary.scalar('disen/' + k, m) for k, m in metrics_disen]
        print('#'*30 + ' Main ' + '#'*30)
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        print('#'*30 + ' Disen ' + '#'*30)
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics_disen))
        self._writer.flush() 

class InverseDreamer(Dreamer):

    def __init__(self, config, datadir, actspace, writer):
        super().__init__(config, datadir, actspace, writer)

    def _train(self, data, log_images):
        with tf.GradientTape() as model_tape:
            embed = self._encode(data)
            post, prior = self._dynamics.observe(embed, data['action'])
            feat = self._dynamics.get_feat(post)

            action_pred = self._decode(feat)
            reward_pred = self._reward(feat)

            likes = tools.AttrDict()
            likes.action = tf.reduce_mean(
                action_pred.log_prob(data['action'][:, :-1]))
            likes.reward = tf.reduce_mean(
                reward_pred.log_prob(data['reward']))
            if self._c.pcont:
                pcont_pred = self._pcont(feat)
                pcont_target = self._c.discount * data['discount']
                likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
                likes.pcont *= self._c.pcont_scale

            prior_dist = self._dynamics.get_dist(prior)
            post_dist = self._dynamics.get_dist(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self._c.free_nats)

            model_loss = self._c.kl_scale * div - sum(likes.values())
            model_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as actor_tape:
            imag_feat = self._imagine_ahead(post)
            reward = self._reward(imag_feat).mode()
            if self._c.pcont:
                pcont = self._pcont(imag_feat).mean()
            else:
                pcont = self._c.discount * tf.ones_like(reward)
            value = self._value(imag_feat).mode()
            returns = tools.lambda_return(
                reward[:-1], value[:-1], pcont[:-1],
                bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            actor_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as value_tape:
            value_pred = self._value(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = - \
                tf.reduce_mean(discount * value_pred.log_prob(target))
            value_loss /= float(self._strategy.num_replicas_in_sync)

        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)

        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            if self._c.log_scalars:
                self._scalar_summaries(
                    data, feat, prior_dist, post_dist, likes, div,
                    model_loss, value_loss, actor_loss, model_norm, value_norm,
                    actor_norm)

    def _build_model(self):
        acts = dict(
            elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
            leaky_relu=tf.nn.leaky_relu)
        cnn_act = acts[self._c.cnn_act]
        act = acts[self._c.dense_act]
        self._encode = models.ConvEncoder(
            self._c.cnn_depth, cnn_act, self._c.image_size)
        self._dynamics = models.RSSM(
            self._c.stoch_size, self._c.deter_size, self._c.deter_size)
        self._decode = models.InverseDecoder(
            self._actdim, 4, self._c.num_units, act=act)
        self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
        if self._c.pcont:
            self._pcont = models.DenseDecoder(
                (), 3, self._c.num_units, 'binary', act=act)
        self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
        self._actor = models.ActionDecoder(
            self._actdim, 4, self._c.num_units, self._c.action_dist,
            init_std=self._c.action_init_std, act=act)
        model_modules = [self._encode, self._dynamics,
                         self._decode, self._reward]
        if self._c.pcont:
            model_modules.append(self._pcont)
        Optimizer = functools.partial(
            tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
            wdpattern=self._c.weight_decay_pattern)
        self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
        self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
        self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
        self.train(next(self._dataset))