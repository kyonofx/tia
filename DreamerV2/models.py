import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

import networks
import tools


class WorldModel(tools.Module):

    def __init__(self, step, config):
        self._step = step
        self._config = config
        channels = (1 if config.atari_grayscale else 3)
        shape = config.size + (channels,)

        ########
        # Main #
        ########
        self.encoder = networks.ConvEncoder(
            config.cnn_depth, config.act, config.encoder_kernels)
        self.dynamics = networks.RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_input_layers, config.dyn_output_layers, config.dyn_shared,
            config.dyn_discrete, config.act, config.dyn_mean_act,
            config.dyn_std_act, config.dyn_min_std, config.dyn_cell)
        self.heads = {}
        self.heads['reward'] = networks.DenseHead(
            [], config.reward_layers, config.units, config.act)
        if config.pred_discount:
            self.heads['discount'] = networks.DenseHead(
                [], config.discount_layers, config.units, config.act, dist='binary')
        self._model_opt = tools.Optimizer(
            'model', config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt)
        self._scales = dict(
            reward=config.reward_scale, discount=config.discount_scale)

        #########
        # Disen #
        #########
        self.disen_encoder = networks.ConvEncoder(
            config.disen_cnn_depth, config.act, config.encoder_kernels)
        self.disen_dynamics = networks.RSSM(
            config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
            config.dyn_input_layers, config.dyn_output_layers, config.dyn_shared,
            config.dyn_discrete, config.act, config.dyn_mean_act,
            config.dyn_std_act, config.dyn_min_std, config.dyn_cell)

        self.disen_heads = {}
        self.disen_heads['reward'] = networks.DenseHead(
            [], config.reward_layers, config.units, config.act)
        if config.pred_discount:
            self.disen_heads['discount'] = networks.DenseHead(
                [], config.discount_layers, config.units, config.act, dist='binary')

        self._disen_model_opt = tools.Optimizer(
            'disen', config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt)

        self._disen_heads_opt = {}
        self._disen_heads_opt['reward'] = tools.Optimizer(
            'disen_reward', config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt)
        if config.pred_discount:
            self._disen_heads_opt['discount'] = tools.Optimizer(
                'disen_pcont', config.model_lr, config.opt_eps, config.grad_clip,
                config.weight_decay, opt=config.opt)

        # negative signs for reward/discount here
        self._disen_scales = dict(disen_only=config.disen_only_scale,
                                  reward=-config.disen_reward_scale, discount=-config.disen_discount_scale)

        self.disen_only_image_head = networks.ConvDecoder(
            config.disen_cnn_depth, config.act, shape, config.decoder_kernels,
            config.decoder_thin)

        ################
        # Joint Decode #
        ################
        self.image_head = networks.ConvDecoderMask(
            config.cnn_depth, config.act, shape, config.decoder_kernels,
            config.decoder_thin)
        self.disen_image_head = networks.ConvDecoderMask(
            config.disen_cnn_depth, config.act, shape, config.decoder_kernels,
            config.decoder_thin)
        self.joint_image_head = networks.ConvDecoderMaskEnsemble(
            self.image_head, self.disen_image_head
        )

    def train(self, data):
        data = self.preprocess(data)
        with tf.GradientTape() as model_tape, tf.GradientTape() as disen_tape:

            # kl schedule
            kl_balance = tools.schedule(self._config.kl_balance, self._step)
            kl_free = tools.schedule(self._config.kl_free, self._step)
            kl_scale = tools.schedule(self._config.kl_scale, self._step)

            # Main
            embed = self.encoder(data)
            post, prior = self.dynamics.observe(embed, data['action'])
            kl_loss, kl_value = self.dynamics.kl_loss(
                post, prior, kl_balance, kl_free, kl_scale)
            feat = self.dynamics.get_feat(post)
            likes = {}
            for name, head in self.heads.items():
                grad_head = (name in self._config.grad_heads)
                inp = feat if grad_head else tf.stop_gradient(feat)
                pred = head(inp, tf.float32)
                like = pred.log_prob(tf.cast(data[name], tf.float32))
                likes[name] = tf.reduce_mean(
                    like) * self._scales.get(name, 1.0)

            # Disen
            embed_disen = self.disen_encoder(data)
            post_disen, prior_disen = self.disen_dynamics.observe(
                embed_disen, data['action'])
            kl_loss_disen, kl_value_disen = self.dynamics.kl_loss(
                post_disen, prior_disen, kl_balance, kl_free, kl_scale)
            feat_disen = self.disen_dynamics.get_feat(post_disen)

            # Optimize disen reward/pcont till optimal
            disen_metrics = dict(reward={}, discount={})
            loss_disen = dict(reward=None, discount=None)
            for _ in range(self._config.num_reward_opt_iters):
                with tf.GradientTape() as disen_reward_tape, tf.GradientTape() as disen_pcont_tape:
                    disen_gradient_tapes = dict(
                        reward=disen_reward_tape, discount=disen_pcont_tape)
                    for name, head in self.disen_heads.items():
                        pred_disen = head(
                            tf.stop_gradient(feat_disen), tf.float32)
                        loss_disen[name] = -tf.reduce_mean(pred_disen.log_prob(
                            tf.cast(data[name], tf.float32)))
                for name, head in self.disen_heads.items():
                    disen_metrics[name] = self._disen_heads_opt[name](
                        disen_gradient_tapes[name], loss_disen[name], [head], prefix='disen_neg')

            # Compute likes for disen model (including negative gradients)
            likes_disen = {}
            for name, head in self.disen_heads.items():
                pred_disen = head(feat_disen, tf.float32)
                like_disen = pred_disen.log_prob(
                    tf.cast(data[name], tf.float32))
                likes_disen[name] = tf.reduce_mean(
                    like_disen) * self._disen_scales.get(name, -1.0)
            disen_only_image_pred = self.disen_only_image_head(
                feat_disen, tf.float32)
            disen_only_image_like = tf.reduce_mean(disen_only_image_pred.log_prob(
                tf.cast(data['image'], tf.float32))) * self._disen_scales.get('disen_only', 1.0)
            likes_disen['disen_only'] = disen_only_image_like

            # Joint decode
            image_pred_joint, _, _, _ = self.joint_image_head(
                feat, feat_disen, tf.float32)
            image_like = tf.reduce_mean(image_pred_joint.log_prob(
                tf.cast(data['image'], tf.float32)))
            likes['image'] = image_like
            likes_disen['image'] = image_like

            # Compute loss
            model_loss = kl_loss - sum(likes.values())
            disen_loss = kl_loss_disen - sum(likes_disen.values())

        model_parts = [self.encoder, self.dynamics,
                       self.joint_image_head] + list(self.heads.values())
        disen_parts = [self.disen_encoder, self.disen_dynamics,
                       self.joint_image_head, self.disen_only_image_head]

        metrics = self._model_opt(
            model_tape, model_loss, model_parts, prefix='main')
        disen_model_metrics = self._disen_model_opt(
            disen_tape, disen_loss, disen_parts, prefix='disen')

        metrics['kl_balance'] = kl_balance
        metrics['kl_free'] = kl_free
        metrics['kl_scale'] = kl_scale
        metrics.update({f'{name}_loss': -like for name,
                        like in likes.items()})

        metrics['disen/disen_only_image_loss'] = -disen_only_image_like
        metrics['disen/disen_reward_loss'] = -likes_disen['reward'] / \
            self._disen_scales.get('reward', -1.0)
        metrics['disen/disen_discount_loss'] = -likes_disen['discount'] / \
            self._disen_scales.get('discount', -1.0)

        metrics['kl'] = tf.reduce_mean(kl_value)
        metrics['prior_ent'] = self.dynamics.get_dist(prior).entropy()
        metrics['post_ent'] = self.dynamics.get_dist(post).entropy()
        metrics['disen/kl'] = tf.reduce_mean(kl_value_disen)
        metrics['disen/prior_ent'] = self.dynamics.get_dist(
            prior_disen).entropy()
        metrics['disen/post_ent'] = self.dynamics.get_dist(
            post_disen).entropy()

        metrics.update(
            {f'{key}': value for key, value in disen_metrics['reward'].items()})
        metrics.update(
            {f'{key}': value for key, value in disen_metrics['discount'].items()})
        metrics.update(
            {f'{key}': value for key, value in disen_model_metrics.items()})

        return embed, post, feat, kl_value, metrics

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()
        obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
        obs['reward'] = getattr(tf, self._config.clip_rewards)(obs['reward'])
        if 'discount' in obs:
            obs['discount'] *= self._config.discount
        for key, value in obs.items():
            if tf.dtypes.as_dtype(value.dtype) in (
                    tf.float16, tf.float32, tf.float64):
                obs[key] = tf.cast(value, dtype)
        return obs

    @tf.function
    def video_pred(self, data):
        data = self.preprocess(data)
        truth = data['image'][:6] + 0.5

        embed = self.encoder(data)
        embed_disen = self.disen_encoder(data)
        states, _ = self.dynamics.observe(
            embed[:6, :5], data['action'][:6, :5])
        states_disen, _ = self.disen_dynamics.observe(
            embed_disen[:6, :5], data['action'][:6, :5])
        feats = self.dynamics.get_feat(states)
        feats_disen = self.disen_dynamics.get_feat(states_disen)
        recon_joint, recon_main, recon_disen, recon_mask = self.joint_image_head(
            feats, feats_disen)
        recon_joint = recon_joint.mode()[:6]
        recon_main = recon_main.mode()[:6]
        recon_disen = recon_disen.mode()[:6]
        recon_mask = recon_mask[:6]

        init = {k: v[:, -1] for k, v in states.items()}
        init_disen = {k: v[:, -1] for k, v in states_disen.items()}
        prior = self.dynamics.imagine(
            data['action'][:6, 5:], init)
        prior_disen = self.disen_dynamics.imagine(
            data['action'][:6, 5:], init_disen)
        _feats = self.dynamics.get_feat(prior)
        _feats_disen = self.disen_dynamics.get_feat(prior_disen)
        openl_joint, openl_main, openl_disen, openl_mask = self.joint_image_head(
            _feats, _feats_disen)
        openl_joint = openl_joint.mode()
        openl_main = openl_main.mode()
        openl_disen = openl_disen.mode()

        model_joint = tf.concat(
            [recon_joint[:, :5] + 0.5, openl_joint + 0.5], 1)
        error_joint = (model_joint - truth + 1) / 2
        model_main = tf.concat(
            [recon_main[:, :5] + 0.5, openl_main + 0.5], 1)
        error_main = (model_main - truth + 1) / 2
        model_disen = tf.concat(
            [recon_disen[:, :5] + 0.5, openl_disen + 0.5], 1)
        error_disen = (model_disen - truth + 1) / 2
        model_mask = tf.concat(
            [recon_mask[:, :5] + 0.5, openl_mask + 0.5], 1)

        output_joint = tf.concat([truth, model_joint, error_joint], 2)
        output_main = tf.concat([truth, model_main, error_main], 2)
        output_disen = tf.concat([truth, model_disen, error_disen], 2)
        output_mask = model_mask

        return output_joint, output_main, output_disen, output_mask


class ImagBehavior(tools.Module):

    def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        self.actor = networks.ActionHead(
            config.num_actions, config.actor_layers, config.units, config.act,
            config.actor_dist, config.actor_init_std, config.actor_min_std,
            config.actor_dist, config.actor_temp, config.actor_outscale)
        self.value = networks.DenseHead(
            [], config.value_layers, config.units, config.act,
            config.value_head)
        if config.slow_value_target or config.slow_actor_target:
            self._slow_value = networks.DenseHead(
                [], config.value_layers, config.units, config.act)
            self._updates = tf.Variable(0, tf.int64)
        kw = dict(wd=config.weight_decay, opt=config.opt)
        self._actor_opt = tools.Optimizer(
            'actor', config.actor_lr, config.opt_eps, config.actor_grad_clip, **kw)
        self._value_opt = tools.Optimizer(
            'value', config.value_lr, config.opt_eps, config.value_grad_clip, **kw)

    def train(
            self, start, objective=None, imagine=None, tape=None, repeats=None):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}
        with (tape or tf.GradientTape()) as actor_tape:
            assert bool(objective) != bool(imagine)
            if objective:
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, repeats)
                reward = objective(imag_feat, imag_state, imag_action)
            else:
                imag_feat, imag_state, imag_action, reward = imagine(start)
            actor_ent = self.actor(imag_feat, tf.float32).entropy()
            state_ent = self._world_model.dynamics.get_dist(
                imag_state, tf.float32).entropy()
            target, weights = self._compute_target(
                imag_feat, reward, actor_ent, state_ent,
                self._config.slow_actor_target)
            actor_loss, mets = self._compute_actor_loss(
                imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
                weights)
            metrics.update(mets)
        if self._config.slow_value_target != self._config.slow_actor_target:
            target, weights = self._compute_target(
                imag_feat, reward, actor_ent, state_ent,
                self._config.slow_value_target)
        with tf.GradientTape() as value_tape:
            value = self.value(imag_feat, tf.float32)[:-1]
            value_loss = -value.log_prob(tf.stop_gradient(target))
            if self._config.value_decay:
                value_loss += self._config.value_decay * value.mode()
            value_loss = tf.reduce_mean(weights[:-1] * value_loss)
        metrics['reward_mean'] = tf.reduce_mean(reward)
        metrics['reward_std'] = tf.math.reduce_std(reward)
        metrics['actor_ent'] = tf.reduce_mean(actor_ent)
        metrics.update(self._actor_opt(actor_tape, actor_loss, [self.actor]))
        metrics.update(self._value_opt(value_tape, value_loss, [self.value]))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            start = {k: tf.repeat(v, repeats, axis=1)
                     for k, v in start.items()}

        def flatten(x): return tf.reshape(x, [-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = tf.stop_gradient(feat) if self._stop_grad_actor else feat
            action = policy(inp).sample()
            succ = dynamics.img_step(
                state, action, sample=self._config.imag_sample)
            return succ, feat, action
        feat = 0 * dynamics.get_feat(start)
        action = policy(feat).mode()
        succ, feats, actions = tools.static_scan(
            step, tf.range(horizon), (start, feat, action))
        states = {k: tf.concat([
            start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            def unfold(tensor):
                s = tensor.shape
                return tf.reshape(tensor, [s[0], s[1] // repeats, repeats] + s[2:])
            states, feats, actions = tf.nest.map_structure(
                unfold, (states, feats, actions))
        return feats, states, actions

    def _compute_target(self, imag_feat, reward, actor_ent, state_ent, slow):
        reward = tf.cast(reward, tf.float32)
        if 'discount' in self._world_model.heads:
            discount = self._world_model.heads['discount'](
                imag_feat, tf.float32).mean()
        else:
            discount = self._config.discount * tf.ones_like(reward)
        if self._config.future_entropy and tf.greater(
                self._config.actor_entropy(), 0):
            reward += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and tf.greater(
                self._config.actor_state_entropy(), 0):
            reward += self._config.actor_state_entropy() * state_ent
        if slow:
            value = self._slow_value(imag_feat, tf.float32).mode()
        else:
            value = self.value(imag_feat, tf.float32).mode()
        target = tools.lambda_return(
            reward[:-1], value[:-1], discount[:-1],
            bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
        weights = tf.stop_gradient(tf.math.cumprod(tf.concat(
            [tf.ones_like(discount[:1]), discount[:-1]], 0), 0))
        return target, weights

    def _compute_actor_loss(
            self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
            weights):
        metrics = {}
        inp = tf.stop_gradient(
            imag_feat) if self._stop_grad_actor else imag_feat
        policy = self.actor(inp, tf.float32)
        actor_ent = policy.entropy()
        if self._config.imag_gradient == 'dynamics':
            actor_target = target
        elif self._config.imag_gradient == 'reinforce':
            imag_action = tf.cast(imag_action, tf.float32)
            actor_target = policy.log_prob(imag_action)[:-1] * tf.stop_gradient(
                target - self.value(imag_feat[:-1], tf.float32).mode())
        elif self._config.imag_gradient == 'both':
            imag_action = tf.cast(imag_action, tf.float32)
            actor_target = policy.log_prob(imag_action)[:-1] * tf.stop_gradient(
                target - self.value(imag_feat[:-1], tf.float32).mode())
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics['imag_gradient_mix'] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and tf.greater(
                self._config.actor_entropy(), 0):
            actor_target += self._config.actor_entropy() * actor_ent[:-1]
        if not self._config.future_entropy and tf.greater(
                self._config.actor_state_entropy(), 0):
            actor_target += self._config.actor_state_entropy() * state_ent[:-1]
        actor_loss = -tf.reduce_mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target or self._config.slow_actor_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.variables, self._slow_value.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)
