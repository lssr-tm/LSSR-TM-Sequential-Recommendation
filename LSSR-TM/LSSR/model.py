import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import TransformerEncoder, MLP, Item_similarity_gating
from reclearn.models.losses import get_loss


class LSSR(Model):
    def __init__(self,
                 feature_columns,
                 seq_len=40,
                 blocks=1,
                 num_heads=1,
                 ffn_hidden_unit=128,
                 dnn_dropout=0.,
                 num_expert=3,
                 expert_units=(128, 128),
                 layer_norm_eps=1e-6,
                 use_l2norm=False,
                 loss_name="binary_cross_entropy",
                 gamma=0.5,
                 embed_reg=0.,
                 seed=None,
                 use_timestamp=True,
                 use_moe=True,
                 use_gate_unit=True,
                 use_abs_time=True,
                 use_st2lt_transfer=True,
                 shared_gate_init=0.5):
        super(LSSR, self).__init__()

        item_embed_dim = feature_columns['item']['embed_dim']
        user_embed_dim = feature_columns['user']['embed_dim']
        if user_embed_dim != item_embed_dim:
            raise ValueError(
                "User and item embedding dim must match. "
                f"Got user_embed_dim={user_embed_dim}, item_embed_dim={item_embed_dim}."
            )

        self.use_timestamp = use_timestamp
        self.use_moe = use_moe
        self.use_gate_unit = use_gate_unit
        self.use_abs_time = use_abs_time
        self.use_st2lt_transfer = use_st2lt_transfer
        self.seq_len = seq_len
        self.embed_dim = item_embed_dim

        self.fixed_avg_alpha = tf.constant(0.5, dtype=tf.float32)
        self.shared_gate_init = float(shared_gate_init)


        self.user_embedding = Embedding(
            input_dim=feature_columns['user']['feat_num'],
            input_length=1,
            output_dim=user_embed_dim,
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        self.item_embedding = Embedding(
            input_dim=feature_columns['item']['feat_num'],
            input_length=1,
            output_dim=item_embed_dim,
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        self.day_embedding = Embedding(
            input_dim=366,
            input_length=1,
            output_dim=item_embed_dim,
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        self.hour_embedding = Embedding(
            input_dim=25,
            input_length=1,
            output_dim=item_embed_dim,
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        self.pos_embedding = Embedding(
            input_dim=seq_len,
            input_length=1,
            output_dim=item_embed_dim,
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        self.dis_embedding = Embedding(
            input_dim=501,
            input_length=1,
            output_dim=item_embed_dim,
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        self.dropout = Dropout(dnn_dropout)


        self.encoder_layer = [
            TransformerEncoder(item_embed_dim, num_heads, ffn_hidden_unit,
                               dnn_dropout, layer_norm_eps)
            for _ in range(blocks)
        ]


        self.num_expert = num_expert
        self.expert_layers = [
            MLP(list(expert_units), activation='relu') for _ in range(self.num_expert)
        ]
        self.gate = MLP(list(expert_units), activation='relu')
        self.gate_dense = Dense(self.num_expert, activation=None)
        self.long_proj = Dense(item_embed_dim, activation=None)


        self.expert_layers_wo_st2lt = [
            MLP(list(expert_units), activation='relu') for _ in range(self.num_expert)
        ]
        self.gate_wo_st2lt = MLP(list(expert_units), activation='relu')
        self.gate_dense_wo_st2lt = Dense(self.num_expert, activation=None)
        self.long_proj_wo_st2lt = Dense(item_embed_dim, activation=None)


        self.long_no_moe_proj = Dense(item_embed_dim, activation=None)


        self.gating = Item_similarity_gating(dnn_dropout)

        self.use_l2norm = use_l2norm
        self.loss_name = loss_name
        self.gamma = gamma


        self._last_gate_weights = None
        self._last_gate_logits = None
        self._last_global_input = None
        self._last_local_info = None
        self._last_global_info = None
        self._last_fusion_weights = None

        tf.random.set_seed(seed)

    def get_shared_gate_alpha(self):
        return self.fixed_avg_alpha

    def call(self, inputs):

        seq_embed = self.item_embedding(inputs['click_seq'])
        mask = tf.expand_dims(
            tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32),
            axis=-1
        )

        pos_embed = tf.expand_dims(self.pos_embedding(tf.range(self.seq_len)), axis=0)
        seq_embed = seq_embed + pos_embed

        if self.use_timestamp:
            dis_embed = self.dis_embedding(inputs['dis_seq'])
            seq_embed = seq_embed + dis_embed

            if self.use_abs_time and ('day_seq' in inputs) and ('hour_seq' in inputs):
                day_embed = self.day_embedding(inputs['day_seq'])
                hour_embed = self.hour_embedding(inputs['hour_seq'])
                seq_embed = seq_embed + day_embed + hour_embed

        seq_embed = self.dropout(seq_embed)
        att_outputs = seq_embed * mask

        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])
            att_outputs = att_outputs * mask

        local_info = tf.squeeze(
            tf.slice(att_outputs, begin=[0, self.seq_len - 1, 0], size=[-1, 1, -1]),
            axis=1
        )


        user_embed = self.user_embedding(inputs['user'])


        self._last_gate_weights = None
        self._last_gate_logits = None
        self._last_global_input = None
        self._last_local_info = local_info
        self._last_global_info = None
        self._last_fusion_weights = None

        if self.use_st2lt_transfer:

            global_input = tf.concat([user_embed, local_info], axis=-1)

            if self.use_moe:
                expert_logits = self.gate_dense(self.gate(global_input))
                expert_gate = tf.nn.softmax(expert_logits, axis=-1)


                self._last_global_input = global_input
                self._last_gate_logits = expert_logits
                self._last_gate_weights = expert_gate

                multi_expert = tf.stack(
                    [expert(global_input) for expert in self.expert_layers],
                    axis=1
                )
                ex_out = tf.reduce_sum(
                    multi_expert * tf.expand_dims(expert_gate, axis=-1),
                    axis=1
                )
                global_info = self.long_proj(ex_out)
            else:
                self._last_global_input = global_input
                global_info = self.long_no_moe_proj(global_input)

        else:

            global_input = user_embed

            expert_logits = self.gate_dense_wo_st2lt(
                self.gate_wo_st2lt(global_input))
            expert_gate = tf.nn.softmax(expert_logits, axis=-1)


            self._last_global_input = global_input
            self._last_gate_logits = expert_logits
            self._last_gate_weights = expert_gate

            multi_expert = tf.stack(
                [expert(global_input) for expert in self.expert_layers_wo_st2lt],
                axis=1
            )
            ex_out = tf.reduce_sum(
                multi_expert * tf.expand_dims(expert_gate, axis=-1),
                axis=1
            )
            global_info = self.long_proj_wo_st2lt(ex_out)

        self._last_global_info = global_info


        pos_info = self.item_embedding(inputs['pos_item'])
        neg_info = self.item_embedding(inputs['neg_item'])
        cand_items = tf.concat([tf.expand_dims(pos_info, axis=1), neg_info], axis=1)
        cand_num = tf.shape(cand_items)[1]

        if self.use_gate_unit:
            seq_last_embed = tf.slice(seq_embed, begin=[0, self.seq_len - 1, 0],
                                      size=[-1, 1, -1])
            weights = self.gating([
                tf.tile(seq_last_embed, [1, cand_num, 1]),
                tf.tile(tf.expand_dims(local_info, axis=1), [1, cand_num, 1]),
                cand_items
            ])


            self._last_fusion_weights = weights

            user_info = (
                tf.expand_dims(local_info, axis=1) * weights
                + tf.expand_dims(global_info, axis=1) * (1.0 - weights)
            )
        else:
            avg_info = 0.5 * (local_info + global_info)
            user_info = tf.tile(tf.expand_dims(avg_info, axis=1), [1, cand_num, 1])

        if self.use_l2norm:
            pos_info = tf.math.l2_normalize(pos_info, axis=-1)
            neg_info = tf.math.l2_normalize(neg_info, axis=-1)
            user_info = tf.math.l2_normalize(user_info, axis=-1)

        pos_scores = tf.reduce_sum(
            user_info[:, 0:1, :] * tf.expand_dims(pos_info, axis=1),
            axis=-1
        )
        neg_scores = tf.reduce_sum(
            user_info[:, 1:, :] * neg_info,
            axis=-1
        )

        self.add_loss(get_loss(pos_scores, neg_scores, self.loss_name, self.gamma))
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        return logits

    def summary(self):
        inputs = {
            'click_seq': Input(shape=(self.seq_len,), dtype=tf.int32),
            'dis_seq':   Input(shape=(self.seq_len,), dtype=tf.int32),
            'day_seq':   Input(shape=(self.seq_len,), dtype=tf.int32),
            'hour_seq':  Input(shape=(self.seq_len,), dtype=tf.int32),
            'user':      Input(shape=(),              dtype=tf.int32),
            'pos_item':  Input(shape=(),              dtype=tf.int32),
            'neg_item':  Input(shape=(1,),            dtype=tf.int32),
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()


if __name__ == '__main__':
    feature_columns = {
        'item': {'feat_name': 'item', 'feat_num': 200, 'embed_dim': 8},
        'user': {'feat_name': 'user', 'feat_num': 100, 'embed_dim': 8},
    }
    model = LSSR(feature_columns=feature_columns,
                 expert_units=[128, 128],
                 num_expert=3)
    model.summary()
