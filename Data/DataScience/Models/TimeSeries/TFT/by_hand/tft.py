import json

import pandas as pd
import pytorch_lightning as pl
import torch
from data.custom_dataset import TFTDataset
from data_formatters.base import DataTypes, InputTypes
from data_formatters.electricity import ElectricityFormatter
from models import (
    GateAddNormNetwork,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
)
from quantile_loss import NormalizedQuantileLossCalculator, QuantileLossCalculator
from torch import nn
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


class TemporalFusionTransformer(pl.LightningModule):
    def __init__(self, **kwargs):
        super(TemporalFusionTransformer, self).__init__()

        electricity = pd.read_csv("data/electricity.csv", index_col=0)
        data_formatter = ElectricityFormatter()
        train, valid, test = data_formatter.split_data(electricity)
        test = test.reset_index(drop=True)

        self.train_dataset = TFTDataset(train)
        self.valid_dataset = TFTDataset(valid)
        self.test_dataset = TFTDataset(test)

        # self.hparams = hparams
        self.save_hyperparameters(kwargs)

        self.name = self.__class__.__name__

        # Data parameters
        self.time_steps = int(
            self.hparams.total_time_steps
        )  # int(params['total_time_steps'])
        self.input_size = int(self.hparams.input_size)  # int(params['input_size'])
        self.output_size = int(self.hparams.output_size)  # int(params['output_size'])
        self.category_counts = json.loads(
            str("[369]")
        )  # json.loads(str(params['category_counts']))
        self.num_categorical_variables = len(self.category_counts)
        self.num_regular_variables = self.input_size - self.num_categorical_variables
        self.n_multiprocessing_workers = int(
            self.hparams.multiprocessing_workers
        )  # int(params['multiprocessing_workers'])

        # Relevant indices for TFT
        self._input_obs_loc = json.loads(
            str(self.hparams.input_obs_loc)
        )  # json.loads(str(params['input_obs_loc']))
        self._static_input_loc = json.loads(
            str(self.hparams.static_input_loc)
        )  # json.loads(str(params['static_input_loc']))
        self._known_regular_input_idx = json.loads(
            str(self.hparams.known_regular_inputs)
        )  # json.loads(str(params['known_regular_inputs']))
        self._known_categorical_input_idx = json.loads(
            str(self.hparams.known_categorical_inputs)
        )  # json.loads(str(params['known_categorical_inputs']))

        self.num_non_static_historical_inputs = self.get_historical_num_inputs()
        self.num_non_static_future_inputs = self.get_future_num_inputs()

        self.column_definition = [
            ("id", DataTypes.REAL_VALUED, InputTypes.ID),
            ("hours_from_start", DataTypes.REAL_VALUED, InputTypes.TIME),
            ("power_usage", DataTypes.REAL_VALUED, InputTypes.TARGET),
            ("hour", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("day_of_week", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("hours_from_start", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("categorical_id", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ]

        # Network params
        self.quantiles = [0.1, 0.5, 0.9]
        #         self.use_cudnn = use_cudnn  # Whether to use GPU optimised LSTM
        self.hidden_layer_size = int(
            self.hparams.hidden_layer_size
        )  # int(params['hidden_layer_size'])
        self.dropout_rate = float(
            self.hparams.dropout_rate
        )  # float(params['dropout_rate'])
        self.max_gradient_norm = float(
            self.hparams.max_gradient_norm
        )  # float(params['max_gradient_norm'])
        self.learning_rate = float(
            self.hparams.learning_rate
        )  # float(params['learning_rate'])
        self.minibatch_size = int(
            self.hparams.minibatch_size
        )  # int(params['minibatch_size'])
        self.num_epochs = int(self.hparams.num_epochs)  # int(params['num_epochs'])
        self.early_stopping_patience = int(
            self.hparams.early_stopping_patience
        )  # int(params['early_stopping_patience'])

        self.num_encoder_steps = int(
            self.hparams.num_encoder_steps
        )  # int(params['num_encoder_steps'])
        self.num_stacks = int(self.hparams.stack_size)  # int(params['stack_size'])
        self.num_heads = int(self.hparams.num_heads)  # int(params['num_heads'])

        # Serialisation options
        #         self._temp_folder = os.path.join(params['model_folder'], 'tmp')
        #         self.reset_temp_folder()

        # Extra components to store Tensorflow nodes for attention computations
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        print("*** {} params ***".format(self.name))
        for k in vars(self.hparams):
            print("# {} = {}".format(k, vars(self.hparams)[k]))

        self.train_criterion = QuantileLossCalculator(self.quantiles, self.output_size)
        self.test_criterion = NormalizedQuantileLossCalculator(
            self.quantiles, self.output_size
        )

        # Build model
        ## Build embeddings
        self.build_embeddings()

        ## Build Static Contex Networks
        self.build_static_context_networks()

        ## Building Variable Selection Networks
        self.build_variable_selection_networks()

        ## Build Lstm
        self.build_lstm()

        ## Build GLU for after lstm encoder decoder and layernorm
        self.build_post_lstm_gate_add_norm()

        ## Build Static Enrichment Layer
        self.build_static_enrichment()

        ## Building decoder multihead attention
        self.build_temporal_self_attention()

        ## Building positionwise decoder
        self.build_position_wise_feed_forward()

        ## Build output feed forward
        self.build_output_feed_forward()

        ## Initializing remaining weights
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if ("lstm" in name and "ih" in name) and "bias" not in name:
                # print(name)
                # print(p.shape)
                torch.nn.init.xavier_uniform_(p)
            #                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif ("lstm" in name and "hh" in name) and "bias" not in name:
                torch.nn.init.orthogonal_(p)

            elif "lstm" in name and "bias" in name:
                # print(name)
                # print(p.shape)
                torch.nn.init.zeros_(p)

    def get_historical_num_inputs(self):
        obs_inputs = [i for i in self._input_obs_loc]

        known_regular_inputs = [
            i for i in self._known_regular_input_idx if i not in self._static_input_loc
        ]

        known_categorical_inputs = [
            i
            for i in self._known_categorical_input_idx
            if i + self.num_regular_variables not in self._static_input_loc
        ]

        wired_embeddings = [
            i
            for i in range(self.num_categorical_variables)
            if i not in self._known_categorical_input_idx
            and i not in self._input_obs_loc
        ]

        unknown_inputs = [
            i
            for i in range(self.num_regular_variables)
            if i not in self._known_regular_input_idx and i not in self._input_obs_loc
        ]

        return len(
            obs_inputs
            + known_regular_inputs
            + known_categorical_inputs
            + wired_embeddings
            + unknown_inputs
        )

    def get_future_num_inputs(self):
        known_regular_inputs = [
            i for i in self._known_regular_input_idx if i not in self._static_input_loc
        ]

        known_categorical_inputs = [
            i
            for i in self._known_categorical_input_idx
            if i + self.num_regular_variables not in self._static_input_loc
        ]

        return len(known_regular_inputs + known_categorical_inputs)

    def build_embeddings(self):
        self.categorical_var_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.category_counts[i], self.hidden_layer_size)
                for i in range(self.num_categorical_variables)
            ]
        )

        self.regular_var_embeddings = nn.ModuleList(
            [
                nn.Linear(1, self.hidden_layer_size)
                for i in range(self.num_regular_variables)
            ]
        )

    def build_variable_selection_networks(self):
        self.static_vsn = VariableSelectionNetwork(
            hidden_layer_size=self.hidden_layer_size,
            input_size=self.hidden_layer_size * len(self._static_input_loc),
            output_size=len(self._static_input_loc),
            dropout_rate=self.dropout_rate,
        )

        self.temporal_historical_vsn = VariableSelectionNetwork(
            hidden_layer_size=self.hidden_layer_size,
            input_size=self.hidden_layer_size * self.num_non_static_historical_inputs,
            output_size=self.num_non_static_historical_inputs,
            dropout_rate=self.dropout_rate,
            additional_context=self.hidden_layer_size,
        )

        self.temporal_future_vsn = VariableSelectionNetwork(
            hidden_layer_size=self.hidden_layer_size,
            input_size=self.hidden_layer_size * self.num_non_static_future_inputs,
            output_size=self.num_non_static_future_inputs,
            dropout_rate=self.dropout_rate,
            additional_context=self.hidden_layer_size,
        )

    def build_static_context_networks(self):
        self.static_context_variable_selection_grn = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )

        self.static_context_enrichment_grn = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )

        self.static_context_state_h_grn = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )

        self.static_context_state_c_grn = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )

    def build_lstm(self):
        self.historical_lstm = nn.LSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            batch_first=True,
        )
        self.future_lstm = nn.LSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            batch_first=True,
        )

    def build_post_lstm_gate_add_norm(self):
        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(
            self.hidden_layer_size,
            self.hidden_layer_size,
            self.dropout_rate,
            activation=None,
        )

    def build_static_enrichment(self):
        self.static_enrichment = GatedResidualNetwork(
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            additional_context=self.hidden_layer_size,
        )

    def build_temporal_self_attention(self):
        self.self_attn_layer = InterpretableMultiHeadAttention(
            n_head=self.num_heads,
            d_model=self.hidden_layer_size,
            dropout=self.dropout_rate,
        )

        self.post_attn_gate_add_norm = GateAddNormNetwork(
            self.hidden_layer_size,
            self.hidden_layer_size,
            self.dropout_rate,
            activation=None,
        )

    def build_position_wise_feed_forward(self):
        self.GRN_positionwise = GatedResidualNetwork(
            self.hidden_layer_size, dropout_rate=self.dropout_rate
        )

        self.post_tfd_gate_add_norm = GateAddNormNetwork(
            self.hidden_layer_size,
            self.hidden_layer_size,
            self.dropout_rate,
            activation=None,
        )

    def build_output_feed_forward(self):
        self.output_feed_forward = torch.nn.Linear(
            self.hidden_layer_size, self.output_size * len(self.quantiles)
        )

    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.
        Args:
        self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1]
        bs = self_attn_inputs.shape[0]
        mask = torch.cumsum(torch.eye(len_s), 0)
        mask = mask.repeat(bs, 1, 1).to(torch.float32)

        return mask.to(DEVICE)

    def get_tft_embeddings(self, regular_inputs, categorical_inputs):
        # Static input
        if self._static_input_loc:
            static_regular_inputs = [
                self.regular_var_embeddings[i](regular_inputs[:, 0, i : i + 1])
                for i in range(self.num_regular_variables)
                if i in self._static_input_loc
            ]
            # print('static_regular_inputs')
            # print([print(emb.shape) for emb in static_regular_inputs])

            # static_categorical_inputs[0].shape: torch.Size([64, 160]) -> 160 = hidden_layer_size = d_model
            static_categorical_inputs = [
                self.categorical_var_embeddings[i](categorical_inputs[Ellipsis, i])[
                    :, 0, :
                ]
                for i in range(self.num_categorical_variables)
                if i + self.num_regular_variables in self._static_input_loc
            ]
            # print('static_categorical_inputs')
            # print([print(emb.shape) for emb in static_categorical_inputs])
            static_inputs = torch.stack(
                static_regular_inputs + static_categorical_inputs, axis=1
            )
        else:
            static_inputs = None

        # Target input
        obs_inputs = torch.stack(
            [
                self.regular_var_embeddings[i](regular_inputs[Ellipsis, i : i + 1])
                for i in self._input_obs_loc
            ],
            axis=-1,
        )

        # Observed (a prioir unknown) inputs
        wired_embeddings = []
        for i in range(self.num_categorical_variables):
            if (
                i not in self._known_categorical_input_idx
                and i not in self._input_obs_loc
            ):
                e = self.categorical_var_embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(self.num_regular_variables):
            if i not in self._known_regular_input_idx and i not in self._input_obs_loc:
                e = self.regular_var_embeddings[i](regular_inputs[Ellipsis, i : i + 1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = torch.stack(unknown_inputs + wired_embeddings, axis=-1)
        else:
            unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = [
            self.regular_var_embeddings[i](regular_inputs[Ellipsis, i : i + 1])
            for i in self._known_regular_input_idx
            if i not in self._static_input_loc
        ]
        # print('known_regular_inputs')
        # print([print(emb.shape) for emb in known_regular_inputs])

        known_categorical_inputs = [
            self.categorical_var_embeddings[i](categorical_inputs[Ellipsis, i])
            for i in self._known_categorical_input_idx
            if i + self.num_regular_variables not in self._static_input_loc
        ]
        # print('known_categorical_inputs')
        # print([print(emb.shape) for emb in known_categorical_inputs])

        known_combined_layer = torch.stack(
            known_regular_inputs + known_categorical_inputs, axis=-1
        )

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def forward(self, all_inputs):
        regular_inputs = all_inputs[:, :, : self.num_regular_variables].to(torch.float)
        # print('regular_inputs')
        # print(regular_inputs.shape)
        categorical_inputs = all_inputs[:, :, self.num_regular_variables :].to(
            torch.long
        )
        # print('categorical_inputs')
        # print(categorical_inputs.shape)

        (
            unknown_inputs,
            known_combined_layer,
            obs_inputs,
            static_inputs,
        ) = self.get_tft_embeddings(regular_inputs, categorical_inputs)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            # num_encoder_steps = k
            historical_inputs = torch.cat(
                [
                    unknown_inputs[:, : self.num_encoder_steps, :],
                    known_combined_layer[:, : self.num_encoder_steps, :],
                    obs_inputs[:, : self.num_encoder_steps, :],
                ],
                axis=-1,
            )

        else:
            historical_inputs = torch.cat(
                [
                    known_combined_layer[:, : self.num_encoder_steps, :],
                    obs_inputs[:, : self.num_encoder_steps, :],
                ],
                axis=-1,
            )

        # print('historical_inputs')
        # print(historical_inputs.shape)

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, self.num_encoder_steps :, :]

        static_encoder, sparse_weights = self.static_vsn(static_inputs)

        static_context_variable_selection = self.static_context_variable_selection_grn(
            static_encoder
        )  # Cs
        static_context_enrichment = self.static_context_enrichment_grn(
            static_encoder
        )  # Ce
        static_context_state_h = self.static_context_state_h_grn(static_encoder)  # Ch
        static_context_state_c = self.static_context_state_c_grn(static_encoder)  # Cc

        historical_features, historical_flags = self.temporal_historical_vsn(
            (historical_inputs, static_context_variable_selection)
        )  # historical_inputs = past_inputs

        future_features, future_flags = self.temporal_future_vsn(
            (future_inputs, static_context_variable_selection)
        )  # future_features = known_future_inputs

        history_lstm, (state_h, state_c) = self.historical_lstm(
            historical_features,
            (static_context_state_h.unsqueeze(0), static_context_state_c.unsqueeze(0)),
        )  # historical_lstm = lstm_encoder

        future_lstm, _ = self.future_lstm(
            future_features, (state_h, state_c)
        )  # future_lstm = lstm_decoder

        # Apply gated skip connection
        input_embeddings = torch.cat((historical_features, future_features), axis=1)

        lstm_layer = torch.cat((history_lstm, future_lstm), axis=1)

        temporal_feature_layer = self.post_seq_encoder_gate_add_norm(
            lstm_layer, input_embeddings
        )

        # Static enrichment layers
        expanded_static_context = static_context_enrichment.unsqueeze(1)

        enriched = self.static_enrichment(
            (temporal_feature_layer, expanded_static_context)
        )

        # Decoder self attention
        # self.mask = self.get_decoder_mask(enriched)
        x, self_att = self.self_attn_layer(
            enriched, enriched, enriched, mask=self.get_decoder_mask(enriched)
        )

        x = self.post_attn_gate_add_norm(x, enriched)

        # Nonlinear processing on outputs
        decoder = self.GRN_positionwise(x)

        # Final skip connection
        transformer_layer = self.post_tfd_gate_add_norm(decoder, temporal_feature_layer)

        outputs = self.output_feed_forward(
            transformer_layer[Ellipsis, self.num_encoder_steps :, :]
        )

        # ipdb.set_trace()

        return outputs

    def loss(self, y_hat, y):
        return self.train_criterion.apply(y_hat, y)

    def test_loss(self, y_hat, y):
        return self.test_criterion.apply(y_hat, y, self.quantiles[1])

    def training_step(self, batch, batch_nb):
        x, y, _ = batch

        # x.torch.Size: [batch_size, total_time_steps, time_series_vector_input(power_usage, hour, day_of_week, hours_from_start, categorical_id)]
        # y.torch.Size: [batch_size, decoder steps (total_time_steps - num_encoder_steps), target(power_usage)]
        # To check encoder and decoder steps, check get_fixed_params from electricity.py
        x = x.to(torch.float)  # torch.Size([64, 192, 5])
        y = y.to(torch.float)  # torch.Size([64, 24, 1])
        #         print('y')
        #         print(y.shape)
        y_hat = self.forward(x)
        #         print('y_hat')
        #         print(y_hat.shape)
        loss = self.loss(y_hat, torch.cat([y, y, y], dim=-1))
        # print(loss.shape)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y, _ = batch
        x = x.to(torch.float)
        y = y.to(torch.float)
        y_hat = self.forward(x)
        # print(y_hat.shape)
        # print(torch.cat([y, y, y], dim = -1).shape)
        loss = self.loss(y_hat, torch.cat([y, y, y], dim=-1))
        # print(loss)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y, _ = batch
        x = x.to(torch.float)
        y = y.to(torch.float)
        y_hat = self.forward(x)
        return {"test_loss": self.test_loss(y_hat[Ellipsis, 1], y[Ellipsis, 0])}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.train_dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1,
        )

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            self.valid_dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1,
        )

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            self.test_dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1,
        )
