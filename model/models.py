import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
import ipdb
from model.base_models import NeuralNetwork, ParallelNetworks
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from transformers import Trainer, TrainingArguments

import pytorch_lightning as pl


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_dims_out=conf.n_dims_out,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(
        self, n_dims, n_positions, n_embd=256, n_layer=12, n_head=8, n_dims_out=1
    ):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims

        # self.time_embedding = nn.Embedding(300, n_embd) #linear system
        self.time_embedding = nn.Embedding(560, n_embd)  # cartpole
        self.state_embedding = nn.Linear(n_dims, n_embd)
        self.control_embedding = nn.Linear(n_dims_out, n_embd)
        # self.switch_embedding = nn.Embedding(2, n_embd)  # add embedding for switching controller for cartpole
        # self.control_embedding = nn.Linear(2, n_embd)  # add label for switching controller for cartpole

        self.embed_ln = nn.LayerNorm(n_embd, eps=1e-5)

        self._backbone = GPT2Model(configuration)

        self._state_head = nn.Linear(n_embd, 4)  # 4/18/2025 cartpole
        # self._state_head = nn.Linear(n_embd, 2) #4/18/2025 pendulum and linear system
        # self._control_head = nn.Linear(n_embd, 1) # no label for switching controller
        self._control_head = nn.Linear(
            n_embd, 2
        )  # add label for switching controller for cartpole

    # @staticmethod
    # def _combine(xs_b, ys_b):
    #     """Interleaves the x's and the y's into a single sequence."""
    #     # print(f"xs_b shape: {xs_b.shape}")
    #     bsize, points, dim = xs_b.shape
    #     ys_b_wide = torch.cat(
    #         (
    #             ys_b.view(bsize, points, 1),
    #             torch.zeros(bsize, points, dim - 1, device=ys_b.device),
    #         ),
    #         axis=2,
    #     )
    #     zs = torch.stack((xs_b, ys_b_wide), dim=2)
    #     zs = zs.view(bsize, 2 * points, dim)
    #     return zs

    # @staticmethod
    # def _combine_ebonye(xs_b, ys_b):
    #     # """Interleaves the x's and the y's into a single sequence of embeddings."""
    #     # bsize, points, dim = states_embed.shape
    #     # zs = torch.stack((states_embed, control_embed), dim=2)
    #     # zs = zs.view(bsize, 2 * points, dim)

    #     """Interleaves the x's and the y's into a single sequence."""
    #     # print(f"xs_b shape: {xs_b.shape}")
    #     bsize, points, state_dim = xs_b.shape
    #     time_dim = 1

    #     ys_b_wide = ys_b.view(bsize, points, 1)

    #     # full_time_idx = torch.linspace(0, 1, steps=300, device=xs_b.device).view(1, 300, 1) # linear system
    #     full_time_idx = torch.linspace(0, 1, steps=560, device=xs_b.device).view(1, 560, 1)  # cartpole
    #     time_idx = full_time_idx[:, :points, :].repeat(bsize, 1, 1)  # Get time index for the current batch

    #     zeros_control_pad = torch.zeros(bsize, points, state_dim -1, device=ys_b.device)  # Padding for control dimension
    #     ys_b_padded = torch.cat((ys_b_wide, zeros_control_pad), dim=-1)

    #     zeros_time_pad = torch.zeros(bsize, points, state_dim - 1, device=xs_b.device)
    #     time_idx_padded = torch.cat((time_idx, zeros_time_pad), dim=-1)  # Padding for time index

    #     stacked = torch.stack((xs_b, ys_b_padded, time_idx_padded), dim=2)
    #     zs = stacked.view(bsize, 3*points, state_dim)

    #     # import pdb; pdb.set_trace()

    #     # xs_b_aug = torch.cat([xs_b, time_idx], dim=-1)  # Augment xs_b with time index
    #     # ys_b_wide = torch.cat(
    #     #     (
    #     #         ys_b.view(bsize, points, 1),
    #     #         torch.zeros(bsize, points, dim, device=ys_b.device),
    #     #     ),
    #     #     axis=2,
    #     # )
    #     # zs = torch.stack((xs_b_aug, ys_b_wide), dim=2)
    #     # zs = zs.view(bsize, 2 * points, dim+1)
    #     return zs

    def forward(self, xs, ys=None, inds=None, inf="no"):
        """
        Inf is for inference mode.
        """
        if inf == "yes":
            xs_b = xs.unsqueeze(0)
            ys_b = ys.unsqueeze(0)

            if xs_b.shape[1] == ys_b.shape[1] + 1:
                # padding = torch.zeros((ys_b.shape[0], 1), device=ys_b.device) # 5/25/2025 padding for ys no label
                padding = torch.zeros(
                    (ys_b.shape[0], 1, 2), device=xs.device
                )  # 5/25/2025 padding for ys with label
                ys_b = torch.cat((ys_b, padding), dim=1)

            # import pdb; pdb.set_trace()
            # zs = self._combine(xs_b, ys_b)
            # zs = self._combine_ebonye(xs_b, ys_b)

            # embeds = self._read_in(zs)

            ### old code

            # output = self._backbone(inputs_embeds=embeds).last_hidden_state
            # # prediction = self._read_out(output)
            # # print(f"output shape: {output.shape}")
            # # state_prediction = self._state_head(output)
            # # control_prediction = self._control_head(output)
            # # state_prediction = self._state_head(output[:, 1::2, :]) #4/11/2025 seems to be predicting state from control (wrong suggestion from gpt)
            # # control_prediction = self._control_head(output[:, ::2, :]) #4/11/2025 seems to be predicting control from state (wrong suggestion from gpt)

            # state_prediction = self._state_head(output[:, 1::3, :])
            # control_prediction = self._control_head(output[:, ::3, :])
            ### end old code

            # ### chatgpt suggestion
            # zs = torch.cat(
            #     [xs_b, ys_b.unsqueeze(-1)], dim=-1
            # )
            # embeds = self._read_in(zs)
            # output = self._backbone(inputs_embeds=embeds).last_hidden_state

            # state_prediction = self._state_head(output)
            # control_prediction = self._control_head(output)
            # ### end chatgpt suggestion

            #### embeddings for switching new code 6/24/2025
            # control_scalar = ys_b[..., 0].unsqueeze(-1)
            # switch_flag = ys_b[..., 1].long()

            states_embed = self.state_embedding(xs_b)
            controls_embed = self.control_embedding(ys_b)

            #### embeddings for switching new code 6/24/2025
            # controls_embed = self.control_embedding(control_scalar)
            timesteps = (
                torch.arange(0, xs_b.shape[1], device=xs_b.device)
                .unsqueeze(0)
                .repeat(xs_b.shape[0], 1)
            )
            time_embed = self.time_embedding(timesteps)

            #### embeddings for switching new code 6/24/2025
            # switch_embed = self.switch_embedding(switch_flag)

            states_embed = states_embed + time_embed
            # states_embed = states_embed + time_embed + switch_embed
            controls_embed = controls_embed + time_embed

            stacked_inputs = torch.stack((states_embed, controls_embed), dim=2)
            zs = stacked_inputs.view(xs_b.shape[0], 2 * xs_b.shape[1], -1)
            zs = self.embed_ln(
                zs
            )  # Apply layer normalization to the combined embeddings

            output = self._backbone(inputs_embeds=zs).last_hidden_state

            control_prediction = self._control_head(
                output[:, ::2, :]
            )  # Control predictions
            state_prediction = self._state_head(output[:, 1::2, :])  # State predictions
            return control_prediction, state_prediction

        # print(f"xs shape: {xs.shape}")
        # print(f"ys shape: {ys.shape}")

        if xs.shape[1] == ys.shape[1] + 1:
            # padding = torch.zeros((ys.shape[0], 1), device=ys.device)
            padding = torch.zeros(
                (ys.shape[0], 1, 2), device=xs.device
            )  # 5/25/2025 padding for xs
            ys = torch.cat((ys, padding), dim=1)

        # import pdb; pdb.set_trace()

        states_embed = self.state_embedding(xs)  # 5/25/2025 separate embeddings
        # controls_embed = self.control_embedding(ys.unsqueeze(-1)) # 5/25/2025 separate embeddings, no labels for control switching
        controls_embed = self.control_embedding(
            ys
        )  # 5/25/2025 separate embeddings, ys is already in the right shape

        timesteps = (
            torch.arange(0, xs.shape[1], device=xs.device)
            .unsqueeze(0)
            .repeat(xs.shape[0], 1)
        )
        time_embed = self.time_embedding(timesteps)

        states_embed = states_embed + time_embed
        # states_embed = states_embed + time_embed + switch_embed
        controls_embed = controls_embed + time_embed

        stacked_inputs = torch.stack((states_embed, controls_embed), dim=2)
        zs = stacked_inputs.view(xs.shape[0], 2 * xs.shape[1], -1)
        zs = self.embed_ln(zs)  # Apply layer normalization to the combined embeddings

        output = self._backbone(inputs_embeds=zs).last_hidden_state

        control_prediction = self._control_head(
            output[:, ::2, :]
        )  # Control predictions
        state_prediction = self._state_head(output[:, 1::2, :])  # State predictions

        # # zs = self._combine(xs, ys)
        # zs = self._combine_ebonye(xs, ys)

        # # zs = self._combine_ebonye(states_embed, controls_embed) # 5/25/2025 separate embeddings

        # # embeds = zs # 5/25/2025 separate embeddings

        # # import pdb; pdb.set_trace()
        # embeds = self._read_in(zs)
        # output = self._backbone(inputs_embeds=embeds).last_hidden_state

        # state_prediction = self._state_head(output[:, 1::3, :]) #4/11/2025 seems to be predicting state from control (wrong suggestion from gpt)
        # control_prediction = self._control_head(output[:, ::3, :]) #4/11/2025 seems to be predicting control from state (wrong suggestion from gpt)

        # state_prediction = self._state_head(output) #4/13/2025
        # control_prediction = self._control_head(output) #4/13/2025

        # prediction = self._read_out(output)

        # import pdb; pdb.set_trace()

        if ys is not None:
            # return prediction[:, ::2, 0]  ##### 4/10/2025 wanting to do loss on states and control
            # return prediction[:, ::2, 0], prediction[:, 1::2, 0]  ##### 4/10/2025 wanting to do loss on states and control
            return control_prediction, state_prediction
            # return control_prediction[:,::2], state_prediction[:, 1::2]  ##### 4/10/2025 wanting to do loss on states and control

        # if prediction.dim() == 3:
        #     return prediction[:, :, 0]
        # elif prediction.dim() == 2:
        #     return prediction[:, 0]
        else:
            raise ValueError("Unexpected number of dimensions in prediction tensor.")


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):
                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
