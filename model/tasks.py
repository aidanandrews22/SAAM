import math
import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def mean_squared_error_excluded(ys_pred, ys):
    ys_pred = ys_pred[:, 1:]
    ys = ys[:, 1:]
    return (ys - ys_pred).square().mean()

def mean_absolute_error(ys_pred, ys):
    return (ys - ys_pred).abs().mean()


def log_mean_squared_error_excluded(ys_pred, ys, epsilon=1e-8):
    ys_pred = ys_pred[:, 1:]
    ys = ys[:, 1:]

    log_pred = torch.log(ys_pred + epsilon)
    log_true = torch.log(ys + epsilon)

    return (log_pred - log_true).square().mean()

def mean_squared_error_relative_weighted_excluded(ys_pred, ys):
    ys_pred = ys_pred[:, 1:]
    ys = ys[:, 1:]

    relative_errors = ((ys - ys_pred).abs() / ys.abs().clamp(min=1e-8))  
    weights = 1 + relative_errors  
    
    weighted_mse = ((ys - ys_pred).square() * weights).mean()
    
    return weighted_mse


def weighted_mse_dynamic_excluded(ys_pred, ys_true, base_weight=10, increment=10, small_value_threshold=1e-3, eps=1e-6):
    ys_pred = ys_pred[:, 1:]
    ys_true = ys_true[:, 1:]

    squared_error = (ys_pred - ys_true).square()

    small_value_threshold_tensor = torch.tensor(
        small_value_threshold, dtype=ys_true.dtype, device=ys_true.device
    )
    import ipdb
    ipdb.set_trace()
    abs_ys_true = ys_true.abs() + eps 
    weights = torch.where(
        abs_ys_true < small_value_threshold_tensor,  
        base_weight + increment * (-torch.log10(abs_ys_true) - torch.log10(small_value_threshold_tensor)),
        torch.tensor(1.0, dtype=ys_true.dtype, device=ys_true.device) 
    )

    weighted_mse = (squared_error * weights).mean()

    return weighted_mse


'''mean_squared_logarithmic_error'''
# def mean_squared_error(ys_pred, ys, epsilon=1e-15): 

#     ys_pred = torch.clamp(ys_pred, min=0)
#     ys = torch.clamp(ys, min=0)
    
#     log_ys_pred = torch.log(ys_pred + epsilon)
#     log_ys = torch.log(ys + epsilon)
    
#     msle = (log_ys - log_ys_pred).pow(2).mean()
    
#     return msle



# def mean_squared_error(ys_pred, ys):
#     squared_diff = (ys - ys_pred).square()
    
#     squared_diff_flat = squared_diff.view(-1)
    
#     min_diff = squared_diff_flat.min()
#     max_diff = squared_diff_flat.max()
    
#     denom = max_diff - min_diff
#     if denom < 1e-12:
#         normalized_diff = torch.ones_like(squared_diff_flat)
#     else:
#         normalized_diff = (squared_diff_flat - min_diff) / denom
    
#     mse_normalized = normalized_diff.mean()
    
#     return mse_normalized


'''smape_loss'''
# def mean_squared_error(y_pred, y_true, epsilon=1e-8):
#     y_true = y_true[:, -1]
#     y_pred = y_pred[:, -1]
#     numerator = torch.abs(y_pred - y_true)
#     denominator = torch.abs(y_pred) + torch.abs(y_true)
#     denominator = torch.clamp(denominator, min=epsilon)  
#     loss = 2.0 * numerator / denominator
#     return torch.mean(loss)


'''log_cosh_loss'''
# def mean_squared_error(y_pred, y_true):
#     y_true = y_true[:, -1]
#     y_pred = y_pred[:, -1]
#     diff = y_pred - y_true

#     loss = torch.log(torch.cosh(diff + 1e-12))  
#     return torch.mean(loss)

"""relative_error"""
# def mean_squared_error(ys_pred, ys):
#     ys_pred_last = ys_pred[:, -1]
#     ys_last = ys[:, -1]
#     relative_errors = ((ys_last - ys_pred_last).abs() / ys_last.abs()).mean()

#     return relative_errors


""" mean_squared_error_normalized"""
# def mean_squared_error(y_pred, y_true):
#     epsilon = 1e-8
#     mse = (y_true - y_pred).square().mean()
#     mse_normalized = mse / (y_true.abs().mean() + epsilon)
#     return mse_normalized


""" log_scaled_mse """
# def mean_squared_error(ys_pred, ys):
#     epsilon = 1e-10
#     log_ys_pred = torch.log(ys_pred.abs() + epsilon)
#     log_ys = torch.log(ys.abs() + epsilon)

#     mse = (log_ys - log_ys_pred).square().mean()
#     return mse

"""stable log cosh"""
# def mean_squared_error(y_true, y_pred):
#     diff = y_pred - y_true
#     return torch.mean(diff + torch.nn.functional.softplus(-2 * diff) - torch.log(torch.tensor(2.0)))

"""hybrid loss mse or log cosh"""
# def mean_squared_error(y_true, y_pred, threshold=1.0):
#     diff = y_pred - y_true

#     large_error_mask = torch.abs(diff) > threshold
#     small_error_mask = ~large_error_mask

#     mse_loss = torch.mean(torch.square(diff[large_error_mask])) if large_error_mask.any() else 0.0
#     log_cosh_loss = torch.mean(torch.log(torch.cosh(diff[small_error_mask]))) if small_error_mask.any() else 0.0

#     total_loss = mse_loss + log_cosh_loss
#     return total_loss


# def mean_squared_error(y_pred, y_true, epsilon=1e-7):
#     diff = y_pred - y_true
#     scaling_factor = torch.max(torch.abs(diff)) + epsilon
#     normalized_diff = diff / scaling_factor
#     loss = torch.log(torch.cosh(normalized_diff))
#     scaled_loss = scaling_factor * loss

#     return torch.mean(scaled_loss)

# def mean_squared_error(y_pred, y_true):
#     diff = y_pred - y_true
#     loss = torch.log(torch.cosh(diff + 1e-12)) 
#     return torch.mean(loss)


""" weighted_mse_acrossbatch """
# def mean_squared_error(ys_pred, ys, start_weight=0.1, increase_rate=1.05):
#     num_points = ys.size(1)  

#     weights = torch.tensor(
#         [start_weight * (increase_rate ** i) for i in range(num_points)],
#         device=ys.device
#     ).unsqueeze(0)  

#     mse = ((ys - ys_pred).square() * weights).mean()

#     return mse

""" weighted_mse_persample"""
# def mean_squared_error(ys_pred, ys, start_weight=0.1, increase_rate=1.05):
#     batch_size, num_points = ys.size()  
#     weights = torch.tensor(
#         [[start_weight * (increase_rate ** i) for i in range(num_points)] for _ in range(batch_size)],
#         device=ys.device
#     )
#     mse = ((ys - ys_pred).square() * weights).mean()
#     return mse

"""log_mse_persample"""
# def mean_squared_error(ys_pred, ys, eps=1e-6):
#     log_ys = torch.log(ys + eps)
#     log_ys_pred = torch.log(ys_pred + eps)
    
#     log_mse = (log_ys - log_ys_pred).square().mean()
#     return log_mse

"""normalized_mse_persample """
# def mean_squared_error(ys_pred, ys, eps=1e-6):
#     nmse = ((ys - ys_pred).square() / (ys.square() + eps)).mean()
#     return nmse

"""mse_persampleProportionaltoMagnitude"""
# def mean_squared_error(ys_pred, ys, eps=1e-6):
#     weights = 1 / (ys.square() + eps)
    
#     weighted_mse = ((ys - ys_pred).square() * weights).mean()
#     return weighted_mse



"""log_cosh_with_l2_normalization"""
# def mean_squared_error(ys_pred, ys, eps=1e-6):

#     def l2_normalize(tensor, eps):

#         norm_vals = torch.norm(tensor, p=2, dim=1, keepdim=True) + eps  
#         return tensor / norm_vals

#     ys_normalized = l2_normalize(ys, eps)
#     ys_pred_normalized = l2_normalize(ys_pred, eps)

#     diff = ys_pred_normalized - ys_normalized
#     log_cosh_loss = torch.log(torch.cosh(diff)).mean() 

#     return log_cosh_loss



""" log_cosh_with_mse_fallback"""
# def mean_squared_error(ys_pred, ys, eps=1e-6):

#     diff = ys_pred - ys
#     try:
#         log_cosh_loss = torch.log(torch.cosh(diff + eps)).mean()
#     except Exception:
#         log_cosh_loss = torch.tensor(float('nan')) 

#     if torch.isnan(log_cosh_loss) or log_cosh_loss.item() == 0:
#         mse_loss = (diff.square().mean())
#         return mse_loss

#     return log_cosh_loss



""" mse_with_fallback_normalization"""
# def mean_squared_error(ys_pred, ys, eps=1e-6, min_points_for_robust=3):

#     def robust_normalize(tensor, eps):
#         num_points = tensor.size(1)
#         if num_points < min_points_for_robust: 
#             return tensor 

#         median_vals = tensor.median(dim=1, keepdim=True).values

#         k1 = max(1, int(0.25 * num_points)) 
#         k3 = max(1, int(0.75 * num_points))

#         q1 = tensor.kthvalue(k1, dim=1).values.unsqueeze(1)  
#         q3 = tensor.kthvalue(k3, dim=1).values.unsqueeze(1)  

#         iqr = torch.clamp(q3 - q1, min=eps)

#         return (tensor - median_vals) / iqr

#     num_points = ys.size(1)

#     if num_points < min_points_for_robust:
#         mse_loss = ((ys_pred - ys).square()).mean()
#         return mse_loss

#     ys_normalized = robust_normalize(ys, eps)
#     ys_pred_normalized = robust_normalize(ys_pred, eps)

#     mse_loss = ((ys_pred_normalized - ys_normalized).square()).mean()
#     return mse_loss


"""weighted_mse_for_small_values"""
# def mean_squared_error(ys_pred, ys_true, small_value_threshold=1e-1, small_value_weight=10, eps=1e-6):
#     squared_error = (ys_pred - ys_true).square()

#     dtype = ys_true.dtype 
#     weights = torch.where(
#         ys_true.abs() < small_value_threshold, 
#         torch.tensor(small_value_weight, dtype=dtype, device=ys_true.device), 
#         torch.tensor(1.0, dtype=dtype, device=ys_true.device)  
#     )

#     weighted_mse = (squared_error * weights).mean()

#     return weighted_mse


"""mse_with_mae_adjustment """
# def mean_squared_error(ys_pred, ys_true, mae_weight=0.1):
#     mse_loss = ((ys_pred - ys_true).square()).mean()

#     mae_loss = (ys_pred - ys_true).abs().mean()

#     hybrid_loss = mse_loss + mae_weight * mae_loss

#     return hybrid_loss




"""log_cosh_with_fallback """
# def mean_squared_error(ys_pred, ys, eps=1e-6):
#     try:
#         diff = ys_pred - ys
#         log_cosh_loss = torch.log(torch.cosh(diff + eps)).mean()

#         if torch.isnan(log_cosh_loss) or torch.isinf(log_cosh_loss) or log_cosh_loss == 0:
#             raise ValueError("Log-Cosh loss produced NaN, Inf, or 0.")

#         return log_cosh_loss

#     except (ValueError, RuntimeError):
#         mse_loss = ((ys_pred - ys).square()).mean()
#         return mse_loss


# def normalized_mse(ys_pred, ys, eps=1e-6):
#     nmse = ((ys - ys_pred).square() / (ys.square() + eps)).mean()
#     return nmse


""" normalized_range_mse """
# def mean_squared_error(ys_pred, ys):
#     range_vals = ys.max(dim=1)[0] - ys.min(dim=1)[0]
#     normalized_error = ((ys - ys_pred).square()) / (range_vals + 1e-10)
#     mse = normalized_error.mean()
#     return mse

""" lastindex normalized_range_mse """
# def mean_squared_error(ys_pred, ys):
#     last_ys_pred = ys_pred[:, -1]
#     last_ys = ys[:, -1]
#     range_vals = last_ys.max() - last_ys.min()
#     squared_error = (last_ys_pred - last_ys).square()
#     normalized_error = squared_error / (range_vals + 1e-10)
#     mse = normalized_error.mean()
#     return mse


""" mean_absolute_percentage_error """
# def mean_squared_error(ys_pred, ys):
#     epsilon = 1e-10  
#     mape = torch.abs((ys - ys_pred) / (ys + epsilon)).mean()
#     return mape


""" relative_squared_error """
# def mean_squared_error(ys_pred, ys):
#     epsilon = 1e-8  
#     rse = ((ys - ys_pred) / (ys + epsilon)).square().mean()
#     return rse


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


# class LinearRegression(Task):
#     def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
#         """scale: a constant by which to scale the randomly sampled weights."""
#         super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
#         self.scale = scale

#         if pool_dict is None and seeds is None:
#             self.w_b = torch.randn(self.b_size, self.n_dims, 1)
#         elif seeds is not None:
#             self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
#             generator = torch.Generator()
#             assert len(seeds) == self.b_size
#             for i, seed in enumerate(seeds):
#                 generator.manual_seed(seed)
#                 self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
#         else:
#             assert "w" in pool_dict
#             indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
#             self.w_b = pool_dict["w"][indices]

#     def evaluate(self, xs_b):
#         w_b = self.w_b.to(xs_b.device)
#         ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
#         return ys_b

#     @staticmethod
#     def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
#         return {"w": torch.randn(num_tasks, n_dims, 1)}

#     @staticmethod
#     def get_metric():
#         return squared_error

#     @staticmethod
#     def get_training_metric():
#         return mean_squared_error


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds, scale)
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds, scale)
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(self.n_dims, hidden_layer_size, generator=generator)
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(self.dt_tensor[i].shape, generator=generator)
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
