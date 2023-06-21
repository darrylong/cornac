import numpy as np
from tqdm.auto import tqdm

from ..recommender import Recommender
from ...utils import get_rng
from ...utils.init_utils import xavier_uniform

class GCMC(Recommender):
    """
    Parameters
    ----------
    k: int, optional, default: 50
        The dimension of the user and item latent factors.
    References
    ----------
    * To be added
    """

    def __init__(
        self,
        name="GCMC",
        k=50,
        use_gpu=False,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        super.__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.name = name
        self.seed = seed

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)
        self.V = self.init_params.get("V", None)

    def _init(self, n_users, n_items):
        rng = get_rng(self.seed)
        # n_users, n_items = self.train_set.num_users, self.train_set.num_items

        if self.U is None:
            self.U = xavier_uniform((n_users, self.k), rng)
        if self.V is None:
            self.V = xavier_uniform((n_items, self.k), rng)


    def fit(self, train_set, val_set=None):
        """Fit the model to observations.
        
        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.
        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        self._init(n_users=train_set.total_users, n_items=train_set.total_items)

        if self.trainable:
            self._fit_torch()

        return self

    def _fit_torch(self):
        import torch
        from .gcmc import GCEncoder

        dtype = torch.float
        device = {
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        }



        user_data = self._build_data(self.train_set.matrix)
        item_data = self._build_data(self.train_set.matrix.T.tocsr())

        n_user = len(user_data[0])
        n_item = len(item_data[0])

        R_user = user_data[1]
        R_item = item_data[1]

        optimizer = torch.optim.Adam(learning_rate=self.learning_rate)


        raise NotImplementedError
    